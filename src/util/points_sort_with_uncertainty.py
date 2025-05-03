import torch
import torch.nn.functional as F
import numpy as np


def extract_features_by_class_and_uncertainty(segmentation_output, feature_maps, low_percent=0.1, high_percent=0.1):
    """
    Extract feature vectors corresponding to the pixels with highest and lowest uncertainty percentages for each class

    Parameters:
    segmentation_output: Segmentation output with shape (batch_size, num_classes, height, width)
    feature_maps: Feature maps with shape (batch_size, feature_dim, height, width)
    low_percent: Percentage of pixels with lowest uncertainty to extract (default: 0.1)
    high_percent: Percentage of pixels with highest uncertainty to extract (default: 0.1)

    Returns:
    Dictionary containing features of pixels with highest and lowest uncertainty for each class.
    low_uncertainty_features are sorted from lowest to highest uncertainty.
    high_uncertainty_features are sorted from highest to lowest uncertainty.
    """
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(segmentation_output, dim=1)

    # Calculate entropy for each pixel
    epsilon = 1e-10
    log_probs = torch.log(probabilities + epsilon)
    entropy = -torch.sum(probabilities * log_probs, dim=1)  # Shape: (batch_size, height, width)

    # Get the predicted class for each pixel (class with maximum probability)
    _, predicted_classes = torch.max(probabilities, dim=1)  # Shape: (batch_size, height, width)

    batch_size = segmentation_output.shape[0]
    num_classes = segmentation_output.shape[1]  # Number of classes, here is 4
    h, w = segmentation_output.shape[2], segmentation_output.shape[3]
    feature_dim = feature_maps.shape[1]  # Feature dimension, here is 16

    # Prepare to store results
    results = []

    for batch_idx in range(batch_size):
        batch_entropy = entropy[batch_idx]  # Shape: (height, width)
        batch_classes = predicted_classes[batch_idx]  # Shape: (height, width)
        batch_features = feature_maps[batch_idx]  # Shape: (feature_dim, height, width)

        # Transpose feature maps to (height, width, feature_dim) for easier indexing
        features_hwc = batch_features.permute(1, 2, 0)  # Shape: (height, width, feature_dim)

        # Process each class separately
        class_results = {}

        for class_idx in range(num_classes):
            # Get mask for pixels belonging to current class
            class_mask = (batch_classes == class_idx)  # Shape: (height, width)

            # If no pixels belong to this class, continue to next class
            if not torch.any(class_mask):
                class_results[class_idx] = {
                    'low_uncertainty_coords': [],
                    'high_uncertainty_coords': [],
                    'low_uncertainty_entropy': torch.tensor([]),
                    'high_uncertainty_entropy': torch.tensor([]),
                    'low_uncertainty_features': torch.zeros((0, feature_dim)),
                    'high_uncertainty_features': torch.zeros((0, feature_dim))
                }
                continue

            # Get coordinates and entropy values for all pixels in current class
            class_coords = torch.nonzero(class_mask, as_tuple=True)  # Shape: (2, num_class_pixels)
            class_entropy = batch_entropy[class_mask]  # Shape: (num_class_pixels)

            # Sort entropy values within current class
            class_sorted_indices = torch.argsort(class_entropy)
            num_class_pixels = class_entropy.size(0)

            # Calculate the number of pixels to select for low and high uncertainty
            low_class_pixels = max(1, int(low_percent * num_class_pixels))  # Take at least 1 pixel
            high_class_pixels = max(1, int(high_percent * num_class_pixels))  # Take at least 1 pixel

            # Get indices of pixels with lowest and highest uncertainty in current class
            class_low_indices = class_sorted_indices[:low_class_pixels]  # Indices for lowest uncertainty pixels
            class_high_indices = class_sorted_indices[-high_class_pixels:]  # Indices for highest uncertainty pixels

            # We want high_uncertainty indices to be sorted from highest to lowest uncertainty
            class_high_indices = class_high_indices.flip(0)  # Reverse the order to get highest first

            # Get coordinates of these pixels in original image
            class_low_rows = class_coords[0][class_low_indices].cpu().numpy()
            class_low_cols = class_coords[1][class_low_indices].cpu().numpy()
            class_low_coords = list(zip(class_low_rows, class_low_cols))

            class_high_rows = class_coords[0][class_high_indices].cpu().numpy()
            class_high_cols = class_coords[1][class_high_indices].cpu().numpy()
            class_high_coords = list(zip(class_high_rows, class_high_cols))

            # Extract feature vectors for these coordinates
            class_low_features = features_hwc[class_low_rows, class_low_cols]  # Shape: (low_class_pixels, feature_dim)
            class_high_features = features_hwc[
                class_high_rows, class_high_cols]  # Shape: (high_class_pixels, feature_dim)

            class_results[class_idx] = {
                'low_uncertainty_coords': class_low_coords,
                'high_uncertainty_coords': class_high_coords,
                'low_uncertainty_entropy': class_entropy[class_low_indices],  # Sorted from lowest to highest
                'high_uncertainty_entropy': class_entropy[class_high_indices],  # Sorted from highest to lowest
                'low_uncertainty_features': class_low_features,  # Sorted from lowest to highest uncertainty
                'high_uncertainty_features': class_high_features  # Sorted from highest to lowest uncertainty
            }

        results.append({
            'batch_idx': batch_idx,
            'class_results': class_results
        })

    return results


def extract_top_features(results, num_features=6):
    """
    Extract the top N features with highest and lowest uncertainty from each class

    Parameters:
    results: Results dictionary returned by extract_features_by_class_and_uncertainty function
    num_features: Number of top features to extract from each category (default: 6)

    Returns:
    Dictionary containing extracted top features for each batch and class
    """
    extracted_features = {}

    for batch_idx, batch_result in enumerate(results):
        batch_features = {'batch_idx': batch_idx}
        class_features = {}

        for class_idx, class_result in batch_result['class_results'].items():
            # Skip if the class has no pixels
            if len(class_result['low_uncertainty_coords']) == 0 or len(class_result['high_uncertainty_features']) == 0:
                continue

            # Extract features with lowest uncertainty (already sorted from lowest to highest)
            # Take the first num_features (or all if less than num_features)
            num_low = min(num_features, len(class_result['low_uncertainty_features']))
            low_features = class_result['low_uncertainty_features'][:num_low]
            low_entropy = class_result['low_uncertainty_entropy'][:num_low]
            low_coords = class_result['low_uncertainty_coords'][:num_low]

            # Extract features with highest uncertainty (already sorted from highest to lowest)
            # Take the first num_features (or all if less than num_features)
            num_high = min(num_features, len(class_result['high_uncertainty_features']))
            high_features = class_result['high_uncertainty_features'][:num_high]
            high_entropy = class_result['high_uncertainty_entropy'][:num_high]
            high_coords = class_result['high_uncertainty_coords'][:num_high]

            # Store features for this class
            class_features[class_idx] = {
                'low_uncertainty_features': low_features,
                'low_uncertainty_entropy': low_entropy,
                'low_uncertainty_coords': low_coords,
                'high_uncertainty_features': high_features,
                'high_uncertainty_entropy': high_entropy,
                'high_uncertainty_coords': high_coords,
                'num_low_features': num_low,
                'num_high_features': num_high
            }

        batch_features['class_features'] = class_features
        extracted_features[batch_idx] = batch_features

    return extracted_features

# Usage example
if __name__ == "__main__":
    # Simulated data
    batch_size = 4
    num_classes = 4
    feature_dim = 16
    height, width = 256, 256

    # Simulate segmentation output and feature maps
    segmentation_output = torch.randn(batch_size, num_classes, height, width)
    feature_maps = torch.randn(batch_size, feature_dim, height, width)

    # Extract features with custom percentages
    results = extract_features_by_class_and_uncertainty(
        segmentation_output,
        feature_maps,
        low_percent=0.1,  # Get lowest 10% uncertain pixels per class
        high_percent=0.1  # Get highest 10% uncertain pixels per class
    )

    # Print results for first batch
    print(f"Batch #{results[0]['batch_idx']} results:")

    # Print by class
    for class_idx in range(num_classes):
        class_result = results[0]['class_results'][class_idx]
        num_low = len(class_result['low_uncertainty_coords'])
        num_high = len(class_result['high_uncertainty_coords'])

        print(f"\nClass {class_idx} results:")
        print(f"Number of pixels with lowest uncertainty in this class: {num_low}")
        if num_low > 0:
            print(
                f"Feature shape of pixels with lowest uncertainty in this class: {class_result['low_uncertainty_features'].shape}")
            print(f"First pixel coordinates: {class_result['low_uncertainty_coords'][0]}")
            print(
                f"First pixel entropy value: {class_result['low_uncertainty_entropy'][0].item()} (lowest uncertainty)")
            print(
                f"Last pixel entropy value: {class_result['low_uncertainty_entropy'][-1].item()} (higher uncertainty)")

        print(f"Number of pixels with highest uncertainty in this class: {num_high}")
        if num_high > 0:
            print(
                f"Feature shape of pixels with highest uncertainty in this class: {class_result['high_uncertainty_features'].shape}")
            print(f"First pixel coordinates: {class_result['high_uncertainty_coords'][0]}")
            print(
                f"First pixel entropy value: {class_result['high_uncertainty_entropy'][0].item()} (highest uncertainty)")
            print(
                f"Last pixel entropy value: {class_result['high_uncertainty_entropy'][-1].item()} (lower uncertainty)")