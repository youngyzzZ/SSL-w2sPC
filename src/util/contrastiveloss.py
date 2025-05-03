import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityContrastiveLoss(nn.Module):
    """
    Contrastive loss function based on cosine similarity

    Parameters:
    - margin: Boundary value to distinguish positive and negative samples
    - temperature: Temperature parameter to control the smoothness of distribution
    """

    def __init__(self, margin=0.5, temperature=0.07):
        super(CosineSimilarityContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, anchor, positive, negative=None):
        """
        Calculate contrastive loss

        Parameters:
        - anchor: Anchor sample features, shape [batch_size, feature_dim]
        - positive: Positive sample features, shape [batch_size, feature_dim]
        - negative: Negative sample features, shape [batch_size, feature_dim] or [batch_size, num_negatives, feature_dim]
                   If None, other samples in the batch will be used as negative samples

        Returns:
        - loss: Contrastive loss value
        """
        # Feature normalization
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)

        # Calculate cosine similarity between anchor and positive samples
        pos_sim = F.cosine_similarity(anchor_norm, positive_norm, dim=1)

        if negative is not None:
            negative_norm = F.normalize(negative, p=2, dim=-1)

            # Multiple negative samples case
            if negative_norm.dim() == 3:
                # Calculate cosine similarity between anchor and all negative samples
                anchor_norm_exp = anchor_norm.unsqueeze(
                    1)  # [batch_size, 1, feature_dim]
                # [batch_size, num_negatives]
                neg_sim = F.cosine_similarity(
                    anchor_norm_exp, negative_norm, dim=2)

                # For each anchor sample, select the hardest negative sample (most similar)
                neg_sim = neg_sim.max(dim=1)[0]  # [batch_size]
            else:  # Single negative sample case
                neg_sim = F.cosine_similarity(
                    anchor_norm, negative_norm, dim=1)

            # Calculate contrastive loss with margin
            losses = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
            loss = losses.mean()

        # Use other samples in the batch as negative samples (InfoNCE loss form)
        else:
            batch_size = anchor_norm.size(0)

            # Calculate similarity matrix between all sample pairs in the batch
            similarity_matrix = torch.matmul(
                anchor_norm, positive_norm.transpose(0, 1)) / self.temperature

            # Elements on the diagonal are positive sample pairs
            labels = torch.arange(batch_size).to(anchor.device)

            # Calculate InfoNCE loss
            loss = F.cross_entropy(similarity_matrix, labels)

        return loss


# Usage example
def example_usage():
    # Create loss function instance
    criterion = CosineSimilarityContrastiveLoss(margin=0.5, temperature=0.07)

    # Simulate batch data
    batch_size, feature_dim = 16, 128

    # Create anchor, positive and negative samples
    anchor = torch.randn(batch_size, feature_dim)
    # Positive samples matching the anchors
    positive = torch.randn(batch_size, feature_dim)
    # Non-matching negative samples
    negative = torch.randn(batch_size, feature_dim)

    # Calculate loss with explicit negative samples
    loss1 = criterion(anchor, positive, negative)
    print(f"Loss with explicit negative samples: {loss1.item()}")

    # Calculate loss using other samples in the batch as negatives
    loss2 = criterion(anchor, positive)
    print(f"Loss with in-batch negatives: {loss2.item()}")

    # Multiple negative samples case
    num_negatives = 5
    multiple_negatives = torch.randn(batch_size, num_negatives, feature_dim)
    loss3 = criterion(anchor, positive, multiple_negatives)
    print(f"Loss with multiple negative samples: {loss3.item()}")


if __name__ == '__main__':
    example_usage()
