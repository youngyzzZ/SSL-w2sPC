import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import networks.net_factory as net_factory
# from networks.net_factory import net_factory
from util import ramps, contrastiveloss, points_sort_with_uncertainty
from util.contrastiveloss import CosineSimilarityContrastiveLoss
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.unet import UNet_W2S
from util.points_sort_with_uncertainty import extract_features_by_class_and_uncertainty, extract_top_features

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Weak_to_Strong_Perturbation', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1268, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
# parser.add_argument('--num_classes', type=int, default=2,
#                     help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Not Implement Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    """
    Train on ACDC with weak-to-strong multi-branch outputs (UNet_W2S).
    Assumptions:
      - model(x) returns a list of branches; each branch is (seg_logits, cont_feat)
      - seg_logits: [B, C, H, W], cont_feat: [B, D, H, W]
      - BaseDataSets with TwoStreamBatchSampler yields mixed labeled/unlabeled in one batch
    """
    # reproducibility & cuDNN
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # build model
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    model.train()

    # dataloaders
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    # sanity for labeled_bs
    labeled_bs = min(args.labeled_bs, batch_size)
    unlabeled_bs = batch_size - labeled_bs
    if unlabeled_bs <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > labeled_bs ({args.labeled_bs}).")

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # optimizer & losses
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    ce_loss = CrossEntropyLoss()
    contrastive_loss_fn = CosineSimilarityContrastiveLoss(margin=0.5, temperature=0.07)

    # logging / snapshot
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(str(Path(snapshot_path) / 'log'))
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_metric = 0.0

    iterator = tqdm(range(max_epoch), ncols=90)
    model.train()

    def dice_per_class(pred, target, num_cls):
        # pred: [B,H,W] int, target: [B,H,W] int
        dices = []
        for c in range(num_cls):
            pred_c = (pred == c).float()
            targ_c = (target == c).float()
            inter = (pred_c * targ_c).sum()
            union = pred_c.sum() + targ_c.sum()
            dice = (2 * inter + 1e-5) / (union + 1e-5)
            dices.append(dice.item())
        return np.mean(dices)

    def consistency_loss_from_branches(seg_list):
        """
        seg_list: list of logits [B,C,H,W] from all branches
        We encourage each branch prob to stay close to the mean prob (L2).
        """
        probs = [F.softmax(s, dim=1) for s in seg_list]
        mean_p = torch.stack(probs, dim=0).mean(dim=0)
        loss_terms = [(p - mean_p).pow(2).mean() for p in probs]
        return sum(loss_terms) / len(loss_terms)

    # (edge-aware contrast removed; we now use CosineSimilarityContrastiveLoss from util.contrastiveloss)
    #
    # training
    #
    global_step = 0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            optimizer.zero_grad()
            outputs = model(volume_batch)                 # list of branches
            # unpack segmentation logits and contrast features
            seg_list = [o[0] for o in outputs]            # each: [B,C,H,W]
            feat_list = [o[1] for o in outputs]           # each: [B,D,H,W]

            # supervised loss on labeled part (first labeled_bs samples)
            sup_loss = ce_loss(seg_list[0][:labeled_bs], label_batch[:labeled_bs])

            # consistency loss (all samples, all branches)
            cons_w = get_current_consistency_weight(epoch_num)
            cons_loss = consistency_loss_from_branches(seg_list) * cons_w

            # contrastive loss (cosine-sim based); pool spatial dims to [B, D]
            if len(feat_list) >= 2:
                anchor_feat = feat_list[0].mean(dim=(2, 3))  # [B, D]
                positive_feat = feat_list[1].mean(dim=(2, 3))  # [B, D]
                if len(feat_list) >= 3:
                    negative_feat = feat_list[2].mean(dim=(2, 3))  # [B, D]
                    ctr_loss = contrastive_loss_fn(anchor_feat, positive_feat, negative_feat)
                else:
                    ctr_loss = contrastive_loss_fn(anchor_feat, positive_feat)
            else:
                ctr_loss = torch.tensor(0.0, device=volume_batch.device)

            total_loss = sup_loss + cons_loss + 0.5 * ctr_loss
            total_loss.backward()
            optimizer.step()

            iter_num += 1
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar('train/supervised', sup_loss.item(), global_step)
                writer.add_scalar('train/consistency', cons_loss.item(), global_step)
                writer.add_scalar('train/contrast', ctr_loss.item(), global_step)
                writer.add_scalar('train/total', total_loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            # simple poly lr decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = max(lr_, 1e-5)

            if iter_num % 200 == 0 or iter_num == 1:
                # quick validation: mean dice over classes (including background)
                model.eval()
                dices = []
                with torch.no_grad():
                    for valb in valloader:
                        img, lab = valb['image'].cuda(), valb['label'].cuda()
                        out = model(img)[0][0]  # main branch logits
                        pred = torch.argmax(out, dim=1)                # [B,H,W]
                        dices.append(dice_per_class(pred, lab.squeeze(1) if lab.ndim==4 else lab, num_classes))
                mean_dice = float(np.mean(dices)) if dices else 0.0
                writer.add_scalar('val/mean_dice', mean_dice, global_step)

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    save_path = os.path.join(snapshot_path, f"best_iter_{iter_num}_dice_{best_metric:.4f}.pth")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Saved best model to {save_path}")

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    # final save
    final_path = os.path.join(snapshot_path, "final.pth")
    torch.save(model.state_dict(), final_path)
    logging.info(f"Training finished. Final model saved to {final_path}")


if __name__ == '__main__':
    exp_dir = Path(args.exp)
    if exp_dir.is_absolute():
        snapshot_path = str(exp_dir)
    else:
        snapshot_path = str((Path("runs") / exp_dir).resolve())
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.info(f"Save dir: {snapshot_path}")
    logging.info(str(args))

    train(args, snapshot_path)
