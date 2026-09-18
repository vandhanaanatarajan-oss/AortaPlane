cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants

cat > m1_nnlandmark.py << 'ARCHEOF'
"""
M1_nnLandmark
--------------
M1's direct 6D-regression head, with nnLandmark's own real internal
network (PlainConvEncoder, from the dynamic_network_architectures
package nnU-Net/nnLandmark is actually built on) as the encoder,
reconfigured for M1/M2's fixed 64^3 input -- nnLandmark's own native
config is built for full-resolution variable-size CT scans, which
doesn't fit this pipeline, so this uses a downsampling schedule
matching M1/M2's other encoders (5 stride-2 stages, 64 -> 2), while
keeping nnLandmark's REAL, characteristic building-block choices:
InstanceNorm3d, LeakyReLU, conv_bias=True -- confirmed from
nnLandmark's actual plans.json (nnUNetPlansV1Match, seen directly on
Apocrita earlier this project) -- not guessed defaults.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

N_STAGES = 6
FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]  # exact match to nnLandmark's real plans.json channel progression
STRIDES = [1, 2, 2, 2, 2, 2]  # stage 0 stays at full input res (nnU-Net's real convention, confirmed in plans.json), enabling the decoder to reconstruct back to 64^3 via skip connections


class M1_nnLandmark(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.encoder = PlainConvEncoder(
            input_channels=1,
            n_stages=N_STAGES,
            features_per_stage=FEATURES_PER_STAGE,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=STRIDES,
            n_conv_per_stage=2,
            conv_bias=True,  # confirmed from nnLandmark's real plans.json
            norm_op=nn.InstanceNorm3d,  # confirmed real choice, not M1/M2's own BatchNorm3d
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,  # confirmed real choice, not M1/M2's own ReLU
            nonlin_kwargs={'inplace': True},
            return_skips=False,  # M1-style has no decoder, so no skip connections needed
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(FEATURES_PER_STAGE[-1], 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        feat = self.encoder(x)  # final feature map, (B, 320, 2, 2, 2)
        out = self.gap(feat).view(feat.size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = M1_nnLandmark()
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"output shape: {out.shape} (expect [2, 6])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 6)
    print("PASS")
ARCHEOF

cat > m2_nnlandmark.py << 'ARCHEOF'
"""
M2_nnLandmark
--------------
M2's landmark-then-plane approach, using nnLandmark's own REAL, COMPLETE
network -- PlainConvUNet, encoder AND decoder together, not just a
backbone -- the most faithful of all 6 variants, since it reuses both
halves of nnLandmark's actual architecture, just reconfigured for
M1/M2's fixed 64^3 input and with num_classes=3 (matching M2's 3
heatmap channels: RC/NC/LC) instead of nnLandmark's own segmentation
output.

Same real, confirmed building-block choices as M1_nnLandmark:
InstanceNorm3d, LeakyReLU, conv_bias=True (from nnLandmark's actual
plans.json). deep_supervision=False, since none of the other 5
variants use multi-scale supervision either -- keeps this comparable.
"""
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from shared_blocks import soft_argmax_3d

N_STAGES = 6
FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]  # exact match to nnLandmark's real plans.json
STRIDES = [1, 2, 2, 2, 2, 2]  # stage 0 stays at full input res -- needed for the decoder to reconstruct back to 64^3


class M2_nnLandmark(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        # num_classes=3 -> nnLandmark's own decoder outputs 3 channels
        # directly (one per landmark: RC/NC/LC) -- no separate head
        # module needed, unlike the other M2-style variants, since
        # PlainConvUNet's decoder already produces exactly this shape.
        self.net = PlainConvUNet(
            input_channels=1,
            n_stages=N_STAGES,
            features_per_stage=FEATURES_PER_STAGE,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=STRIDES,
            n_conv_per_stage=2,
            num_classes=3,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        heatmaps = self.net(x)  # (B, 3, 64, 64, 64) -- nnLandmark's decoder upsamples all the way back to input resolution natively

        rc = soft_argmax_3d(heatmaps[:, 0:1])
        nc = soft_argmax_3d(heatmaps[:, 1:2])
        lc = soft_argmax_3d(heatmaps[:, 2:3])

        landmarks = torch.cat([rc, nc, lc], dim=1)
        return landmarks, heatmaps


if __name__ == "__main__":
    model = M2_nnLandmark()
    x = torch.randn(2, 1, 64, 64, 64)
    landmarks, heatmaps = model(x)
    print(f"landmarks shape: {landmarks.shape} (expect [2, 9])")
    print(f"heatmaps shape:  {heatmaps.shape} (expect [2, 3, 64, 64, 64])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert landmarks.shape == (2, 9)
    assert heatmaps.shape == (2, 3, 64, 64, 64)
    print("PASS")
ARCHEOF

cd /data/DERI-ecgai/__users/Vandhanaa

cat > train_m1_variants.py << 'ARCHEOF'
#!/usr/bin/env python3
"""
train_m1_variants.py
---------------------
Trains M1-style (direct 6D regression) architecture variants -- M1_UNet
or M1_DenseNet3D -- reusing the EXACT real data loading, loss function,
optimizer, scheduler, and training loop mechanics as train.py (verified
against train.py's actual source, not guessed), so results are
genuinely comparable to M1 itself. Only the model class differs.

All hyperparameters below are copied from the REAL M1_fold0.sh job
script (confirmed on Apocrita, 9 Aug 2026) -- not defaults or guesses.

Usage (from /data/DERI-ecgai/__users/Vandhanaa/, same directory train.py
itself runs from -- needed for the `from src...` imports to resolve):
    python train_m1_variants.py --arch unet     --fold 0 --run_name M1_unet_fold0
    python train_m1_variants.py --arch densenet3d --fold 0 --run_name M1_densenet3d_fold0

Run once per fold (0-4) for a full 5-fold comparison, matching M1's own
5-fold CV setup.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import create_data_loaders
from src.loss import PlaneLoss, angle_error_degrees, centre_distance_voxels

import sys
sys.path.insert(0, "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants")  # CHECK: adjust if you placed the variant files elsewhere
from m1_unet import M1_UNet
from m1_densenet3d import M1_DenseNet3D
from m1_nnlandmark import M1_nnLandmark


def parse_args():
    p = argparse.ArgumentParser(description="Train M1-style architecture variant")
    p.add_argument('--arch', type=str, required=True, choices=['unet', 'densenet3d', 'nnlandmark'])
    p.add_argument('--fold', type=int, required=True, help='Fold index 0-4 from fold_config.json')
    p.add_argument('--fold_config', type=str, default='fold_config.json')
    p.add_argument('--root_path', type=str,
                   default='/data/DERI-ecgai/__users/Vandhanaa/processed_output')
    p.add_argument('--manifest_path', type=str, default=None)
    p.add_argument('--target_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--w_normal', type=float, default=1.0)
    p.add_argument('--w_centre', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', type=str, required=True)
    p.add_argument('--save_dir', type=str, default=None)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_angle, total_centre, n_batches = 0.0, 0.0, 0.0, 0
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss, components = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += components['total']
        total_angle += components['angular']
        total_centre += components['centre']
        n_batches += 1
    return {'loss': total_loss / n_batches, 'angular': total_angle / n_batches,
            'centre_mse': total_centre / n_batches}


@torch.no_grad()
def validate(model, loader, loss_fn, device, image_size):
    model.eval()
    total_loss, total_angle, total_centre, n_batches = 0.0, 0.0, 0.0, 0
    angle_errors, centre_dists = [], []
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss, components = loss_fn(preds, targets)
        ang = angle_error_degrees(preds[:, :3], targets[:, :3])
        dist = centre_distance_voxels(preds[:, 3:], targets[:, 3:], image_size)
        angle_errors.extend(ang.cpu().numpy().tolist())
        centre_dists.extend(dist.cpu().numpy().tolist())
        total_loss += components['total']
        total_angle += components['angular']
        total_centre += components['centre']
        n_batches += 1
    return {
        'loss': total_loss / n_batches, 'angular': total_angle / n_batches,
        'centre_mse': total_centre / n_batches,
        'angle_error_mean': float(np.mean(angle_errors)), 'angle_error_std': float(np.std(angle_errors)),
        'centre_dist_mean': float(np.mean(centre_dists)), 'centre_dist_std': float(np.std(centre_dists)),
    }


def main():
    args = parse_args()

    with open(args.fold_config) as f:
        fc = json.load(f)
    fold_data = fc['folds'][args.fold]
    train_ids, val_ids = fold_data['train'], fold_data['val']
    print(f'Using fold {args.fold}: train={len(train_ids)} val={len(val_ids)}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.save_dir or f'checkpoints/{args.run_name}'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_data_loaders(
        args.root_path, train_ids, val_ids, val_ids,
        target_size=args.target_size, batch_size=args.batch_size,
        manifest_path=args.manifest_path,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.arch == 'unet':
        model = M1_UNet(dropout_p=args.dropout).to(device)
    elif args.arch == 'densenet3d':
        model = M1_DenseNet3D(dropout_p=args.dropout).to(device)
    elif args.arch == 'nnlandmark':
        model = M1_nnLandmark(dropout_p=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    loss_fn = PlaneLoss(w_normal=args.w_normal, w_centre=args.w_centre)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device, args.target_size)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"angle_err={val_metrics['angle_error_mean']:.3f}+/-{val_metrics['angle_error_std']:.3f}  "
              f"centre_err={val_metrics['centre_dist_mean']:.3f}+/-{val_metrics['centre_dist_std']:.3f}  "
              f"({elapsed:.1f}s)")

        history.append({'epoch': epoch + 1, **{f'train_{k}': v for k, v in train_metrics.items()},
                         **{f'val_{k}': v for k, v in val_metrics.items()}})

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'{args.run_name}_best.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, f'{args.run_name}_last.pth'))
    with open(os.path.join(save_dir, f'{args.run_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Final val: angle_error={history[-1]['val_angle_error_mean']:.3f}, "
          f"centre_error={history[-1]['val_centre_dist_mean']:.3f}")


if __name__ == "__main__":
    main()
ARCHEOF

cat > train_m2_variants.py << 'ARCHEOF'
#!/usr/bin/env python3
"""
train_m2_variants.py
---------------------
Trains M2-style (landmarks-via-heatmaps -> plane) architecture variants
-- M2_ResNet3D_NoSkip or M2_DenseNet3D -- reusing the EXACT real data
loading, loss function, optimizer, scheduler, and training loop
mechanics as train_m2.py (verified against train_m2.py's actual source,
not guessed). Only the model class differs.

All hyperparameters copied from the REAL M2_fold0.sh job script plus
train_m2.py's own parse_args defaults (confirmed on Apocrita, 9 Aug 2026).

Key differences from the M1-variant script, both confirmed real:
  - test_ids comes from fold_config.json's top-level 'test_patients'
    field (the real fixed locked test set), NOT a val_ids fallback.
  - Validation converts landmarks -> plane via landmarks_to_plane()
    before computing angle/centre error, since these models output
    9D landmarks, not a 6D plane directly.
  - LandmarkPlaneLoss takes a third weight, w_spread (default 0.1),
    that PlaneLoss doesn't have.

Usage (run from anywhere -- sys.path.insert below matches train_m2.py's
own pattern so the src.* imports resolve regardless of cwd):
    python train_m2_variants.py --arch resnet3d_noskip --fold 0 --run_name M2_resnet3d_noskip_fold0
    python train_m2_variants.py --arch densenet3d       --fold 0 --run_name M2_densenet3d_fold0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants')  # CHECK: adjust if you placed the variant files elsewhere

from src.dataset import create_data_loaders
from src.model import landmarks_to_plane
from src.loss import LandmarkPlaneLoss, angle_error_degrees, centre_distance_voxels

from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_densenet3d import M2_DenseNet3D
from m2_nnlandmark import M2_nnLandmark


def parse_args():
    p = argparse.ArgumentParser(description="Train M2-style architecture variant")
    p.add_argument('--arch', type=str, required=True, choices=['resnet3d_noskip', 'densenet3d', 'nnlandmark'])
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--fold_config', type=str, default='fold_config.json')
    p.add_argument('--root_path', type=str,
                   default='/data/DERI-ecgai/__users/Vandhanaa/processed_output')
    p.add_argument('--target_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--w_normal', type=float, default=1.0)
    p.add_argument('--w_centre', type=float, default=1.0)
    p.add_argument('--w_spread', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--run_name', type=str, required=True)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        landmarks, _ = model(imgs)
        loss, _ = loss_fn(landmarks, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device, image_size):
    model.eval()
    total_loss, n_batches = 0.0, 0
    angle_errors, centre_dists = [], []
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        landmarks, _ = model(imgs)
        loss, _ = loss_fn(landmarks, targets)
        total_loss += loss.item()
        n_batches += 1
        plane = landmarks_to_plane(landmarks)
        ang = angle_error_degrees(plane[:, :3], targets[:, :3])
        dist = centre_distance_voxels(plane[:, 3:], targets[:, 3:], image_size)
        angle_errors.extend(ang.cpu().numpy().tolist())
        centre_dists.extend(dist.cpu().numpy().tolist())
    return {
        'loss': total_loss / n_batches,
        'angle_error_mean': float(np.mean(angle_errors)),
        'angle_error_std': float(np.std(angle_errors)),
        'centre_dist_mean': float(np.mean(centre_dists)),
        'centre_dist_std': float(np.std(centre_dists)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.fold_config) as f:
        fc = json.load(f)
    fold_data = fc['folds'][args.fold]
    train_ids, val_ids = fold_data['train'], fold_data['val']
    test_ids = fc['test_patients']  # real fixed locked test set, matches train_m2.py exactly

    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  {args.run_name}')
    print(f'  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}')
    print(f'{"="*60}\n')

    train_loader, val_loader, test_loader = create_data_loaders(
        args.root_path, train_ids, val_ids, test_ids,
        target_size=args.target_size, batch_size=args.batch_size,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.arch == 'resnet3d_noskip':
        model = M2_ResNet3D_NoSkip(dropout_p=args.dropout).to(device)
    elif args.arch == 'densenet3d':
        model = M2_DenseNet3D(dropout_p=args.dropout).to(device)
    elif args.arch == 'nnlandmark':
        model = M2_nnLandmark(dropout_p=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Device: {device}  |  Parameters: {total_params:,}')

    loss_fn = LandmarkPlaneLoss(w_normal=args.w_normal, w_centre=args.w_centre, w_spread=args.w_spread)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device, args.target_size)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"angle_err={val_metrics['angle_error_mean']:.3f}+/-{val_metrics['angle_error_std']:.3f}  "
              f"centre_err={val_metrics['centre_dist_mean']:.3f}+/-{val_metrics['centre_dist_std']:.3f}  "
              f"({elapsed:.1f}s)")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss, **val_metrics})

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'{args.run_name}_best.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, f'{args.run_name}_last.pth'))
    with open(os.path.join(save_dir, f'{args.run_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Final val: angle_error={history[-1]['angle_error_mean']:.3f}, "
          f"centre_error={history[-1]['centre_dist_mean']:.3f}")


if __name__ == "__main__":
    main()
ARCHEOF

cat > evaluate_arch_variants_testset.py << 'ARCHEOF'
#!/usr/bin/env python3
"""
evaluate_arch_variants_testset.py
-----------------------------------
Loads each of the 4 finished architecture variants' saved checkpoints
(*_best.pth, already trained -- no retraining here) and evaluates them
on the TRUE locked test set from fold_config.json['test_patients'],
using the exact same evaluation logic (angle_error_degrees,
centre_distance_voxels, and for M2-style models, landmarks_to_plane)
as the real train.py/train_m2.py validate() functions -- just run once,
on the test set, instead of during training on the val set.

This fixes the val-vs-test gap in the original training scripts: neither
train_m1_variants.py nor train_m2_variants.py ran a final test-set pass,
only validation during training.

Usage (from /data/DERI-ecgai/__users/Vandhanaa/):
    python evaluate_arch_variants_testset.py --fold 0
"""
import argparse
import json

import numpy as np
import torch

import sys
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants')

from src.dataset import create_data_loaders
from src.model import landmarks_to_plane
from src.loss import angle_error_degrees, centre_distance_voxels

from m1_unet import M1_UNet
from m1_densenet3d import M1_DenseNet3D
from m1_nnlandmark import M1_nnLandmark
from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_densenet3d import M2_DenseNet3D
from m2_nnlandmark import M2_nnLandmark


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--fold_config', type=str, default='fold_config.json')
    p.add_argument('--root_path', type=str,
                   default='/data/DERI-ecgai/__users/Vandhanaa/processed_output')
    p.add_argument('--target_size', type=int, default=64)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return p.parse_args()


@torch.no_grad()
def evaluate_m1_style(model, loader, device, image_size):
    model.eval()
    angle_errors, centre_dists = [], []
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        ang = angle_error_degrees(preds[:, :3], targets[:, :3])
        dist = centre_distance_voxels(preds[:, 3:], targets[:, 3:], image_size)
        angle_errors.extend(ang.cpu().numpy().tolist())
        centre_dists.extend(dist.cpu().numpy().tolist())
    return angle_errors, centre_dists


@torch.no_grad()
def evaluate_m2_style(model, loader, device, image_size):
    model.eval()
    angle_errors, centre_dists = [], []
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        landmarks, _ = model(imgs)
        plane = landmarks_to_plane(landmarks)
        ang = angle_error_degrees(plane[:, :3], targets[:, :3])
        dist = centre_distance_voxels(plane[:, 3:], targets[:, 3:], image_size)
        angle_errors.extend(ang.cpu().numpy().tolist())
        centre_dists.extend(dist.cpu().numpy().tolist())
    return angle_errors, centre_dists


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.fold_config) as f:
        fc = json.load(f)
    fold_data = fc['folds'][args.fold]
    train_ids, val_ids = fold_data['train'], fold_data['val']
    test_ids = fc['test_patients']  # the real, fixed, locked test set

    print(f"True locked test set: {test_ids}\n")

    _, _, test_loader = create_data_loaders(
        args.root_path, train_ids, val_ids, test_ids,
        target_size=args.target_size, batch_size=2,
    )

    variants = [
        ("M1_unet", "m1_style", M1_UNet, f"{args.checkpoint_dir}/M1_unet_fold{args.fold}/M1_unet_fold{args.fold}_best.pth"),
        ("M1_densenet3d", "m1_style", M1_DenseNet3D, f"{args.checkpoint_dir}/M1_densenet3d_fold{args.fold}/M1_densenet3d_fold{args.fold}_best.pth"),
        ("M1_nnlandmark", "m1_style", M1_nnLandmark, f"{args.checkpoint_dir}/M1_nnlandmark_fold{args.fold}/M1_nnlandmark_fold{args.fold}_best.pth"),
        ("M2_resnet3d_noskip", "m2_style", M2_ResNet3D_NoSkip, f"{args.checkpoint_dir}/M2_resnet3d_noskip_fold{args.fold}/M2_resnet3d_noskip_fold{args.fold}_best.pth"),
        ("M2_densenet3d", "m2_style", M2_DenseNet3D, f"{args.checkpoint_dir}/M2_densenet3d_fold{args.fold}/M2_densenet3d_fold{args.fold}_best.pth"),
        ("M2_nnlandmark", "m2_style", M2_nnLandmark, f"{args.checkpoint_dir}/M2_nnlandmark_fold{args.fold}/M2_nnlandmark_fold{args.fold}_best.pth"),
    ]

    results = {}
    for name, style, model_cls, ckpt_path in variants:
        try:
            model = model_cls().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except FileNotFoundError:
            print(f"SKIP {name}: checkpoint not found at {ckpt_path}")
            continue

        if style == "m1_style":
            angle_errors, centre_dists = evaluate_m1_style(model, test_loader, device, args.target_size)
        else:
            angle_errors, centre_dists = evaluate_m2_style(model, test_loader, device, args.target_size)

        angle_mean, angle_std = float(np.mean(angle_errors)), float(np.std(angle_errors))
        centre_mean, centre_std = float(np.mean(centre_dists)), float(np.std(centre_dists))
        results[name] = {
            "angle_error_mean": angle_mean, "angle_error_std": angle_std,
            "centre_error_mean": centre_mean, "centre_error_std": centre_std,
            "n_test_cases": len(test_ids),
        }
        print(f"{name:25s}  angle_err={angle_mean:6.2f}+/-{angle_std:5.2f} deg   "
              f"centre_err={centre_mean:6.2f}+/-{centre_std:5.2f} mm")

    with open("arch_variants_testset_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to arch_variants_testset_results.json")
    print("\nFor comparison -- M1 (native) test set: 11.33+/-7.34 deg, 9.36+/-5.72mm")
    print("                   M2 (native) test set: 13.28+/-9.19 deg, 16.64+/-8.24mm")


if __name__ == "__main__":
    main()
ARCHEOF

cat > train_arch_variants.sh << 'ARCHEOF'
#!/bin/bash
#SBATCH --job-name=arch_variants
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arch_variant_%a_%j.out
#SBATCH --error=logs/arch_variant_%a_%j.err
#SBATCH --array=0-5

source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa

# Array index -> (script, arch, run_name), fold 0 only for now (matching
# M1_fold0.sh/M2_fold0.sh -- the baseline single-fold comparison point).
# CHECK: once these look right, submit folds 1-4 the same way (matching
# M1/M2's own full 5-fold CV) for a complete comparison, same principle
# used for PartialUnfreeze earlier in this project.
case $SLURM_ARRAY_TASK_ID in
  0)
    python train_m1_variants.py \
      --arch unet --fold 0 --run_name M1_unet_fold0
    ;;
  1)
    python train_m1_variants.py \
      --arch densenet3d --fold 0 --run_name M1_densenet3d_fold0
    ;;
  2)
    python train_m2_variants.py \
      --arch resnet3d_noskip --fold 0 --run_name M2_resnet3d_noskip_fold0
    ;;
  3)
    python train_m2_variants.py \
      --arch densenet3d --fold 0 --run_name M2_densenet3d_fold0
    ;;
  4)
    python train_m1_variants.py \
      --arch nnlandmark --fold 0 --run_name M1_nnlandmark_fold0
    ;;
  5)
    python train_m2_variants.py \
      --arch nnlandmark --fold 0 --run_name M2_nnlandmark_fold0
    ;;
esac
ARCHEOF

echo "=== Testing new variants locally ==="
cd nnlandmark_comparison/arch_variants
python3 m1_nnlandmark.py
python3 m2_nnlandmark.py