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
