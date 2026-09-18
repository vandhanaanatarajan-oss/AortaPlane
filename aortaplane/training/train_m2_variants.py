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
