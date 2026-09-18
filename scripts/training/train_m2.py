"""
train_m2.py
-----------
Training script for M2 — AnnulusLandmarkNet (Type A approach).
Reads fold splits from fold_config.json, same as train.py.

Usage:
    python train_m2.py --fold 0 --no_wandb
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import wandb
except:
    wandb = None

import sys
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')

from src.dataset import create_data_loaders
from src.model   import AnnulusLandmarkNet, landmarks_to_plane
from src.loss    import LandmarkPlaneLoss, angle_error_degrees, centre_distance_voxels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fold',        type=int, required=True)
    p.add_argument('--fold_config', type=str, default='fold_config.json')
    p.add_argument('--root_path',   type=str,
                   default='/data/DERI-ecgai/__users/Vandhanaa/processed_output')
    p.add_argument('--target_size', type=int, default=64)
    p.add_argument('--batch_size',  type=int, default=2)
    p.add_argument('--epochs',      type=int, default=300)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--dropout',     type=float, default=0.3)
    p.add_argument('--w_normal',    type=float, default=1.0)
    p.add_argument('--w_centre',    type=float, default=1.0)
    p.add_argument('--w_spread',    type=float, default=0.1)
    p.add_argument('--patience',    type=int, default=60)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--save_dir',    type=str, default='checkpoints')
    p.add_argument('--no_wandb',    action='store_true')
    p.add_argument('--wandb_project', type=str, default='AnnulusPlaneDetector')
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for imgs, targets, _ in loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        landmarks, _ = model(imgs)
        loss, _      = loss_fn(landmarks, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device, image_size):
    model.eval()
    total_loss   = 0.0
    angle_errors = []
    centre_dists = []
    n_batches    = 0
    for imgs, targets, _ in loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)
        landmarks, _ = model(imgs)
        loss, _      = loss_fn(landmarks, targets)
        total_loss  += loss.item()
        n_batches   += 1
        # Convert landmarks → plane for interpretable metrics
        plane = landmarks_to_plane(landmarks)
        ang   = angle_error_degrees(plane[:, :3], targets[:, :3])
        dist  = centre_distance_voxels(plane[:, 3:], targets[:, 3:], image_size)
        angle_errors.extend(ang.cpu().numpy().tolist())
        centre_dists.extend(dist.cpu().numpy().tolist())
    return {
        'loss'             : total_loss / n_batches,
        'angle_error_mean' : float(np.mean(angle_errors)),
        'angle_error_std'  : float(np.std(angle_errors)),
        'centre_dist_mean' : float(np.mean(centre_dists)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load fold
    with open(args.fold_config) as f:
        fc = json.load(f)
    fold_data  = fc['folds'][args.fold]
    train_ids  = fold_data['train']
    val_ids    = fold_data['val']
    test_ids   = fc['test_patients']

    run_name   = f'M2_fold{args.fold}'
    save_dir   = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path  = os.path.join(save_dir, f'{run_name}_best.pth')
    last_path  = os.path.join(save_dir, f'{run_name}_last.pth')

    print(f'\n{"="*60}')
    print(f'  M2 Training — {run_name}')
    print(f'  Train: {len(train_ids)}  Val: {len(val_ids)}')
    print(f'{"="*60}\n')

    # Data
    train_loader, val_loader, _ = create_data_loaders(
        args.root_path, train_ids, val_ids, test_ids,
        target_size=args.target_size, batch_size=args.batch_size,
    )

    # Model + loss
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = AnnulusLandmarkNet(dropout_p=args.dropout,
                                  use_cbam=True).to(device)
    loss_fn = LandmarkPlaneLoss(w_normal=args.w_normal,
                                 w_centre=args.w_centre,
                                 w_spread=args.w_spread)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Device: {device}  |  Parameters: {total_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # WandB
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss  = train_one_epoch(model, train_loader, optimizer,
                                      loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn,
                               device, args.target_size)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f'Ep {epoch+1:04d}/{args.epochs} '
            f'| train {train_loss:.4f} '
            f'| val {val_metrics["loss"]:.4f} '
            f'| angle {val_metrics["angle_error_mean"]:.2f}° '
            f'| centre {val_metrics["centre_dist_mean"]:.2f}vx '
            f'| {elapsed:.1f}s'
        )

        if use_wandb:
            wandb.log({'epoch': epoch+1, 'train/loss': train_loss,
                       **{f'val/{k}': v for k, v in val_metrics.items()}})

        # Save last
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_loss': val_metrics['loss'], 'args': vars(args)},
                   last_path)

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss    = val_metrics['loss']
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_metrics['loss'], 'args': vars(args)},
                       best_path)
            print(f'  ★ New best val_loss = {best_val_loss:.4f}')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    if use_wandb:
        wandb.finish()
    print(f'\nDone. Best checkpoint → {best_path}')


if __name__ == '__main__':
    main()
