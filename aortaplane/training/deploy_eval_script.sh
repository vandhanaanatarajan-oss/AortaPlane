cat > /data/DERI-ecgai/__users/Vandhanaa/evaluate_arch_variants_testset.py << 'EVALEOF'
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
from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_densenet3d import M2_DenseNet3D


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
        ("M2_resnet3d_noskip", "m2_style", M2_ResNet3D_NoSkip, f"{args.checkpoint_dir}/M2_resnet3d_noskip_fold{args.fold}/M2_resnet3d_noskip_fold{args.fold}_best.pth"),
        ("M2_densenet3d", "m2_style", M2_DenseNet3D, f"{args.checkpoint_dir}/M2_densenet3d_fold{args.fold}/M2_densenet3d_fold{args.fold}_best.pth"),
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
EVALEOF