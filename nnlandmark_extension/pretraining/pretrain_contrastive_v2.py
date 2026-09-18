#!/usr/bin/env python3
"""
Self-supervised contrastive pretraining using nnLandmark's ACTUAL network
architecture (built via get_network_from_plans), so the resulting encoder
weights load directly into nnLM_train -pretrained_weights with zero shape
mismatches. Only the .encoder submodule is trained/used for the contrastive
objective; the decoder is built but unused here (kept so the state_dict
structure matches exactly what nnU-Net expects on load).
"""
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import sys

sys.path.insert(0, "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLandmark")
from nnlandmark.utilities.get_network_from_plans import get_network_from_plans

MANIFEST_PATH = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/manifest.json"
PLANS_PATH = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed/Dataset741_AnnulusCT/nnUNetPlans.json"
CHECKPOINT_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/checkpoints_v2"
PATCH_SIZE = (56, 192, 160)  # matches Dataset741's original 3d_fullres patch size
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 1e-3
TEMPERATURE = 0.5
EMBED_DIM = 128

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_volume_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, -200, 800)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr


def random_patch(arr, patch_size):
    z, y, x = arr.shape
    pz, py, px = patch_size
    if z < pz or y < py or x < px:
        pad_z, pad_y, pad_x = max(0, pz - z), max(0, py - y), max(0, px - x)
        arr = np.pad(arr, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        z, y, x = arr.shape
    zi = random.randint(0, z - pz)
    yi = random.randint(0, y - py)
    xi = random.randint(0, x - px)
    return arr[zi:zi+pz, yi:yi+py, xi:xi+px].copy()


def augment(patch):
    for axis in [0, 1, 2]:
        if random.random() < 0.5:
            patch = np.flip(patch, axis=axis).copy()
    if random.random() < 0.5:
        patch = patch + np.random.normal(0, 0.1, patch.shape).astype(np.float32)
    if random.random() < 0.5:
        patch = patch * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
    return patch


class ContrastiveCTDataset(Dataset):
    def __init__(self, manifest_path, patch_size):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.patch_size = patch_size
        self._cache = {}

    def __len__(self):
        return len(self.manifest)

    def _get_array(self, idx):
        if idx not in self._cache:
            path = self.manifest[idx]["path"]
            self._cache[idx] = load_volume_array(path)
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def __getitem__(self, idx):
        arr = self._get_array(idx)
        base_patch = random_patch(arr, self.patch_size)
        view1 = augment(base_patch)
        view2 = augment(base_patch)
        return (
            torch.from_numpy(view1).unsqueeze(0),
            torch.from_numpy(view2).unsqueeze(0),
        )


def build_nnlandmark_network(plans_path, input_channels=1, output_channels=4):
    with open(plans_path) as f:
        plans = json.load(f)
    config = plans["configurations"]["3d_fullres"]["architecture"]
    network = get_network_from_plans(
        arch_class_name=config["network_class_name"],
        arch_kwargs=config["arch_kwargs"],
        arch_kwargs_req_import=config["_kw_requires_import"],
        input_channels=input_channels,
        output_channels=output_channels,
        deep_supervision=False,
    )
    return network


class ProjectionHead(nn.Module):
    """Small projection head on top of the encoder's bottleneck features for
    the contrastive objective. Only used during pretraining; discarded after."""
    def __init__(self, in_features, embed_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features), nn.ReLU(inplace=True), nn.Linear(in_features, embed_dim)
        )

    def forward(self, feat_list):
        # nnU-Net's PlainConvEncoder returns a list of per-stage features when
        # return_skips=True; take the deepest (bottleneck) one
        bottleneck = feat_list[-1] if isinstance(feat_list, (list, tuple)) else feat_list
        h = self.pool(bottleneck).flatten(1)
        z = self.proj(h)
        return F.normalize(z, dim=1)


def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    full_network = build_nnlandmark_network(PLANS_PATH).to(device)
    encoder = full_network.encoder
    encoder.return_skips = True  # ensure it returns feature list so we can grab the bottleneck
    bottleneck_channels = 320  # last stage's features_per_stage value from plans

    proj_head = ProjectionHead(bottleneck_channels, EMBED_DIM).to(device)

    dataset = ContrastiveCTDataset(MANIFEST_PATH, PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=LR)

    print(f"Dataset size: {len(dataset)} volumes", flush=True)
    print(f"Starting pretraining for {NUM_EPOCHS} epochs", flush=True)

    ACCUM_STEPS = 6  # effective batch size = BATCH_SIZE * ACCUM_STEPS = 12

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0
        n_updates = 0
        accum_z1, accum_z2 = [], []

        for step_idx, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            feats1 = encoder(view1)
            feats2 = encoder(view2)
            z1, z2 = proj_head(feats1), proj_head(feats2)
            accum_z1.append(z1)
            accum_z2.append(z2)

            if len(accum_z1) == ACCUM_STEPS:
                big_z1 = torch.cat(accum_z1, dim=0)
                big_z2 = torch.cat(accum_z2, dim=0)
                loss = nt_xent_loss(big_z1, big_z2, TEMPERATURE)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_updates += 1
                accum_z1, accum_z2 = [], []
        n_batches = n_updates

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: contrastive_loss={avg_loss:.4f}", flush=True)

        if (epoch + 1) % 10 == 0:
            # Save in nnU-Net checkpoint format so it loads via -pretrained_weights directly
            ckpt = {"network_weights": full_network.state_dict()}
            ckpt_path = f"{CHECKPOINT_DIR}/pretrained_epoch{epoch+1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

    ckpt = {"network_weights": full_network.state_dict()}
    torch.save(ckpt, f"{CHECKPOINT_DIR}/pretrained_final.pth")
    print("Pretraining complete.", flush=True)


if __name__ == "__main__":
    main()
