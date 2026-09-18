#!/usr/bin/env python3
"""
Self-supervised contrastive pretraining on unlabelled CTA volumes.
SimCLR-style: two augmented 3D patches from the same volume location are
pulled together in embedding space; patches from different volumes/locations
are pushed apart (NT-Xent loss).
"""
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

MANIFEST_PATH = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/manifest.json"
CHECKPOINT_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/checkpoints"
PATCH_SIZE = (64, 64, 64)
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 1e-3
TEMPERATURE = 0.5

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_volume_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    # basic CT windowing/normalisation
    arr = np.clip(arr, -200, 800)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr


def random_patch(arr, patch_size):
    z, y, x = arr.shape
    pz, py, px = patch_size
    if z < pz or y < py or x < px:
        # pad if volume smaller than patch
        pad_z, pad_y, pad_x = max(0, pz - z), max(0, py - y), max(0, px - x)
        arr = np.pad(arr, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        z, y, x = arr.shape
    zi = random.randint(0, z - pz)
    yi = random.randint(0, y - py)
    xi = random.randint(0, x - px)
    return arr[zi:zi+pz, yi:yi+py, xi:xi+px].copy()


def augment(patch):
    # random flip
    for axis in [0, 1, 2]:
        if random.random() < 0.5:
            patch = np.flip(patch, axis=axis).copy()
    # random gaussian noise
    if random.random() < 0.5:
        patch = patch + np.random.normal(0, 0.1, patch.shape).astype(np.float32)
    # random intensity scale/shift
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
            if len(self._cache) > 2:  # simple cache cap to limit memory
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


class Simple3DEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1), nn.InstanceNorm3d(32), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm3d(64), nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm3d(128), nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm3d(256), nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        h = self.net(x).flatten(1)
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

    dataset = ContrastiveCTDataset(MANIFEST_PATH, PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    model = Simple3DEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Dataset size: {len(dataset)} volumes", flush=True)
    print(f"Starting pretraining for {NUM_EPOCHS} epochs", flush=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            z1, z2 = model(view1), model(view2)
            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: contrastive_loss={avg_loss:.4f}", flush=True)

        if (epoch + 1) % 10 == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/encoder_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/encoder_final.pth")
    print("Pretraining complete.", flush=True)


if __name__ == "__main__":
    main()
