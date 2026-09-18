"""
dataset.py
----------
Loads preprocessed .npy volumes + labels from dataset_manifest.csv.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


def load_processed_data(root_path: str, patient_id: str, target_size: int = 64) -> np.ndarray:
    """Load preprocessed .npy volume (already normalised + cropped + resized)."""
    img_path = os.path.join(root_path, patient_id,
                            f'{patient_id}_resized_{target_size}.npy')
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Image not found: {img_path}')
    return np.load(img_path).astype(np.float32)


class AnnulusDataset(Dataset):
    """
    Dataset for aortic annulus plane regression.

    Parameters
    ----------
    root_path   : path to processed_output directory
    patient_ids : list of patient ID strings
    manifest_df : pandas DataFrame with columns:
                  patient_id, centre_D, centre_H, centre_W,
                  normal_D, normal_H, normal_W
    target_size : cubic volume size (default 64)
    augment     : whether to apply augmentation
    rotate_aug  : whether to include rotation augmentation
    """

    def __init__(self,
                 root_path: str,
                 patient_ids: list,
                 target_size: int = 64,
                 augment: bool = False,
                 rotate_aug: bool = True,
                 manifest_df=None):
        self.root_path   = root_path
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.augment     = augment
        self.rotate_aug  = rotate_aug
        if manifest_df is None:
            raise ValueError("manifest_df is required")
        self.manifest_df = manifest_df.set_index('patient_id')

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # Load .npy (already normalised + resized to 64^3)
        img = load_processed_data(self.root_path, pid, target_size=self.target_size)  # (D,H,W)

        # Build 6D label from manifest
        row = self.manifest_df.loc[pid]
        centre_norm = np.array([row['centre_D'],
                                row['centre_H'],
                                row['centre_W']], dtype=np.float32) / self.target_size
        normal = np.array([row['normal_D'],
                           row['normal_H'],
                           row['normal_W']], dtype=np.float32)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-6:
            normal = normal / n_len

        # target: [normal_D, normal_H, normal_W, centre_D, centre_H, centre_W]
        target = np.concatenate([normal, centre_norm], axis=0)

        # Augmentation
        if self.augment:
            # Random flips
            for ax in range(3):
                if np.random.rand() < 0.5:
                    img = np.flip(img, axis=ax).copy()
                    # Flip corresponding normal component
                    normal[ax] = -normal[ax]
                    target = np.concatenate([normal, centre_norm], axis=0)

            # Intensity jitter
            img = img + np.random.uniform(-0.05, 0.05)
            img = np.clip(img, 0.0, 1.0)

        # Add channel dim → (1, D, H, W)
        img_tensor    = torch.from_numpy(img[np.newaxis]).float()
        target_tensor = torch.from_numpy(target).float()

        return img_tensor, target_tensor


# ---------------------------------------------------------------------------
# Patch: __getitem__ must return (img, target, pid) — train.py unpacks 3 vals
# ---------------------------------------------------------------------------
_orig_getitem = AnnulusDataset.__getitem__

def _new_getitem(self, idx):
    img_tensor, target_tensor = _orig_getitem(self, idx)
    pid = self.patient_ids[idx]
    return img_tensor, target_tensor, pid

AnnulusDataset.__getitem__ = _new_getitem


# ---------------------------------------------------------------------------
# create_data_loaders — called by train.py
# ---------------------------------------------------------------------------
import pandas as pd
from torch.utils.data import DataLoader as _DataLoader
from pathlib import Path as _Path

def create_data_loaders(
    root_path:   str,
    train_ids:   list,
    val_ids:     list,
    test_ids:    list,
    target_size: int  = 64,
    batch_size:  int  = 2,
    manifest_path: str = None,
    num_workers: int  = 2,
):
    """
    Build train / val / test DataLoaders from pre-split patient ID lists.
    Manifest is loaded from <root_path>/../dataset_manifest.csv by default.
    """
    if manifest_path is None:
        manifest_path = str(_Path(root_path).parent / 'dataset_manifest.csv')

    manifest_df = pd.read_csv(manifest_path)

    train_ds = AnnulusDataset(
        root_path=root_path, patient_ids=train_ids,
        target_size=target_size, augment=True, manifest_df=manifest_df,
    )
    val_ds = AnnulusDataset(
        root_path=root_path, patient_ids=val_ids,
        target_size=target_size, augment=False, manifest_df=manifest_df,
    )
    test_ds = AnnulusDataset(
        root_path=root_path, patient_ids=test_ids,
        target_size=target_size, augment=False, manifest_df=manifest_df,
    )

    train_loader = _DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=True)
    val_loader   = _DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    test_loader  = _DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)

    print(f'Loaders ready — train: {len(train_ds)}, '
          f'val: {len(val_ds)}, test: {len(test_ds)}')
    return train_loader, val_loader, test_loader
