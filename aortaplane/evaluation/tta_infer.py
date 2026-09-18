"""
tta_infer.py  (v2 — corrected for leak-free fold assignment)
==============================================================
Test-Time Augmentation (TTA) wrapper for the finalised M1 (AnnulusPlaneNet)
checkpoints. Inference-only — no retraining, no checkpoint changes.

CRITICAL FIX vs v1
-------------------
v1 ensembled all 5 fold checkpoints for every patient. That's WRONG for
the 37 CV patients: per fold_config.json, each CV patient is in exactly
one fold's val list and in train for the other 4 -- so ensembling all
5 checkpoints would mean 4 of them saw that patient during training
(leakage). Confirmed against results/ensemble_results.csv, which
contains ONLY the 5 locked test patients:

  - 5 locked test patients (never in any fold's train/val)
        -> ensemble ALL 5 fold checkpoints (mean centre, sign-aligned
           mean normal, std = uncertainty) -- same as ensemble_results.csv
  - 37 CV patients
        -> evaluate using ONLY the single fold checkpoint where that
           patient appears in val (no cross-fold ensembling)

MODEL CONSTRUCTION
-------------------
Trained with --use_cbam, target_size=64, no --use_coordconv (per
jobs/M1_fold0.sh), so checkpoints load as:
    AnnulusPlaneNet(use_cbam=True, use_coordconv=False, coordconv_r=False)
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

try:
    from src.model import AnnulusPlaneNet
except ImportError as e:
    raise ImportError(
        "Could not import AnnulusPlaneNet from src.model. "
        "Run this from the repo root (/data/DERI-ecgai/__users/Vandhanaa)."
    ) from e


VOXEL_TO_MM = {64: 2.0, 128: 1.0}

ALL_FLIP_COMBOS = []
for r in range(0, 4):
    ALL_FLIP_COMBOS.extend(itertools.combinations([0, 1, 2], r))
# -> [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]


def flip_volume(vol, flip_axes):
    if not flip_axes:
        return vol
    dims = [ax + 2 for ax in flip_axes]
    return torch.flip(vol, dims=dims)


def unflip_prediction(normal, centre, size, flip_axes):
    n = normal.copy()
    c = centre.copy()
    for ax in flip_axes:
        n[ax] = -n[ax]
        c[ax] = (size - 1) - c[ax]
    return n, c


def align_sign(reference, n):
    return n if np.dot(reference, n) >= 0 else -n


@torch.no_grad()
def tta_predict_single_checkpoint(model, vol, size, device):
    preds = []
    for combo in ALL_FLIP_COMBOS:
        vol_aug = flip_volume(vol, combo).to(device)
        out = model(vol_aug).squeeze(0).cpu().numpy()
        n_seen, c_seen = out[:3], out[3:]
        n_back, c_back = unflip_prediction(n_seen, c_seen, size, combo)
        preds.append((n_back, c_back))

    ref_normal = preds[0][0]
    aligned_normals = [align_sign(ref_normal, n) for n, _ in preds]
    mean_normal = np.mean(aligned_normals, axis=0)
    norm = np.linalg.norm(mean_normal)
    mean_normal = mean_normal / norm if norm > 1e-8 else mean_normal
    mean_centre = np.mean([c for _, c in preds], axis=0)
    return mean_normal, mean_centre


def angle_error_deg(n_pred, n_gt):
    cos_sim = abs(float(np.dot(n_pred, n_gt)) /
                  (np.linalg.norm(n_pred) * np.linalg.norm(n_gt) + 1e-8))
    cos_sim = min(1.0, max(-1.0, cos_sim))
    return float(np.degrees(np.arccos(cos_sim)))


def centre_error_mm(c_pred, c_gt, voxel_to_mm):
    return float(np.linalg.norm(np.asarray(c_pred) - np.asarray(c_gt)) * voxel_to_mm)


def load_volume(root_path, patient_id, target_size):
    path = os.path.join(root_path, patient_id, f"{patient_id}_resized_{target_size}.npy")
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def build_fold_assignment(fold_config):
    test_patients = set(fold_config["test_patients"])
    cv_fold_of = {}
    for fold_entry in fold_config["folds"]:
        fold_idx = fold_entry["fold"]
        for pid in fold_entry["val"]:
            if pid in cv_fold_of:
                raise ValueError(f"{pid} appears in val for >1 fold.")
            cv_fold_of[pid] = fold_idx
    return test_patients, cv_fold_of


def load_model(checkpoint_path, use_cbam, use_coordconv, coordconv_r, device):
    model = AnnulusPlaneNet(use_cbam=use_cbam, use_coordconv=use_coordconv,
                             coordconv_r=coordconv_r)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_path", required=True)
    ap.add_argument("--root_path", required=True)
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--checkpoint_subdir_pattern", default="M1_fold{fold}")
    ap.add_argument("--checkpoint_filename_pattern", default="M1_fold{fold}_best.pth")
    ap.add_argument("--fold_config", default="fold_config.json")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--target_size", type=int, default=64, choices=[64, 128])
    ap.add_argument("--use_cbam", action="store_true", default=False)
    ap.add_argument("--use_coordconv", action="store_true", default=False)
    ap.add_argument("--coordconv_r", action="store_true", default=False)
    ap.add_argument("--out_csv", default="results/ensemble_results_tta.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_to_mm = VOXEL_TO_MM[args.target_size]

    with open(args.fold_config) as f:
        fold_config = json.load(f)
    test_patients, cv_fold_of = build_fold_assignment(fold_config)

    manifest = pd.read_csv(args.manifest_path)
    fold_models = {}

    def get_model(fold_idx):
        if fold_idx not in fold_models:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                args.checkpoint_subdir_pattern.format(fold=fold_idx),
                args.checkpoint_filename_pattern.format(fold=fold_idx),
            )
            fold_models[fold_idx] = load_model(
                ckpt_path, args.use_cbam, args.use_coordconv, args.coordconv_r, device
            )
            print(f"Loaded fold {fold_idx}: {ckpt_path}")
        return fold_models[fold_idx]

    rows = []
    for _, row in manifest.iterrows():
        patient_id = row["patient_id"]
        gt_normal = np.array([row["normal_D"], row["normal_H"], row["normal_W"]], dtype=float)
        gt_centre = np.array([row["centre_D"], row["centre_H"], row["centre_W"]], dtype=float)

        if patient_id in test_patients:
            eval_type = "test_ensemble"
            fold_indices = list(range(args.n_folds))
        elif patient_id in cv_fold_of:
            eval_type = "cv_single_fold"
            fold_indices = [cv_fold_of[patient_id]]
        else:
            print(f"[skip] {patient_id}: not in test set and not in any fold's val list")
            continue

        try:
            vol = load_volume(args.root_path, patient_id, args.target_size)
        except FileNotFoundError:
            print(f"[skip] volume not found for {patient_id}")
            continue

        fold_normals, fold_centres = [], []
        for fidx in fold_indices:
            model = get_model(fidx)
            n, c = tta_predict_single_checkpoint(model, vol, args.target_size, device)
            fold_normals.append(n)
            fold_centres.append(c)

        if len(fold_normals) > 1:
            ref = fold_normals[0]
            aligned = [align_sign(ref, n) for n in fold_normals]
            ens_normal = np.mean(aligned, axis=0)
            nrm = np.linalg.norm(ens_normal)
            ens_normal = ens_normal / nrm if nrm > 1e-8 else ens_normal
            ens_centre = np.mean(fold_centres, axis=0)
            normal_std = float(np.std(aligned, axis=0).mean())
            centre_std_vox = float(np.std(fold_centres, axis=0).mean())
        else:
            ens_normal, ens_centre = fold_normals[0], fold_centres[0]
            normal_std, centre_std_vox = 0.0, 0.0

        rows.append({
            "patient_id": patient_id,
            "eval_type": eval_type,
            "fold_indices_used": ",".join(map(str, fold_indices)),
            "angle_error_deg_tta": angle_error_deg(ens_normal, gt_normal),
            "centre_error_mm_tta": centre_error_mm(ens_centre, gt_centre, voxel_to_mm),
            "normal_uncertainty_std_tta": normal_std,
            "centre_uncertainty_std_vox_tta": centre_std_vox,
        })
        print(f"{patient_id} [{eval_type}, folds={fold_indices}]: "
              f"angle={rows[-1]['angle_error_deg_tta']:.2f} deg  "
              f"centre={rows[-1]['centre_error_mm_tta']:.2f} mm")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    if out_df.empty:
        print(f"\nSaved 0 patients -> {args.out_csv}. Check [skip] messages above.")
        return

    print(f"\nSaved {len(out_df)} patients -> {args.out_csv}")
    cv_df = out_df[out_df.eval_type == "cv_single_fold"]
    test_df = out_df[out_df.eval_type == "test_ensemble"]
    if len(cv_df):
        print(f"CV (n={len(cv_df)}) mean angle (TTA): {cv_df.angle_error_deg_tta.mean():.2f} deg "
              f"| mean centre (TTA): {cv_df.centre_error_mm_tta.mean():.2f} mm")
        print("Compare against your headline CV numbers: 11.33 deg / 9.36 mm")
    if len(test_df):
        print(f"Test set (n={len(test_df)}) mean angle (TTA): {test_df.angle_error_deg_tta.mean():.2f} deg "
              f"| mean centre (TTA): {test_df.centre_error_mm_tta.mean():.2f} mm")
        print("Compare against ensemble_results.csv M1 columns")


if __name__ == "__main__":
    main()
