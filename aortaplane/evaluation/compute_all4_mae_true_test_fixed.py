#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants')

from src.model import AnnulusPlaneNet, AnnulusLandmarkNet
from m1_nnlandmark import M1_nnLandmark
from m2_nnlandmark import M2_nnLandmark

MANIFEST_PATH = "dataset_manifest.csv"
TEST_IDS = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8", "K10", "R10"]

def angle_error_deg(pred_n, gt_n):
    pred_n = pred_n / (np.linalg.norm(pred_n) + 1e-8)
    gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(pred_n, gt_n)), 0, 1))))

def align_normals(normals):
    ref = normals[0]
    return np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])

def predict_m1style(model, npy_path, device):
    vol = torch.tensor(np.load(npy_path)).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(vol)
    out = raw.cpu().numpy()[0]
    n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
    c = out[3:] * 64
    return n, c

def predict_landmark_style(model, npy_path, device):
    vol = torch.tensor(np.load(npy_path)).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(vol)
    lm9 = raw[0].cpu().numpy()[0] * 64
    rc, nc, lc = lm9[0:3], lm9[3:6], lm9[6:9]
    centre = (rc + nc + lc) / 3
    normal = np.cross(nc - rc, lc - rc)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-8 else np.array([0., 0., 1.])
    return normal, centre

VARIANTS = [
    ("AortaPlane (native)",     "M1_fold{f}/M1_fold{f}_best.pth",             lambda: AnnulusPlaneNet(use_cbam=True), True,  predict_m1style),
    ("AortaPlane + nnLandmark", "M1_nnlandmark_fold{f}/M1_nnlandmark_fold{f}_best.pth", M1_nnLandmark,                    False, predict_m1style),
    ("Baseline (native)",       "M2_fold{f}/M2_fold{f}_best.pth",             lambda: AnnulusLandmarkNet(),           True,  predict_landmark_style),
    ("Baseline + nnLandmark",   "M2_nnlandmark_fold{f}/M2_nnlandmark_fold{f}_best.pth", M2_nnLandmark,                    False, predict_landmark_style),
]

def load_model(ckpt_path, model_fn, wrapped, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = model_fn()
    if wrapped:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(MANIFEST_PATH).set_index("patient_id")

    print(f"{'Variant':<28} {'Angle error (deg)':>22} {'Centre MAE (mm)':>20}")
    for name, path_tpl, model_fn, wrapped, predict_fn in VARIANTS:
        per_patient_angle, per_patient_mae = [], []
        for pid in TEST_IDS:
            row = manifest.loc[pid]
            gt_n = np.array([row.normal_D, row.normal_H, row.normal_W])
            gt_c = np.array([row.centre_D, row.centre_H, row.centre_W])

            fold_normals, fold_centres = [], []
            for fold in range(5):
                ckpt_path = f"checkpoints/{path_tpl.format(f=fold)}"
                model = load_model(ckpt_path, model_fn, wrapped, device)
                n, c = predict_fn(model, row["image_path"], device)
                fold_normals.append(n)
                fold_centres.append(c)

            fold_normals = align_normals(np.array(fold_normals))
            mean_n = fold_normals.mean(0)
            mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-8)
            mean_c = np.mean(fold_centres, axis=0)

            ae = angle_error_deg(mean_n, gt_n)
            diff_mm = (mean_c - gt_c) * 2.0
            mae = float(np.mean(np.abs(diff_mm)))
            per_patient_angle.append(ae)
            per_patient_mae.append(mae)

        print(f"{name:<28} {np.mean(per_patient_angle):>10.2f} +/- {np.std(per_patient_angle):<7.2f} {np.mean(per_patient_mae):>10.2f} +/- {np.std(per_patient_mae):<7.2f}")

if __name__ == "__main__":
    main()
