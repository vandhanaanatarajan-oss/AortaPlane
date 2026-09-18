#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants')

from src.model import AnnulusPlaneNet, AnnulusLandmarkNet
from m1_unet import M1_UNet
from m1_densenet3d import M1_DenseNet3D
from m1_nnlandmark import M1_nnLandmark
from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_densenet3d import M2_DenseNet3D
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
    ("AortaPlane (native)",    "M1_fold{f}/M1_fold{f}_best.pth",                                 lambda: AnnulusPlaneNet(use_cbam=True), True,  predict_m1style),
    ("AortaPlane + U-Net",     "M1_unet_fold{f}/M1_unet_fold{f}_best.pth",                       M1_UNet,                                False, predict_m1style),
    ("AortaPlane + DenseNet3D","M1_densenet3d_fold{f}/M1_densenet3d_fold{f}_best.pth",           M1_DenseNet3D,                           False, predict_m1style),
    ("AortaPlane + nnLandmark","M1_nnlandmark_fold{f}/M1_nnlandmark_fold{f}_best.pth",           M1_nnLandmark,                           False, predict_m1style),
    ("Baseline (native)",     "M2_fold{f}/M2_fold{f}_best.pth",                                 lambda: AnnulusLandmarkNet(),           True,  predict_landmark_style),
    ("Baseline + ResNet3D(noskip)", "M2_resnet3d_noskip_fold{f}/M2_resnet3d_noskip_fold{f}_best.pth", M2_ResNet3D_NoSkip,                False, predict_landmark_style),
    ("Baseline + DenseNet3D", "M2_densenet3d_fold{f}/M2_densenet3d_fold{f}_best.pth",           M2_DenseNet3D,                           False, predict_landmark_style),
    ("Baseline + nnLandmark", "M2_nnlandmark_fold{f}/M2_nnlandmark_fold{f}_best.pth",           M2_nnLandmark,                           False, predict_landmark_style),
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

    print(f"{'Variant':<30} {'Angle error (deg)':>20} {'Centre error (mm)':>20} {'Centre MAE (mm)':>18}")
    for name, path_tpl, model_fn, wrapped, predict_fn in VARIANTS:
        per_patient_angle, per_patient_centre_err, per_patient_mae = [], [], []
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
            ce = float(np.linalg.norm(mean_c - gt_c)) * 2.0
            diff_mm = (mean_c - gt_c) * 2.0
            mae = float(np.mean(np.abs(diff_mm)))
            per_patient_angle.append(ae)
            per_patient_centre_err.append(ce)
            per_patient_mae.append(mae)

        n_leq10 = sum(1 for a in per_patient_angle if a <= 10)
        n_total = len(per_patient_angle)
        print(f"{name:<30} {np.mean(per_patient_angle):>8.2f} +/- {np.std(per_patient_angle):<7.2f} "
              f"{np.mean(per_patient_centre_err):>8.2f} +/- {np.std(per_patient_centre_err):<7.2f} "
              f"{np.mean(per_patient_mae):>8.2f} +/- {np.std(per_patient_mae):<7.2f} "
              f"{n_leq10}/{n_total}")

if __name__ == "__main__":
    main()
