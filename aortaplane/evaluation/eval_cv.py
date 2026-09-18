import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet, AnnulusLandmarkNet

MANIFEST_PATH = "dataset_manifest.csv"
FOLD_CONFIG   = "fold_config.json"
DATA_DIR      = "processed_output"
CKPT_DIR      = "checkpoints"

def angle_error_deg(pred_n, gt_n):
    pred_n = pred_n / (np.linalg.norm(pred_n) + 1e-8)
    gt_n   = gt_n   / (np.linalg.norm(gt_n)   + 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(pred_n, gt_n)), 0, 1))))

def centre_error_vx(pred_c, gt_c):
    return float(np.linalg.norm(pred_c - gt_c))

def load_model(ckpt_path, model_type, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    if model_type == "M1":
        model = AnnulusPlaneNet(use_cbam=True)
    elif model_type == "CoordConv":
        model = AnnulusPlaneNet(use_cbam=True, use_coordconv=True)
    else:
        model = AnnulusLandmarkNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

def predict(model, npy_path, device):
    vol = torch.tensor(np.load(npy_path)).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(vol)
    if isinstance(raw, tuple):
        # M2: AnnulusLandmarkNet returns (landmarks (B,9), heatmaps)
        lm9 = raw[0].cpu().numpy()[0] * 64  # scale to voxel space
        rc, nc, lc = lm9[0:3], lm9[3:6], lm9[6:9]
        centre = (rc + nc + lc) / 3
        normal = np.cross(nc - rc, lc - rc)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
    else:
        # M1: AnnulusPlaneNet returns (B,6)
        out    = raw.cpu().numpy()[0]
        normal = out[:3]
        centre = out[3:] * 64
    return normal, centre

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    manifest  = pd.read_csv(MANIFEST_PATH).set_index("patient_id")
    fold_cfg  = json.load(open(FOLD_CONFIG))

    # Build map: patient_id -> fold index (val set membership)
    patient_to_fold = {}
    for fd in fold_cfg["folds"]:
        for pid in fd["val"]:
            patient_to_fold[pid] = fd["fold"]

    results = []

    for model_type in ["M1", "M2", "CoordConv"]:
        print(f"\n{'='*60}")
        print(f"  {model_type} — Cross-validated evaluation (val patients)")
        print(f"{'='*60}")

        all_angle, all_centre, all_mae, all_mse = [], [], [], []

        for pid, fold in sorted(patient_to_fold.items()):
            if pid not in manifest.index:
                print(f"  SKIP {pid} — not in manifest")
                continue

            row      = manifest.loc[pid]
            npy_path = row["image_path"]
            gt_n     = np.array([row.normal_D, row.normal_H, row.normal_W])
            gt_c     = np.array([row.centre_D, row.centre_H, row.centre_W])

            ckpt_name = "M1_coordconv" if model_type == "CoordConv" else model_type
            ckpt_path = f"{CKPT_DIR}/{ckpt_name}_fold{fold}/{ckpt_name}_fold{fold}_best.pth"
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {pid} — checkpoint missing: {ckpt_path}")
                continue

            model    = load_model(ckpt_path, model_type, device)
            pred_n, pred_c = predict(model, npy_path, device)

            ae = angle_error_deg(pred_n, gt_n)
            ce = centre_error_vx(pred_c, gt_c) * 2.0  # voxels → mm (128mm ROI / 64 voxels = 2mm/voxel)
            diff_mm = (pred_c - gt_c) * 2.0  # per-axis centre error in mm
            mae = float(np.mean(np.abs(diff_mm)))
            mse = float(np.mean(diff_mm ** 2))
            all_angle.append(ae)
            all_centre.append(ce)
            all_mae.append(mae)
            all_mse.append(mse)

            print(f"  Fold{fold} | {pid:<45} angle={ae:6.2f}°  centre={ce:5.2f}mm  MAE={mae:5.2f}mm  MSE={mse:6.2f}mm²")
            results.append(dict(model=model_type, patient_id=pid, fold=fold,
                                angle=round(ae,3), centre=round(ce,3), mae=round(mae,3), mse=round(mse,3),
                                pred_normal_D=float(pred_n[0]), pred_normal_H=float(pred_n[1]), pred_normal_W=float(pred_n[2]),
                                pred_centre_D=float(pred_c[0]), pred_centre_H=float(pred_c[1]), pred_centre_W=float(pred_c[2]),
                                gt_normal_D=float(gt_n[0]), gt_normal_H=float(gt_n[1]), gt_normal_W=float(gt_n[2]),
                                gt_centre_D=float(gt_c[0]), gt_centre_H=float(gt_c[1]), gt_centre_W=float(gt_c[2])))

        print(f"\n  {model_type} CV Results ({len(all_angle)} patients):")
        print(f"    Angle  — mean={np.mean(all_angle):.2f}°  std={np.std(all_angle):.2f}°  median={np.median(all_angle):.2f}°")
        print(f"    Centre — mean={np.mean(all_centre):.2f}mm  std={np.std(all_centre):.2f}mm  median={np.median(all_centre):.2f}mm")
        print(f"    Centre MAE — mean={np.mean(all_mae):.2f}mm  std={np.std(all_mae):.2f}mm")
        print(f"    Centre MSE — mean={np.mean(all_mse):.2f}mm²  std={np.std(all_mse):.2f}mm²")
        print(f"    ≤10°: {sum(a<=10 for a in all_angle)}/{len(all_angle)}  |  ≤5°: {sum(a<=5 for a in all_angle)}/{len(all_angle)}")

    # Save
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/cv_results.csv", index=False)
    print(f"\nSaved: results/cv_results.csv")
