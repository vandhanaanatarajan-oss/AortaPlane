import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet, AnnulusLandmarkNet

MANIFEST_PATH = "dataset_manifest.csv"
RESULTS_DIR   = "results"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# patient_id -> the single fold it was held out on (from cv_disagreement_full.csv)
CV_PATIENTS = {
    "D1": 4,
    "CONTCT_R_13_FBA": 0,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_volume(path):
    vol = np.load(path).astype(np.float32)
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)


def derive_plane(lm9):
    rc, nc, lc = lm9[0:3], lm9[3:6], lm9[6:9]
    centre = (rc + nc + lc) / 3.0
    normal = np.cross(nc - rc, lc - rc)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-8 else np.array([0., 0., 1.])
    return normal, centre


def main():
    print(f"Device: {DEVICE}")
    manifest = pd.read_csv(MANIFEST_PATH).set_index("patient_id")
    rows = []

    for pid, fold in CV_PATIENTS.items():
        if pid not in manifest.index:
            print(f"SKIP {pid} (not in manifest)")
            continue
        r = manifest.loc[pid]
        vol = load_volume(r["image_path"])
        gt_n = np.array([r["normal_D"], r["normal_H"], r["normal_W"]], dtype=np.float32)
        gt_c = np.array([r["centre_D"], r["centre_H"], r["centre_W"]], dtype=np.float32)
        gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)
        print(f"── {pid} (held-out fold {fold}) ──")

        # Final model: M1 + CoordConv, single correct fold
        ck_final = f"checkpoints/M1_coordconv_fold{fold}/M1_coordconv_fold{fold}_best.pth"
        with torch.no_grad():
            m = AnnulusPlaneNet(use_cbam=True, use_coordconv=True).to(DEVICE)
            m.load_state_dict(torch.load(ck_final, map_location=DEVICE)["model_state_dict"])
            m.eval()
            out = m(vol).cpu().numpy()[0]
            n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
            c = out[3:] * 64
            rows.append(dict(patient_id=pid, model="final_coordconv", fold=fold,
                pred_normal_D=n[0], pred_normal_H=n[1], pred_normal_W=n[2],
                pred_centre_D=c[0], pred_centre_H=c[1], pred_centre_W=c[2],
                gt_normal_D=gt_n[0], gt_normal_H=gt_n[1], gt_normal_W=gt_n[2],
                gt_centre_D=gt_c[0], gt_centre_H=gt_c[1], gt_centre_W=gt_c[2]))
            print(f"  final (coordconv) fold{fold} done")

        # M2: landmark-derived, single correct fold
        ck_m2 = f"checkpoints/M2_fold{fold}/M2_fold{fold}_best.pth"
        with torch.no_grad():
            m = AnnulusLandmarkNet().to(DEVICE)
            m.load_state_dict(torch.load(ck_m2, map_location=DEVICE)["model_state_dict"])
            m.eval()
            out = m(vol)
            lm = (out[0] if isinstance(out, tuple) else out).cpu().numpy()[0]
            lm = lm * 64
            n, c = derive_plane(lm)
            rows.append(dict(patient_id=pid, model="M2", fold=fold,
                pred_normal_D=n[0], pred_normal_H=n[1], pred_normal_W=n[2],
                pred_centre_D=c[0], pred_centre_H=c[1], pred_centre_W=c[2],
                gt_normal_D=gt_n[0], gt_normal_H=gt_n[1], gt_normal_W=gt_n[2],
                gt_centre_D=gt_c[0], gt_centre_H=gt_c[1], gt_centre_W=gt_c[2]))
            print(f"  M2 fold{fold} done")

    df = pd.DataFrame(rows)
    out_path = f"{RESULTS_DIR}/fig1_cv_patient_vectors.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}  (expect {len(CV_PATIENTS)} patients x 2 models = {len(CV_PATIENTS) * 2})")


if __name__ == "__main__":
    main()