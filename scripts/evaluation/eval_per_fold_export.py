import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet, AnnulusLandmarkNet

MANIFEST_PATH  = "dataset_manifest.csv"
RESULTS_DIR    = "results"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M1_CHECKPOINTS = [f"checkpoints/M1_fold{i}/M1_fold{i}_best.pth" for i in range(5)]
M2_CHECKPOINTS = [f"checkpoints/M2_fold{i}/M2_fold{i}_best.pth" for i in range(5)]
TEST_PATIENTS  = ["CONTCT_R_08_FBA","CONTCT_R_27_FBA_no_extended_seg","D8","K10","R10"]
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_volume(path):
    vol = np.load(path).astype(np.float32)
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)

def derive_plane(lm9):
    rc, nc, lc = lm9[0:3], lm9[3:6], lm9[6:9]
    centre = (rc + nc + lc) / 3.0
    normal = np.cross(nc - rc, lc - rc)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-8 else np.array([0.,0.,1.])
    return normal, centre

def main():
    print(f"Device: {DEVICE}")
    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest[manifest["patient_id"].isin(TEST_PATIENTS)].set_index("patient_id")
    rows = []

    for pid in TEST_PATIENTS:
        if pid not in test_rows.index:
            print(f"SKIP {pid}"); continue
        r   = test_rows.loc[pid]
        vol = load_volume(r["image_path"])
        gt_n = np.array([r["normal_D"], r["normal_H"], r["normal_W"]], dtype=np.float32)
        gt_c = np.array([r["centre_D"], r["centre_H"], r["centre_W"]], dtype=np.float32)
        gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)
        print(f"── {pid} ──")

        # M1 — direct regression, per fold
        with torch.no_grad():
            for fold, ck in enumerate(M1_CHECKPOINTS):
                m = AnnulusPlaneNet(use_cbam=True).to(DEVICE)
                m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_state_dict"])
                m.eval()
                out = m(vol).cpu().numpy()[0]
                n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
                c = out[3:] * 64
                rows.append(dict(patient_id=pid, model="M1", fold=fold,
                    pred_normal_D=n[0], pred_normal_H=n[1], pred_normal_W=n[2],
                    pred_centre_D=c[0], pred_centre_H=c[1], pred_centre_W=c[2],
                    gt_normal_D=gt_n[0], gt_normal_H=gt_n[1], gt_normal_W=gt_n[2],
                    gt_centre_D=gt_c[0], gt_centre_H=gt_c[1], gt_centre_W=gt_c[2]))
                print(f"  M1 fold{fold} done")

        # M2 — landmark-derived, per fold
        with torch.no_grad():
            for fold, ck in enumerate(M2_CHECKPOINTS):
                m = AnnulusLandmarkNet().to(DEVICE)
                m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_state_dict"])
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
    out_path = f"{RESULTS_DIR}/per_fold_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}  (expect {len(TEST_PATIENTS)} patients x 2 models x 5 folds = {len(TEST_PATIENTS)*10})")

if __name__ == "__main__":
    main()
