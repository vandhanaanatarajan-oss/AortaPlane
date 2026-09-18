import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet, AnnulusLandmarkNet

MANIFEST_PATH  = "dataset_manifest_128.csv"
RESULTS_DIR    = "results_128"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M1_CHECKPOINTS = [f"checkpoints/M1_128_fold{i}/M1_128_fold{i}_best.pth" for i in range(5)]
M2_CHECKPOINTS = [f"checkpoints/M2_128_fold{i}/M2_128_fold{i}_best.pth" for i in range(5)]
TEST_PATIENTS  = ["CONTCT_R_08_FBA","CONTCT_R_27_FBA_no_extended_seg","D8","K10","R10"]
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_volume(path):
    vol = np.load(path).astype(np.float32)
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)

def angle_error_deg(pred_n, gt_n):
    pred_n = pred_n / (np.linalg.norm(pred_n) + 1e-8)
    gt_n   = gt_n   / (np.linalg.norm(gt_n)   + 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(pred_n, gt_n)), 0, 1))))

def centre_error_vx(pred_c, gt_c):
    return float(np.linalg.norm(pred_c - gt_c))

def derive_plane(lm9):
    rc, nc, lc = lm9[0:3], lm9[3:6], lm9[6:9]
    centre = (rc + nc + lc) / 3.0
    normal = np.cross(nc - rc, lc - rc)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 1e-8 else np.array([0.,0.,1.])
    return normal, centre

def align_normals(normals):
    ref = normals[0]
    return np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])

def run_m1(vol, ckpts):
    preds = []
    with torch.no_grad():
        for ck in ckpts:
            m = AnnulusPlaneNet(use_cbam=True).to(DEVICE)
            m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_state_dict"])
            m.eval()
            out = m(vol).cpu().numpy()[0]
            n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
            c = out[3:] * 128
            preds.append(np.concatenate([n, c]))
    preds = np.array(preds)
    normals = align_normals(preds[:, :3])
    centres = preds[:, 3:]
    mn = normals.mean(0); mn = mn / (np.linalg.norm(mn) + 1e-8)
    mc = centres.mean(0)
    return mn, mc, normals.std(0).mean(), centres.std(0).mean(), preds

def run_m2(vol, ckpts):
    planes, lms = [], []
    with torch.no_grad():
        for ck in ckpts:
            m = AnnulusLandmarkNet().to(DEVICE)
            m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_state_dict"])
            m.eval()
            out = m(vol)
            lm = (out[0] if isinstance(out, tuple) else out).cpu().numpy()[0]
            lm = lm * 128
            lms.append(lm)
            n, c = derive_plane(lm)
            planes.append(np.concatenate([n, c]))
    planes = np.array(planes)
    normals = align_normals(planes[:, :3])
    centres = planes[:, 3:]
    mn = normals.mean(0); mn = mn / (np.linalg.norm(mn) + 1e-8)
    mc = centres.mean(0)
    return mn, mc, normals.std(0).mean(), centres.std(0).mean(), planes, np.array(lms)

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
        m1n, m1c, m1sn, m1sc, m1p = run_m1(vol, M1_CHECKPOINTS)
        m1a = angle_error_deg(m1n, gt_n); m1ce = centre_error_vx(m1c, gt_c)
        m1fa = [angle_error_deg(m1p[i,:3], gt_n) for i in range(5)]
        print(f"  M1 → angle={m1a:.2f}°  centre={m1ce:.2f}vx  std_n={m1sn:.4f}  std_c={m1sc:.4f}")
        print(f"  M1 folds: {[f'{a:.1f}' for a in m1fa]}")
        rows.append(dict(patient_id=pid,
            m1_angle=round(m1a,3), m1_centre=round(m1ce,3), m1_std_n=round(float(m1sn),4), m1_std_c=round(float(m1sc),4)))
    df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print(f"{'Patient':<38} {'M1°':>6} {'M1vx':>6}")
    print("-"*65)
    for _, r in df.iterrows():
        print(f"{r.patient_id:<38} {r.m1_angle:>6.2f} {r.m1_centre:>6.2f}")
    print("-"*65)
    print(f"{'MEAN':<38} {df.m1_angle.mean():>6.2f} {df.m1_centre.mean():>6.2f}")
    print(f"{'MEDIAN':<38} {df.m1_angle.median():>6.2f}")
    print(f"{'<=10deg':<38} {(df.m1_angle<10).sum():>6}")
    print(f"\nUncertainty — M1 std_n:{df.m1_std_n.mean():.4f} std_c:{df.m1_std_c.mean():.4f}")
    print("="*65)
    df.to_csv(f"{RESULTS_DIR}/ensemble_results.csv", index=False)
    print(f"Saved: {RESULTS_DIR}/ensemble_results.csv")

if __name__ == "__main__":
    main()
