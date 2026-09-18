import os, sys, json
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet

MANIFEST_PATH  = "dataset_manifest.csv"
RESULTS_DIR    = "results_coordconv"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "M1" slot = baseline M1 (no CoordConv)   |   "M2" slot reused = CoordConv M1
BASELINE_CHECKPOINTS  = [f"checkpoints/M1_fold{i}/M1_fold{i}_best.pth" for i in range(5)]
COORDCONV_CHECKPOINTS = [f"checkpoints/M1_coordconv_fold{i}/M1_coordconv_fold{i}_best.pth" for i in range(5)]

TEST_PATIENTS  = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8", "K10", "R10"]

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


def align_normals(normals):
    ref = normals[0]
    return np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])


def run_ensemble(vol, ckpts, use_coordconv):
    """Runs an M1-family ensemble (baseline or CoordConv) and returns
    mean normal, mean centre, normal std, centre std, and per-fold preds."""
    preds = []
    with torch.no_grad():
        for ck in ckpts:
            m = AnnulusPlaneNet(use_cbam=True, use_coordconv=use_coordconv).to(DEVICE)
            m.load_state_dict(torch.load(ck, map_location=DEVICE)["model_state_dict"])
            m.eval()
            out = m(vol).cpu().numpy()[0]
            n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
            c = out[3:] * 64
            preds.append(np.concatenate([n, c]))
    preds = np.array(preds)
    normals = align_normals(preds[:, :3])
    centres = preds[:, 3:]
    mn = normals.mean(0); mn = mn / (np.linalg.norm(mn) + 1e-8)
    mc = centres.mean(0)
    return mn, mc, normals.std(0).mean(), centres.std(0).mean(), preds


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

        bn, bc, bsn, bsc, bp = run_ensemble(vol, BASELINE_CHECKPOINTS, use_coordconv=False)
        ba = angle_error_deg(bn, gt_n); bce = centre_error_vx(bc, gt_c)
        bfa = [angle_error_deg(bp[i, :3], gt_n) for i in range(5)]
        print(f"  Baseline M1  → angle={ba:.2f}°  centre={bce:.2f}vx  std_n={bsn:.4f}  std_c={bsc:.4f}")
        print(f"  Baseline folds: {[f'{a:.1f}' for a in bfa]}")

        cn, cc, csn, csc, cp = run_ensemble(vol, COORDCONV_CHECKPOINTS, use_coordconv=True)
        ca = angle_error_deg(cn, gt_n); cce = centre_error_vx(cc, gt_c)
        cfa = [angle_error_deg(cp[i, :3], gt_n) for i in range(5)]
        print(f"  CoordConv M1 → angle={ca:.2f}°  centre={cce:.2f}vx  std_n={csn:.4f}  std_c={csc:.4f}")
        print(f"  CoordConv folds: {[f'{a:.1f}' for a in cfa]}")

        rows.append(dict(
            patient_id=pid,
            baseline_angle=round(ba, 3), baseline_centre=round(bce, 3),
            baseline_std_n=round(float(bsn), 4), baseline_std_c=round(float(bsc), 4),
            coordconv_angle=round(ca, 3), coordconv_centre=round(cce, 3),
            coordconv_std_n=round(float(csn), 4), coordconv_std_c=round(float(csc), 4),
            baseline_centre_D=float(bc[0]), baseline_centre_H=float(bc[1]), baseline_centre_W=float(bc[2]),
            baseline_normal_D=float(bn[0]), baseline_normal_H=float(bn[1]), baseline_normal_W=float(bn[2]),
            coordconv_centre_D=float(cc[0]), coordconv_centre_H=float(cc[1]), coordconv_centre_W=float(cc[2]),
            coordconv_normal_D=float(cn[0]), coordconv_normal_H=float(cn[1]), coordconv_normal_W=float(cn[2]),
            gt_centre_D=float(gt_c[0]), gt_centre_H=float(gt_c[1]), gt_centre_W=float(gt_c[2]),
            gt_normal_D=float(gt_n[0]), gt_normal_H=float(gt_n[1]), gt_normal_W=float(gt_n[2]),
        ))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 75)
    print(f"{'Patient':<38} {'Base°':>7} {'Basevx':>7} {'CC°':>7} {'CCvx':>7} {'Win':>5}")
    print("-" * 75)
    for _, r in df.iterrows():
        w = "CC" if r.coordconv_angle < r.baseline_angle else "Base"
        print(f"{r.patient_id:<38} {r.baseline_angle:>7.2f} {r.baseline_centre:>7.2f} "
              f"{r.coordconv_angle:>7.2f} {r.coordconv_centre:>7.2f} {w:>5}")
    print("-" * 75)
    print(f"{'MEAN':<38} {df.baseline_angle.mean():>7.2f} {df.baseline_centre.mean():>7.2f} "
          f"{df.coordconv_angle.mean():>7.2f} {df.coordconv_centre.mean():>7.2f}")
    print(f"{'MEDIAN':<38} {df.baseline_angle.median():>7.2f} {'':>7} {df.coordconv_angle.median():>7.2f}")
    print(f"{'<=10deg':<38} {(df.baseline_angle < 10).sum():>7} {'':>7} {(df.coordconv_angle < 10).sum():>7}")
    print(f"\nUncertainty — Baseline std_n:{df.baseline_std_n.mean():.4f} std_c:{df.baseline_std_c.mean():.4f}")
    print(f"Uncertainty — CoordConv std_n:{df.coordconv_std_n.mean():.4f} std_c:{df.coordconv_std_c.mean():.4f}")
    print("=" * 75)

    df.to_csv(f"{RESULTS_DIR}/ensemble_results_coordconv_vs_baseline.csv", index=False)
    print(f"Saved: {RESULTS_DIR}/ensemble_results_coordconv_vs_baseline.csv")


if __name__ == "__main__":
    main()