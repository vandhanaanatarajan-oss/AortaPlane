import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet, AnnulusLandmarkNet

MANIFEST_PATH   = "dataset_manifest.csv"
FOLD_CONFIG     = "fold_config.json"
RESULTS_DIR     = "results"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    normal = normal / norm if norm > 1e-8 else np.array([0., 0., 1.])
    return normal, centre


def predict_m1(vol, ckpt_path):
    m = AnnulusPlaneNet(use_cbam=True).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    m.eval()
    with torch.no_grad():
        out = m(vol).cpu().numpy()[0]
    n = out[:3] / (np.linalg.norm(out[:3]) + 1e-8)
    c = out[3:] * 64
    return n, c


def predict_m2(vol, ckpt_path):
    m = AnnulusLandmarkNet().to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    m.eval()
    with torch.no_grad():
        out = m(vol)
    lm = (out[0] if isinstance(out, tuple) else out).cpu().numpy()[0]
    lm = lm * 64
    n, c = derive_plane(lm)
    return n, c


def main():
    print(f"Device: {DEVICE}")
    manifest = pd.read_csv(MANIFEST_PATH).set_index("patient_id")
    with open(FOLD_CONFIG) as f:
        fold_config = json.load(f)

    rows = []
    for fold_info in fold_config["folds"]:
        fold_i = fold_info["fold"]
        val_patients = fold_info["val"]

        m1_ckpt = f"checkpoints/M1_fold{fold_i}/M1_fold{fold_i}_best.pth"
        m2_ckpt = f"checkpoints/M2_fold{fold_i}/M2_fold{fold_i}_best.pth"

        if not (os.path.exists(m1_ckpt) and os.path.exists(m2_ckpt)):
            print(f"SKIP fold {fold_i} — missing checkpoint(s)")
            continue

        print(f"\n── Fold {fold_i} — {len(val_patients)} val patients ──")

        # Load each fold's checkpoints once, reuse across its val patients
        m1_model = AnnulusPlaneNet(use_cbam=True).to(DEVICE)
        m1_model.load_state_dict(torch.load(m1_ckpt, map_location=DEVICE)["model_state_dict"])
        m1_model.eval()

        m2_model = AnnulusLandmarkNet().to(DEVICE)
        m2_model.load_state_dict(torch.load(m2_ckpt, map_location=DEVICE)["model_state_dict"])
        m2_model.eval()

        for pid in val_patients:
            if pid not in manifest.index:
                print(f"  SKIP {pid} — not in manifest"); continue
            r = manifest.loc[pid]
            vol = load_volume(r["image_path"])
            gt_n = np.array([r["normal_D"], r["normal_H"], r["normal_W"]], dtype=np.float32)
            gt_c = np.array([r["centre_D"], r["centre_H"], r["centre_W"]], dtype=np.float32)
            gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)

            with torch.no_grad():
                out1 = m1_model(vol).cpu().numpy()[0]
            n1 = out1[:3] / (np.linalg.norm(out1[:3]) + 1e-8)
            c1 = out1[3:] * 64

            with torch.no_grad():
                out2 = m2_model(vol)
            lm2 = (out2[0] if isinstance(out2, tuple) else out2).cpu().numpy()[0] * 64
            n2, c2 = derive_plane(lm2)

            m1_angle_err = angle_error_deg(n1, gt_n)
            m2_angle_err = angle_error_deg(n2, gt_n)
            m1_centre_err = centre_error_vx(c1, gt_c)
            m2_centre_err = centre_error_vx(c2, gt_c)

            # cross-model disagreement (sign-ambiguity-safe, same convention as errors above)
            disagreement_angle = angle_error_deg(n1, n2)
            disagreement_centre = centre_error_vx(c1, c2)

            rows.append(dict(
                patient_id=pid, fold=fold_i,
                m1_angle_err=round(m1_angle_err, 3), m2_angle_err=round(m2_angle_err, 3),
                m1_centre_err=round(m1_centre_err, 3), m2_centre_err=round(m2_centre_err, 3),
                disagreement_angle=round(disagreement_angle, 3),
                disagreement_centre=round(disagreement_centre, 3),
            ))
            print(f"  {pid:<38} disagree={disagreement_angle:>6.2f}°  "
                  f"M1_err={m1_angle_err:>6.2f}°  M2_err={m2_angle_err:>6.2f}°")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/cv_disagreement_full.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/cv_disagreement_full.csv  ({len(df)} patients)")

    # --- Does disagreement predict which model is more accurate, or that both are worse? ---
    df["worse_model_err"] = df[["m1_angle_err", "m2_angle_err"]].max(axis=1)
    df["better_model_err"] = df[["m1_angle_err", "m2_angle_err"]].min(axis=1)

    from scipy.stats import spearmanr
    rho_worse, p_worse = spearmanr(df["disagreement_angle"], df["worse_model_err"])
    rho_better, p_better = spearmanr(df["disagreement_angle"], df["better_model_err"])

    print("\n" + "=" * 70)
    print(f"n = {len(df)} out-of-fold CV patients (leak-free: each evaluated only")
    print("by the fold where it was held out of training)")
    print("=" * 70)
    print(f"disagreement vs WORSE model's error  -> rho={rho_worse:.3f}  p={p_worse:.4f}")
    print(f"disagreement vs BETTER model's error -> rho={rho_better:.3f}  p={p_better:.4f}")
    print("\nInterpretation: if disagreement correlates with the worse model's error")
    print("but not the better model's, that supports using disagreement as a")
    print("review-flag trigger (high disagreement = at least one model is likely wrong).")


if __name__ == "__main__":
    main()