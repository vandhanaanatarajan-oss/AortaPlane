"""
Verify Table 2 (headline M1 vs M2, n=37) against results/cv_results.csv.

Run from your project root on Apocrita:
    cd /data/DERI-ecgai/__users/Vandhanaa/
    source venv/bin/activate
    python verify_headline_table_v2.py

Schema confirmed from your actual file:
    columns: model, patient_id, fold, angle, centre, mae, mse
    74 rows = 37 patients x 2 models (M1, M2)
"""

import pandas as pd
import numpy as np

CV_RESULTS_CSV = "results/cv_results.csv"

# Values currently typed into the manuscript (Table 2)
MANUSCRIPT = {
    "M1": {"angle_mean": 11.33, "angle_std": 7.34, "centre_mean": 9.36, "centre_std": 5.72,
           "mae_mean": 4.84, "mse_mean": 40.10, "n_under_10": 18},
    "M2": {"angle_mean": 13.28, "angle_std": 9.19, "centre_mean": 16.64, "centre_std": 8.24,
           "mae_mean": 8.12, "mse_mean": 115.02, "n_under_10": 16},
}

TOL = 0.01

def check(label, computed, stated):
    diff = abs(computed - stated)
    flag = "OK      " if diff <= TOL else "MISMATCH"
    print(f"  [{flag}] {label}: computed={computed:.4f}  manuscript={stated:.4f}  diff={diff:.4f}")

def main():
    df = pd.read_csv(CV_RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {CV_RESULTS_CSV}")
    print(f"Unique patients: {df['patient_id'].nunique()}  (expect 37)")
    print(f"Models present: {df['model'].unique().tolist()}")

    # Sanity check: each patient should appear exactly once per model
    counts = df.groupby(["model", "patient_id"]).size()
    if (counts != 1).any():
        print("\n[WARNING] Some patient/model pairs appear more than once — check for duplicate rows:")
        print(counts[counts != 1])

    for model in ["M1", "M2"]:
        sub = df[df["model"] == model]
        n = len(sub)
        print(f"\n=== {model} (n={n}) ===")

        angle_mean = sub["angle"].mean()
        angle_std  = sub["angle"].std(ddof=1)
        centre_mean = sub["centre"].mean()
        centre_std  = sub["centre"].std(ddof=1)
        mae_mean = sub["mae"].mean()
        mse_mean = sub["mse"].mean()
        n_under_10 = (sub["angle"] <= 10).sum()

        m = MANUSCRIPT[model]
        check("angle mean (deg)", angle_mean, m["angle_mean"])
        check("angle std (deg)", angle_std, m["angle_std"])
        check("centre mean (mm)", centre_mean, m["centre_mean"])
        check("centre std (mm)", centre_std, m["centre_std"])
        check("centre MAE mean (mm)", mae_mean, m["mae_mean"])
        check("centre MSE mean (mm^2)", mse_mean, m["mse_mean"])
        if n_under_10 != m["n_under_10"]:
            print(f"  [MISMATCH] <=10 deg count: computed={n_under_10}  manuscript={m['n_under_10']}")
        else:
            print(f"  [OK      ] <=10 deg count: {n_under_10}")

    # Paired Wilcoxon cross-check (recompute independently)
    from scipy.stats import wilcoxon
    m1 = df[df["model"] == "M1"].set_index("patient_id")
    m2 = df[df["model"] == "M2"].set_index("patient_id")
    common = m1.index.intersection(m2.index)
    print(f"\n=== Wilcoxon recomputation (n={len(common)}) ===")
    stat_angle, p_angle = wilcoxon(m1.loc[common, "angle"], m2.loc[common, "angle"])
    stat_centre, p_centre = wilcoxon(m1.loc[common, "centre"], m2.loc[common, "centre"])
    print(f"  angle:  statistic={stat_angle:.3f}  p={p_angle:.4f}  (manuscript claims p=0.32, not significant)")
    print(f"  centre: statistic={stat_centre:.3f}  p={p_centre:.6f}  (manuscript claims p<0.0001, significant)")

if __name__ == "__main__":
    main()