"""Paired Wilcoxon: TTA vs baseline M1, leak-free n=37 CV patients."""
import argparse
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--baseline_csv", default="results/cv_results.csv")
ap.add_argument("--tta_csv", default="results/ensemble_results_tta.csv")
args = ap.parse_args()

baseline = pd.read_csv(args.baseline_csv)
baseline = baseline[baseline["model"] == "M1"][["patient_id", "angle", "centre"]].rename(
    columns={"angle": "angle_baseline", "centre": "centre_baseline"})

tta = pd.read_csv(args.tta_csv)
tta = tta[tta["eval_type"] == "cv_single_fold"][
    ["patient_id", "angle_error_deg_tta", "centre_error_mm_tta"]].rename(
    columns={"angle_error_deg_tta": "angle_tta", "centre_error_mm_tta": "centre_tta"})

merged = baseline.merge(tta, on="patient_id", how="inner")
print(f"Matched {len(merged)} patients (baseline had {len(baseline)}, tta had {len(tta)})")

missing_from_tta = set(baseline.patient_id) - set(tta.patient_id)
missing_from_baseline = set(tta.patient_id) - set(baseline.patient_id)
if missing_from_tta:
    print(f"In baseline but not TTA: {sorted(missing_from_tta)}")
if missing_from_baseline:
    print(f"In TTA but not baseline: {sorted(missing_from_baseline)}")

for metric in ["angle", "centre"]:
    base_col, tta_col = f"{metric}_baseline", f"{metric}_tta"
    diffs = merged[tta_col] - merged[base_col]
    n_improved = int((diffs < 0).sum())
    n_worsened = int((diffs > 0).sum())
    stat, p = wilcoxon(merged[base_col], merged[tta_col])
    print(f"\n--- {metric.upper()} ERROR ---")
    print(f"Baseline mean: {merged[base_col].mean():.3f}  TTA mean: {merged[tta_col].mean():.3f}")
    print(f"Improved: {n_improved}/{len(merged)}  Worsened: {n_worsened}/{len(merged)}")
    print(f"Wilcoxon: stat={stat:.3f}, p={p:.4f}")
    if p < 0.05:
        d = "IMPROVED" if merged[tta_col].mean() < merged[base_col].mean() else "WORSENED"
        print(f"=> Significant. TTA {d} {metric} error.")
    else:
        print(f"=> Not significant (p >= 0.05).")
