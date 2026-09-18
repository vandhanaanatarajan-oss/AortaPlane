import pandas as pd
import numpy as np

RESULTS_DIR = "results"

unc = pd.read_csv(f"{RESULTS_DIR}/uncertainty_summary.csv")
err = pd.read_csv(f"{RESULTS_DIR}/ensemble_results.csv")

# --- Build one row per patient with both models' uncertainty + error side by side ---
unc_wide = unc.pivot(index="patient_id", columns="model",
                      values=["normal_uncertainty_deg", "centre_uncertainty_vx"])
unc_wide.columns = [f"{a}_{b}" for a, b in unc_wide.columns]
unc_wide = unc_wide.reset_index()

df = unc_wide.merge(err, on="patient_id", how="inner")

if len(df) == 0:
    raise SystemExit("No matching patients between uncertainty_summary.csv and "
                      "ensemble_results.csv — check patient_id spelling matches.")

print(f"Patients: {len(df)}\n")

# --- Cross-model disagreement: how far apart are M1's and M2's own uncertainty estimates? ---
df["m2_m1_uncertainty_ratio"] = df["normal_uncertainty_deg_M2"] / (df["normal_uncertainty_deg_M1"] + 1e-6)

# --- Gating rule ---
# Rationale (from calibration_analysis.py):
#   - M1's own fold-disagreement does NOT predict its error (rho ~ 0)
#   - M2's own fold-disagreement DOES track its error reasonably (rho ~ 0.8)
#   - So: M2's uncertainty spiking relative to M1's is the actual warning sign,
#     not M1's uncertainty alone.
#
# Rule: if M2's uncertainty is much higher than M1's (ratio above threshold),
# flag the case for manual review rather than trusting the automatic ensemble mean.
RATIO_THRESHOLD = 3.0  # M2 uncertainty at least 3x M1's -> flag

df["flag_for_review"] = df["m2_m1_uncertainty_ratio"] > RATIO_THRESHOLD

print("=" * 90)
print(f"{'Patient':<38} {'M1_unc':>7} {'M2_unc':>7} {'ratio':>7} {'M1_err':>7} {'M2_err':>7}  Flag")
print("-" * 90)
for _, r in df.iterrows():
    flag_str = "REVIEW" if r["flag_for_review"] else "ok"
    print(f"{r['patient_id']:<38} {r['normal_uncertainty_deg_M1']:>7.2f} "
          f"{r['normal_uncertainty_deg_M2']:>7.2f} {r['m2_m1_uncertainty_ratio']:>7.2f} "
          f"{r['m1_angle']:>7.2f} {r['m2_angle']:>7.2f}  {flag_str}")
print("=" * 90)

# --- Does flagging actually catch the high-error cases? ---
print("\nDoes the flag correlate with which model is actually more accurate on that patient?")
for _, r in df.iterrows():
    if r["flag_for_review"]:
        better = "M2" if r["m2_angle"] < r["m1_angle"] else "M1"
        print(f"  {r['patient_id']}: flagged — {better} was actually more accurate "
              f"(M1={r['m1_angle']:.2f}°, M2={r['m2_angle']:.2f}°)")

df.to_csv(f"{RESULTS_DIR}/gating_rule_results.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/gating_rule_results.csv")

print("\nNOTE: threshold (3.0x) was chosen to flag the known R10 case as a sanity")
print("check, not fit from a validation set. Do not report this threshold as")
print("tuned/validated in the dissertation without testing it on more patients")
print("than the 5 in the current locked test set.")