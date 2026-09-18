import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = "results"

# --- Load uncertainty (long format: patient_id, model, normal_uncertainty_deg, ...) ---
unc = pd.read_csv(f"{RESULTS_DIR}/uncertainty_summary.csv")

# --- Load actual errors (wide format: m1_angle, m1_centre, m2_angle, m2_centre) ---
err = pd.read_csv(f"{RESULTS_DIR}/ensemble_results.csv")

# reshape errors to long format so it can be merged with uncertainty on (patient_id, model)
err_long = pd.concat([
    err[["patient_id", "m1_angle", "m1_centre"]].rename(
        columns={"m1_angle": "angle_error", "m1_centre": "centre_error"}
    ).assign(model="M1"),
    err[["patient_id", "m2_angle", "m2_centre"]].rename(
        columns={"m2_angle": "angle_error", "m2_centre": "centre_error"}
    ).assign(model="M2"),
], ignore_index=True)

df = unc.merge(err_long, on=["patient_id", "model"], how="inner")

if len(df) == 0:
    raise SystemExit(
        "No matching (patient_id, model) rows between uncertainty_summary.csv "
        "and ensemble_results.csv — check patient_id spellings match exactly."
    )

print(f"Matched {len(df)} patient x model rows ({df.patient_id.nunique()} patients, "
      f"{df.model.nunique()} models)\n")

# --- Does higher fold-disagreement (uncertainty) actually predict higher error? ---
print("=" * 70)
print("CALIBRATION CHECK: does uncertainty track actual error?")
print("=" * 70)
print("NOTE: n is very small here (5 test patients x models). Treat correlation")
print("values as directional signal only, not a statistically robust claim.\n")

results = []
for model in df["model"].unique():
    sub = df[df["model"] == model]
    if len(sub) < 3:
        continue
    rho_n, p_n = spearmanr(sub["normal_uncertainty_deg"], sub["angle_error"])
    rho_c, p_c = spearmanr(sub["centre_uncertainty_vx"], sub["centre_error"])
    print(f"{model}:")
    print(f"  normal_uncertainty_deg vs angle_error   -> spearman rho={rho_n:.3f}  p={p_n:.4f}")
    print(f"  centre_uncertainty_vx  vs centre_error  -> spearman rho={rho_c:.3f}  p={p_c:.4f}")
    results.append(dict(model=model, angle_rho=round(float(rho_n), 3), angle_p=round(float(p_n), 4),
                         centre_rho=round(float(rho_c), 3), centre_p=round(float(p_c), 4)))

# --- Flag "confidently wrong" cases: uncertainty below median, error above median ---
print("\n" + "=" * 70)
print("'CONFIDENTLY WRONG' FLAGGING")
print("=" * 70)
print("Flagged when normal_uncertainty_deg is below its model's median")
print("but angle_error is above its model's median (i.e. the model looked")
print("confident across folds, but the ensemble prediction was still off).\n")

flagged = []
for model in df["model"].unique():
    sub = df[df["model"] == model].copy()
    if len(sub) < 3:
        continue
    unc_median = sub["normal_uncertainty_deg"].median()
    err_median = sub["angle_error"].median()
    sub["confidently_wrong"] = (
        (sub["normal_uncertainty_deg"] < unc_median) & (sub["angle_error"] > err_median)
    )
    for _, r in sub.iterrows():
        marker = "  <-- CONFIDENTLY WRONG" if r["confidently_wrong"] else ""
        print(f"  {model} {r['patient_id']:<38} unc={r['normal_uncertainty_deg']:>6.2f}° "
              f"err={r['angle_error']:>6.2f}°{marker}")
        flagged.append(dict(model=model, patient_id=r["patient_id"],
                             normal_uncertainty_deg=r["normal_uncertainty_deg"],
                             angle_error=r["angle_error"],
                             confidently_wrong=bool(r["confidently_wrong"])))

# --- Save outputs ---
pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/calibration_correlation.csv", index=False)
pd.DataFrame(flagged).to_csv(f"{RESULTS_DIR}/calibration_flagged_cases.csv", index=False)

print(f"\nSaved: {RESULTS_DIR}/calibration_correlation.csv")
print(f"Saved: {RESULTS_DIR}/calibration_flagged_cases.csv")