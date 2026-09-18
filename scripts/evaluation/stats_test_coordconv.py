import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "results_coordconv"
df = pd.read_csv(f"{RESULTS_DIR}/ensemble_results_coordconv_vs_baseline.csv")

pairs = [
    ("angle",  "baseline_angle",  "coordconv_angle"),
    ("centre", "baseline_centre", "coordconv_centre"),
]

print(f"Paired patients: {len(df)}")

results = []
for name, base_col, cc_col in pairs:
    base_vals = df[base_col]
    cc_vals   = df[cc_col]
    n = len(df)

    stat, p = wilcoxon(base_vals, cc_vals)
    base_mean = base_vals.mean()
    cc_mean   = cc_vals.mean()
    sig = "significant" if p < 0.05 else "not significant"

    print(f"\n{name.upper()} — Wilcoxon signed-rank test")
    print(f"  n pairs        : {n}")
    print(f"  Baseline mean  : {base_mean:.3f}")
    print(f"  CoordConv mean : {cc_mean:.3f}")
    print(f"  statistic      : {stat:.3f}")
    print(f"  p-value        : {p:.4f}  ({sig} at alpha=0.05)")

    results.append(dict(
        metric=name, n=n,
        baseline_mean=round(float(base_mean), 3),
        coordconv_mean=round(float(cc_mean), 3),
        statistic=round(float(stat), 3),
        p_value=round(float(p), 4),
        significant=bool(p < 0.05),
    ))

out_df = pd.DataFrame(results)
out_df.to_csv(f"{RESULTS_DIR}/wilcoxon_results_coordconv.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/wilcoxon_results_coordconv.csv")

print("\nNOTE: n=5 paired test patients gives very low statistical power.")
print("A non-significant p-value here does not mean 'no effect' — it means")
print("this sample is too small to detect one reliably. Report the raw")
print("per-patient numbers and effect direction alongside this test, not")
print("the p-value alone.")