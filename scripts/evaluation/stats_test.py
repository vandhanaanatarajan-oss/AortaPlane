import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "results"

df = pd.read_csv(f"{RESULTS_DIR}/cv_results.csv")

angle_wide  = df.pivot(index="patient_id", columns="model", values="angle")
centre_wide = df.pivot(index="patient_id", columns="model", values="centre")

angle_wide  = angle_wide.dropna(subset=["M1", "M2"])
centre_wide = centre_wide.dropna(subset=["M1", "M2"])

print(f"Paired patients — angle: {len(angle_wide)}  centre: {len(centre_wide)}")

results = []

for name, wide in [("angle", angle_wide), ("centre", centre_wide)]:
    stat, p = wilcoxon(wide["M1"], wide["M2"])
    n = len(wide)
    m1_mean = wide["M1"].mean()
    m2_mean = wide["M2"].mean()
    sig = "significant" if p < 0.05 else "not significant"
    print(f"\n{name.upper()} — Wilcoxon signed-rank test")
    print(f"  n pairs   : {n}")
    print(f"  M1 mean   : {m1_mean:.3f}")
    print(f"  M2 mean   : {m2_mean:.3f}")
    print(f"  statistic : {stat:.3f}")
    print(f"  p-value   : {p:.4f}  ({sig} at alpha=0.05)")
    results.append(dict(metric=name, n=n, m1_mean=round(m1_mean,3), m2_mean=round(m2_mean,3),
                         statistic=round(float(stat),3), p_value=round(float(p),4), significant=(p < 0.05)))

out_df = pd.DataFrame(results)
out_df.to_csv(f"{RESULTS_DIR}/wilcoxon_results.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/wilcoxon_results.csv")