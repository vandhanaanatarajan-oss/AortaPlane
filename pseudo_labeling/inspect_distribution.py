import csv

with open("/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_manifest.csv") as f:
    reader = csv.DictReader(f)
    mean_liks = sorted(float(row["mean_likelihood"]) for row in reader)

n = len(mean_liks)
percentiles = [10, 25, 50, 75, 90]
print(f"Total cases: {n}")
for p in percentiles:
    idx = int(n * p / 100)
    print(f"  p{p}: {mean_liks[idx]:.6f}")
print(f"  min: {mean_liks[0]:.6f}, max: {mean_liks[-1]:.6f}")

# Count how many cases would survive at a few candidate thresholds
for thresh in [0.01, 0.05, 0.1, 0.2, 0.3]:
    kept = sum(1 for v in mean_liks if v >= thresh)
    print(f"  threshold >= {thresh}: keeps {kept}/{n} cases ({100*kept/n:.1f}%)")
