import csv

THRESHOLD = 0.3
IN_FILE = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_manifest.csv"
OUT_FILE = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_manifest_filtered.csv"

with open(IN_FILE) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if float(row["mean_likelihood"]) >= THRESHOLD]

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Kept {len(rows)} cases at threshold >= {THRESHOLD}")
print("Case IDs kept:")
for row in rows:
    print(f"  {row['case_id']}  (mean_likelihood={float(row['mean_likelihood']):.4f})")
