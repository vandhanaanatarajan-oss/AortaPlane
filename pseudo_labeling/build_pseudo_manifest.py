import json, os, glob
import csv

OUTPUT_DIR = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/output"
MANIFEST_OUT = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_manifest.csv"

SKIP_FILES = {"dataset.json", "plans.json", "predict_from_raw_data_args.json"}

json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
json_files = [f for f in json_files if os.path.basename(f) not in SKIP_FILES]

print(f"Found {len(json_files)} per-case prediction files")

rows = []
for jf in json_files:
    case_id = os.path.basename(jf)
    # strip known suffixes to get a clean case name
    for suffix in ["_0000.nii.json", ".nii.json", ".json"]:
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]
            break

    with open(jf) as f:
        data = json.load(f)

    row = {"case_id": case_id}
    likelihoods = []
    for cls_id, entry in data.items():
        coord = entry["coordinates"]
        lik = entry["likelihood"]
        row[f"landmark_{cls_id}_x"] = coord[0]
        row[f"landmark_{cls_id}_y"] = coord[1]
        row[f"landmark_{cls_id}_z"] = coord[2]
        row[f"landmark_{cls_id}_likelihood"] = lik
        likelihoods.append(lik)

    row["min_likelihood"] = min(likelihoods) if likelihoods else None
    row["mean_likelihood"] = sum(likelihoods) / len(likelihoods) if likelihoods else None
    rows.append(row)

# Write CSV
if rows:
    fieldnames = ["case_id"] + [k for k in rows[0].keys() if k != "case_id"]
    with open(MANIFEST_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print(f"Wrote manifest with {len(rows)} cases to {MANIFEST_OUT}")

# Quick summary stats
mean_liks = [r["mean_likelihood"] for r in rows if r["mean_likelihood"] is not None]
min_liks = [r["min_likelihood"] for r in rows if r["min_likelihood"] is not None]
if mean_liks:
    mean_liks_sorted = sorted(mean_liks)
    print(f"mean_likelihood: min={min(mean_liks):.6f}, max={max(mean_liks):.6f}, "
          f"median={mean_liks_sorted[len(mean_liks_sorted)//2]:.6f}")
if min_liks:
    min_liks_sorted = sorted(min_liks)
    print(f"min_likelihood (worst landmark per case): min={min(min_liks):.6f}, max={max(min_liks):.6f}, "
          f"median={min_liks_sorted[len(min_liks_sorted)//2]:.6f}")
