import json, os
import SimpleITK as sitk

MANIFEST = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/manifest.json"
OUT_DIR = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/input"

with open(MANIFEST) as f:
    entries = json.load(f)

print(f"Total entries in manifest: {len(entries)}")

failed = []
for i, entry in enumerate(entries):
    src_path = entry["path"]
    source = entry["source"]
    case_id = os.path.splitext(os.path.basename(src_path))[0]
    out_name = f"{source}_{case_id}_0000.nii.gz"
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path):
        continue

    try:
        img = sitk.ReadImage(src_path)
        sitk.WriteImage(img, out_path)
    except Exception as e:
        print(f"FAILED: {src_path} -> {e}")
        failed.append(src_path)

    if (i + 1) % 20 == 0:
        print(f"Converted {i+1}/{len(entries)}")

print(f"Done. Failed: {len(failed)}")
if failed:
    print("Failed files:", failed)
