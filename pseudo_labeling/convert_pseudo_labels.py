#!/usr/bin/env python3
import json, csv, os
import numpy as np
import SimpleITK as sitk

FILTERED_MANIFEST = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_manifest_filtered.csv"
INPUT_DIR = "/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/input"
OUT_RAW = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT"
IMAGES_TR = os.path.join(OUT_RAW, "imagesTr")
LABELS_TR = os.path.join(OUT_RAW, "labelsTr")
os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)

half = 1  # same 3x3x3 cube as convert_annulus.py

def find_source_image(case_id):
    """Handle the stray '.nii' left in some case_ids by convert_manifest.py's
    splitext() on double-extension .nii.gz files."""
    candidates = [
        os.path.join(INPUT_DIR, f"{case_id}_0000.nii.gz"),
        os.path.join(INPUT_DIR, f"{case_id}.nii_0000.nii.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

with open(FILTERED_MANIFEST) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Converting {len(rows)} pseudo-labeled cases")

n_written = 0
n_skipped = 0
pseudo_case_ids = []
for row in rows:
    case_id = f"PSEUDO_{row['case_id']}"
    src_img_path = find_source_image(row['case_id'])

    if src_img_path is None:
        print(f"SKIP {case_id}: source image not found under either naming variant")
        n_skipped += 1
        continue

    img = sitk.ReadImage(src_img_path)
    size = img.GetSize()  # (x,y,z)

    out_img_path = os.path.join(IMAGES_TR, f"{case_id}_0000.nii.gz")
    sitk.WriteImage(img, out_img_path)

    label_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)  # z,y,x
    for cls_id in [1, 2, 3]:
        x = int(round(float(row[f"landmark_{cls_id}_x"])))
        y = int(round(float(row[f"landmark_{cls_id}_y"])))
        z = int(round(float(row[f"landmark_{cls_id}_z"])))
        x0, x1 = max(0, x-half), min(size[0], x+half+1)
        y0, y1 = max(0, y-half), min(size[1], y+half+1)
        z0, z1 = max(0, z-half), min(size[2], z+half+1)
        label_arr[z0:z1, y0:y1, x0:x1] = cls_id

    label_img = sitk.GetImageFromArray(label_arr)
    label_img.CopyInformation(img)
    sitk.WriteImage(label_img, os.path.join(LABELS_TR, f"{case_id}.nii.gz"))

    pseudo_case_ids.append(case_id)
    n_written += 1
    print(f"Converted {case_id}")

print(f"Done. Wrote {n_written} pseudo-labeled cases, skipped {n_skipped}, into {OUT_RAW}")

with open("/data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/pseudo_case_ids.json", "w") as f:
    json.dump(pseudo_case_ids, f, indent=2)
