#!/usr/bin/env python3
"""
Fixes Dataset743_AnnulusSemiSup_iter1's case-ID bug without re-running
prediction. Root cause: case_id was derived via raw_path.stem, which only
strips the LAST suffix -- for "X.nii.gz" files this leaves a stray ".nii"
in the case_id (e.g. "Subject001_CTA.nii" instead of "Subject001_CTA").

Confirmed effect: SimpleITK silently collapses a trailing "X.nii.nii.gz"
down to "X.nii.gz" when WRITING label maps (verified directly), but does
NOT touch ".nii" embedded mid-filename like "X.nii_0000.nii.gz" for
images. So for every case sourced from a .nii.gz raw file:
  - labelsTr/{case_id}.nii.gz got silently written CLEAN (no stray .nii)
  - imagesTr/{case_id}_0000.nii.gz kept the stray .nii, uncollapsed
  - all_landmarks_voxel.json / spacing.json keys still have the stray .nii

This script renames images (stripping the stray .nii) and fixes the two
JSON files' keys, so everything consistently uses the clean case name
that the labels already have -- no re-prediction needed.
"""
import json
from pathlib import Path

OUT_RAW = Path("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset743_AnnulusSemiSup_iter1")
IMAGES_TR = OUT_RAW / "imagesTr"
LABELS_TR = OUT_RAW / "labelsTr"

n_fixed = 0
n_ok = 0
rename_map = {}  # old_case_id -> clean_case_id

for img_file in sorted(IMAGES_TR.glob("*_0000.nii.gz")):
    dirty_case_id = img_file.name[: -len("_0000.nii.gz")]
    if not dirty_case_id.endswith(".nii"):
        n_ok += 1
        continue  # not affected -- image name is already clean

    clean_case_id = dirty_case_id[: -len(".nii")]
    clean_label_path = LABELS_TR / f"{clean_case_id}.nii.gz"
    if not clean_label_path.exists():
        print(f"WARNING: expected clean label for {dirty_case_id} not found "
              f"at {clean_label_path} -- skipping, needs manual check")
        continue

    new_img_path = IMAGES_TR / f"{clean_case_id}_0000.nii.gz"
    img_file.rename(new_img_path)
    rename_map[dirty_case_id] = clean_case_id
    n_fixed += 1
    print(f"Renamed image: {img_file.name} -> {new_img_path.name}")

print(f"\n{n_fixed} cases fixed, {n_ok} cases were already clean.")

# Fix JSON metadata keys to match
for json_name in ["all_landmarks_voxel.json", "spacing.json"]:
    path = OUT_RAW / json_name
    with open(path) as f:
        data = json.load(f)
    updated = {}
    for key, value in data.items():
        new_key = rename_map.get(key, key)  # rename if affected, else keep as-is
        updated[new_key] = value
    with open(path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated {len(rename_map)} keys in {json_name}")

print("\nDone. Re-run nnLM_extract_fingerprint / nnLM_plan_experiment / nnLM_preprocess now.")