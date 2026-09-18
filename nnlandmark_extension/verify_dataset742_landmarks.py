#!/usr/bin/env python3
"""
Sanity-checks fix_dataset742_metadata.py's recovered all_landmarks_voxel.json
against the ACTUAL label maps already on disk (burned by the original,
trusted convert_annulus_cropped.py run, never touched by the recovery
script). If the recovered voxel coordinate for e.g. annulus_1 falls inside
the region labelled "1" in that case's label map, the coordinates and the
X/Y/Z axis-order conversion are confirmed correct.
"""
import json
import numpy as np
import SimpleITK as sitk

OUT_RAW = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped"

with open(f"{OUT_RAW}/all_landmarks_voxel.json") as f:
    landmarks = json.load(f)

n_checked = 0
n_pass = 0
for case_id, case_landmarks in landmarks.items():
    label_path = f"{OUT_RAW}/labelsTr/{case_id}.nii.gz"
    try:
        label_img = sitk.ReadImage(label_path)
    except RuntimeError:
        label_path = f"{OUT_RAW}/labelsTs/{case_id}.nii.gz"
        try:
            label_img = sitk.ReadImage(label_path)
        except RuntimeError:
            print(f"SKIP {case_id}: no label map found in Tr or Ts")
            continue

    label_arr = sitk.GetArrayFromImage(label_img)  # (D,H,W) = (Z,Y,X)

    case_pass = True
    for i, name in enumerate(["annulus_1", "annulus_2", "annulus_3"], start=1):
        x, y, z = case_landmarks[name]  # recovered as X,Y,Z
        # label_arr is indexed [Z,Y,X] = [d,h,w]
        value_at_point = label_arr[z, y, x]
        if value_at_point != i:
            case_pass = False
            print(f"MISMATCH {case_id} {name}: recovered voxel (x={x},y={y},z={z}) "
                  f"-> label_arr[z,y,x]={value_at_point}, expected {i}")

    n_checked += 1
    if case_pass:
        n_pass += 1

print(f"\n{n_pass} / {n_checked} cases: all 3 landmarks land exactly on their labelled voxel.")
if n_pass == n_checked:
    print("PASS -- axis-order conversion confirmed correct.")
else:
    print("FAIL -- do not trust all_landmarks_voxel.json yet, axis order or coordinates are wrong.")
    