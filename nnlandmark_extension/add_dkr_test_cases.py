#!/usr/bin/env python3
"""
Adds D8, K10, R10 as extra TEST cases to Dataset741_AnnulusCT -- restoring
the fair comparison to the full original 5-patient locked test set (was
previously only 2/5, since only CONTCT_R patients had been converted).

These 3 are TEST-ONLY: no retraining needed, the already-trained model is
evaluated on them exactly like CONTCT_R_08/CONTCT_R_27.

Source data confirmed on Apocrita (7 Aug 2026):
  D8:  AC_cases_AS_OF_17042025/Completed__confirmed/D8/
  K10: AC_cases_AS_OF_17042025/Completed__uncertain__to_be_checked/K10/
  R10: AC_cases_AS_OF_17042025/Completed__uncertain__to_be_checked/R10/
Each has <ID>.nrrd (raw CT) and <ID>_LB.mrk.json (3 labelled control
points: RC/NC/LC, coordinateSystem: LPS -- confirmed, not the historical
mislabelled-RAS bug that affected the OLD 42-patient CONTCT_R data).

UNVERIFIED ASSUMPTION (flagged explicitly): annulus_1=RC, annulus_2=NC,
annulus_3=LC, based on RC/NC/LC always appearing in that file order across
every patient checked (R_08, R_15, D8, K10, R10), matching R_08's already-
converted all_landmarks_voxel.json entry by inference, not direct
confirmation. The script prints each landmark's converted voxel position
so you can sanity-check it lands in a plausible location (e.g. against a
quick slice view) before trusting the evaluation numbers.

Unlike the OLD CONTCT_R raw files (which had a coordinate-system bug --
world coordinates mislabelled, needed an explicit LPS<->RAS handling
fix), these 3 files explicitly declare "coordinateSystem": "LPS" and
SimpleITK's TransformPhysicalPointToIndex assumes LPS natively -- so this
is a direct, correct conversion with no sign-flip needed. Confirm this
assumption too by checking the printed voxel positions land inside the
image bounds and near the expected anatomical region.
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW_BASE = Path("/gpfs/DERI-ecgai/001_CTA_Segmention/data/Laura_shared/AC_cases_AS_OF_17042025")
OUT_RAW = Path("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT")
IMAGES_TS = OUT_RAW / "imagesTs"
LABELS_TS = OUT_RAW / "labelsTs"

PATIENTS = {
    "D8":  RAW_BASE / "Completed__confirmed" / "D8",
    "K10": RAW_BASE / "Completed__uncertain__to_be_checked" / "K10",
    "R10": RAW_BASE / "Completed__uncertain__to_be_checked" / "R10",
}

half = 1  # 3x3x3 cube, matches convert_annulus.py's existing convention
LABEL_ORDER = ["RC", "NC", "LC"]  # -> annulus_1, annulus_2, annulus_3 (see assumption above)

with open(OUT_RAW / "all_landmarks_voxel.json") as f:
    all_landmarks_voxel = json.load(f)
with open(OUT_RAW / "spacing.json") as f:
    spacing_map = json.load(f)
with open(OUT_RAW / "dataset.json") as f:
    dataset_json = json.load(f)

n_added = 0
for pid, pdir in PATIENTS.items():
    img_path = pdir / f"{pid}.nrrd"
    lm_path = pdir / f"{pid}_LB.mrk.json"
    if not (img_path.exists() and lm_path.exists()):
        print(f"SKIP {pid}: missing files at {pdir}")
        continue

    img = sitk.ReadImage(str(img_path))
    size = img.GetSize()

    with open(lm_path) as f:
        lm_data = json.load(f)
    cps = {cp["label"]: cp["position"] for cp in lm_data["markups"][0]["controlPoints"]}
    coord_sys = lm_data["markups"][0].get("coordinateSystem")
    if coord_sys != "LPS":
        print(f"WARNING {pid}: coordinateSystem is '{coord_sys}', not 'LPS' -- stop and check before trusting this case")
        continue

    out_img_path = IMAGES_TS / f"{pid}_0000.nii.gz"
    sitk.WriteImage(img, str(out_img_path))

    label_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    case_landmarks_voxel = {}
    print(f"\n{pid}:")
    for i, name in enumerate(LABEL_ORDER, start=1):
        world_pos = cps[name]
        xi, yi, zi = img.TransformPhysicalPointToIndex(tuple(world_pos))
        case_landmarks_voxel[f"annulus_{i}"] = [xi, yi, zi]
        in_bounds = (0 <= xi < size[0]) and (0 <= yi < size[1]) and (0 <= zi < size[2])
        print(f"  annulus_{i} ({name}): voxel=({xi},{yi},{zi})  in_bounds={in_bounds}")
        x0, x1 = max(0, xi - half), min(size[0], xi + half + 1)
        y0, y1 = max(0, yi - half), min(size[1], yi + half + 1)
        z0, z1 = max(0, zi - half), min(size[2], zi + half + 1)
        label_arr[z0:z1, y0:y1, x0:x1] = i

    label_img = sitk.GetImageFromArray(label_arr)
    label_img.CopyInformation(img)
    sitk.WriteImage(label_img, str(LABELS_TS / f"{pid}.nii.gz"))

    all_landmarks_voxel[pid] = case_landmarks_voxel
    spacing_map[pid] = {"image_spacing": list(img.GetSpacing()), "annotation_spacing": None}
    n_added += 1

with open(OUT_RAW / "all_landmarks_voxel.json", "w") as f:
    json.dump(all_landmarks_voxel, f, indent=2)
with open(OUT_RAW / "spacing.json", "w") as f:
    json.dump(spacing_map, f, indent=2)
# numTraining is unaffected (these are test-only); dataset.json doesn't track numTest,
# so no change needed there -- confirmed by inspecting the existing schema.

print(f"\nDone. Added {n_added}/3 patients as new test cases.")
print("CHECK: verify each 'in_bounds=True' above, and that the voxel positions look "
      "plausible before trusting predictions/evaluation on these 3 cases.")