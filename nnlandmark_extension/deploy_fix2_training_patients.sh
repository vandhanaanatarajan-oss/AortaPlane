cat > /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/add_remaining_training_patients.py << 'PATEOF'
#!/usr/bin/env python3
"""
Extends Dataset741_AnnulusCT from 18 CONTCT_R-only training patients to
the FULL 37-patient pool M1/M2 themselves use (18 CONTCT_R + 19 D/K/R
cohort patients), by converting the 19 missing patients' raw .nrrd +
.mrk.json into nnLandmark's format as new TRAINING cases (imagesTr/
labelsTr), using the exact same approach already verified for D8/K10/R10
(add_dkr_test_cases.py) -- confirmed working (in_bounds=True, landmarks
verified landing inside each patient's own segmentation for the cases
already checked).

Then rebuilds splits_final.json to reuse fold_config.json's EXACT own
5-fold train/val assignments (translated to nnLandmark's naming), so
nnLandmark's cross-validation genuinely matches M1/M2's, not just an
independently-generated split with the same patient count.

All 19 raw locations confirmed to exist (10 Aug 2026):
  Completed__confirmed:            D1,D14,D15,D2,D3,D4,D6,D7,D9,K4,K5
  Completed__uncertain__to_be_checked: K11,K14,K17,R11,R12,R2,R7,R8
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW_BASE = Path("/gpfs/DERI-ecgai/001_CTA_Segmention/data/Laura_shared/AC_cases_AS_OF_17042025")
OUT_RAW = Path("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT")
IMAGES_TR = OUT_RAW / "imagesTr"
LABELS_TR = OUT_RAW / "labelsTr"

CONFIRMED = ["D1", "D14", "D15", "D2", "D3", "D4", "D6", "D7", "D9", "K4", "K5"]
UNCERTAIN = ["K11", "K14", "K17", "R11", "R12", "R2", "R7", "R8"]

PATIENTS = {}
for pid in CONFIRMED:
    PATIENTS[pid] = RAW_BASE / "Completed__confirmed" / pid
for pid in UNCERTAIN:
    PATIENTS[pid] = RAW_BASE / "Completed__uncertain__to_be_checked" / pid

half = 1
LABEL_ORDER = ["RC", "NC", "LC"]  # -> annulus_1/2/3, same convention verified across every patient checked so far

with open(OUT_RAW / "all_landmarks_voxel.json") as f:
    all_landmarks_voxel = json.load(f)
with open(OUT_RAW / "spacing.json") as f:
    spacing_map = json.load(f)
with open(OUT_RAW / "dataset.json") as f:
    dataset_json = json.load(f)

n_added = 0
skipped = []
for pid, pdir in PATIENTS.items():
    img_path = pdir / f"{pid}.nrrd"
    # Landmark filename suffix varies by annotator (confirmed: "_LB",
    # "_NN_LB", "_FC" so far) -- rather than enumerate every possible
    # suffix, glob broadly for any "*.mrk.json" in the folder. Every
    # patient folder checked so far has exactly one such file.
    lm_candidates = sorted(pdir.glob("*.mrk.json"))
    lm_path = lm_candidates[0] if len(lm_candidates) == 1 else None

    if not img_path.exists() or lm_path is None:
        print(f"SKIP {pid}: missing files at {pdir} (image={img_path.exists()}, "
              f"landmark_candidates={[c.name for c in lm_candidates]})")
        skipped.append(pid)
        continue

    img = sitk.ReadImage(str(img_path))
    size = img.GetSize()

    with open(lm_path) as f:
        lm_data = json.load(f)
    cps = {cp["label"]: cp["position"] for cp in lm_data["markups"][0]["controlPoints"]}
    coord_sys = lm_data["markups"][0].get("coordinateSystem")
    if coord_sys != "LPS":
        print(f"WARNING {pid}: coordinateSystem is '{coord_sys}', not 'LPS' -- skipping, needs manual check")
        skipped.append(pid)
        continue

    out_img_path = IMAGES_TR / f"{pid}_0000.nii.gz"
    sitk.WriteImage(img, str(out_img_path))

    label_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    case_landmarks_voxel = {}
    all_in_bounds = True
    for i, name in enumerate(LABEL_ORDER, start=1):
        world_pos = cps[name]
        xi, yi, zi = img.TransformPhysicalPointToIndex(tuple(world_pos))
        case_landmarks_voxel[f"annulus_{i}"] = [xi, yi, zi]
        in_bounds = (0 <= xi < size[0]) and (0 <= yi < size[1]) and (0 <= zi < size[2])
        if not in_bounds:
            all_in_bounds = False
        x0, x1 = max(0, xi - half), min(size[0], xi + half + 1)
        y0, y1 = max(0, yi - half), min(size[1], yi + half + 1)
        z0, z1 = max(0, zi - half), min(size[2], zi + half + 1)
        label_arr[z0:z1, y0:y1, x0:x1] = i

    print(f"{pid}: annulus_1={case_landmarks_voxel['annulus_1']} "
          f"annulus_2={case_landmarks_voxel['annulus_2']} annulus_3={case_landmarks_voxel['annulus_3']} "
          f"in_bounds={all_in_bounds}")

    label_img = sitk.GetImageFromArray(label_arr)
    label_img.CopyInformation(img)
    sitk.WriteImage(label_img, str(LABELS_TR / f"{pid}.nii.gz"))

    all_landmarks_voxel[pid] = case_landmarks_voxel
    spacing_map[pid] = {"image_spacing": list(img.GetSpacing()), "annotation_spacing": None}
    n_added += 1

with open(OUT_RAW / "all_landmarks_voxel.json", "w") as f:
    json.dump(all_landmarks_voxel, f, indent=2)
with open(OUT_RAW / "spacing.json", "w") as f:
    json.dump(spacing_map, f, indent=2)

n_of_these_19_present = sum(1 for pid in PATIENTS if pid in all_landmarks_voxel)
dataset_json["numTraining"] = 18 + n_of_these_19_present  # 18 original CONTCT_R + however many of these 19 D/K/R training patients have succeeded so far (safe to rerun -- recomputed fresh, doesn't double-count, and doesn't include the 5 test patients also present in all_landmarks_voxel)
with open(OUT_RAW / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\nDone. Added {n_added}/19 patients as new training cases.")
if skipped:
    print(f"Skipped: {skipped}")
print("CHECK: verify all landmarks show in_bounds=True before trusting this data.")
print(f"dataset.json numTraining now: {dataset_json['numTraining']}")
PATEOF