#!/usr/bin/env python3
"""
Adds D8, K10, R10 as TEST cases to Dataset742_AnnulusCropped, using the
EXACT same crop-and-resize preprocessing as M1/M2 (128mm crop centred on
the true landmark centroid, resampled to 1.0mm isotropic) -- to test
whether this normalization step (which erases raw scan spacing
differences) also fixes the out-of-distribution failure seen on K10/R10
in the full-resolution Dataset741 test.

Unlike convert_annulus_cropped.py's original CONTCT_R patients (which
start from an already-voxel-space intermediate landmark file), D8/K10/R10
give us raw WORLD-SPACE (mm) landmarks directly from their .mrk.json
files -- actually a SIMPLER starting point, since we can go straight from
world mm to the resampled image's voxel space in one step, no
original-image voxel round-trip needed.

TEST-ONLY: no retraining. Uses the already-trained Dataset742 model
(nnLandmark__nnUNetPlans__3d_fullres, from-scratch, fold_0).
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW_BASE = Path("/gpfs/DERI-ecgai/001_CTA_Segmention/data/Laura_shared/AC_cases_AS_OF_17042025")
OUT_RAW = Path("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped")
IMAGES_TS = OUT_RAW / "imagesTs"
LABELS_TS = OUT_RAW / "labelsTs"
IMAGES_TS.mkdir(parents=True, exist_ok=True)
LABELS_TS.mkdir(parents=True, exist_ok=True)

PATIENTS = {
    "D8":  RAW_BASE / "Completed__confirmed" / "D8",
    "K10": RAW_BASE / "Completed__uncertain__to_be_checked" / "K10",
    "R10": RAW_BASE / "Completed__uncertain__to_be_checked" / "R10",
}

RESAMPLE_SPACING = 1.0
CROP_SIZE = 128
half = 1
LABEL_ORDER = ["RC", "NC", "LC"]  # -> annulus_1, annulus_2, annulus_3 (same assumption as Dataset741 addition, verified there)


def resample_to_isotropic(sitk_img, target_spacing=1.0, interpolator=sitk.sitkLinear, default_value=-1000.0):
    original_spacing = np.array(sitk_img.GetSpacing())
    original_size = np.array(sitk_img.GetSize())
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing)) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([target_spacing] * 3)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(default_value)
    resample.SetInterpolator(interpolator)
    return resample.Execute(sitk_img)


def crop_around_centre(volume, centre_vox, crop_size=128):
    D, H, W = volume.shape
    cd, ch, cw = np.asarray(centre_vox, dtype=np.float32).astype(int)
    cd = int(np.clip(cd, 0, D - 1))
    ch = int(np.clip(ch, 0, H - 1))
    cw = int(np.clip(cw, 0, W - 1))
    crop_size = int(min(crop_size, D, H, W))
    half_c = crop_size // 2
    d0 = max(cd - half_c, 0)
    h0 = max(ch - half_c, 0)
    w0 = max(cw - half_c, 0)
    d1 = min(d0 + crop_size, D)
    h1 = min(h0 + crop_size, H)
    w1 = min(w0 + crop_size, W)
    d0 = max(d1 - crop_size, 0)
    h0 = max(h1 - crop_size, 0)
    w0 = max(w1 - crop_size, 0)
    roi = volume[d0:d1, h0:h1, w0:w1]
    if roi.size == 0 or 0 in roi.shape:
        raise ValueError(f"Empty ROI crop: volume={volume.shape}, centre_vox={(cd,ch,cw)}")
    return roi, np.array([d0, h0, w0], dtype=np.float32)


with open(OUT_RAW / "all_landmarks_voxel.json") as f:
    all_landmarks_voxel = json.load(f)
with open(OUT_RAW / "spacing.json") as f:
    spacing_map = json.load(f)

n_added = 0
for pid, pdir in PATIENTS.items():
    img_path = pdir / f"{pid}.nrrd"
    lm_path = pdir / f"{pid}_LB.mrk.json"
    if not (img_path.exists() and lm_path.exists()):
        print(f"SKIP {pid}: missing files")
        continue

    img = sitk.ReadImage(str(img_path))
    img_iso = resample_to_isotropic(img, RESAMPLE_SPACING)
    arr = sitk.GetArrayFromImage(img_iso).astype(np.float32)  # (Z,Y,X) = (D,H,W)

    with open(lm_path) as f:
        lm_data = json.load(f)
    cps = {cp["label"]: cp["position"] for cp in lm_data["markups"][0]["controlPoints"]}
    coord_sys = lm_data["markups"][0].get("coordinateSystem")
    if coord_sys != "LPS":
        print(f"WARNING {pid}: coordinateSystem is '{coord_sys}', not 'LPS' -- skipping")
        continue

    # World mm -> resampled image's continuous voxel index, directly (no
    # original-image round-trip needed, since we start from real world
    # coordinates already -- simpler than convert_annulus_cropped.py's
    # CONTCT_R path).
    pts_dhw = []
    for name in LABEL_ORDER:
        world_pos = cps[name]
        idx_resampled = img_iso.TransformPhysicalPointToContinuousIndex(tuple(world_pos))  # (x,y,z)
        pts_dhw.append(np.array([idx_resampled[2], idx_resampled[1], idx_resampled[0]]))  # -> (D,H,W)

    centre_dhw = (pts_dhw[0] + pts_dhw[1] + pts_dhw[2]) / 3.0
    roi, origin = crop_around_centre(arr, centre_dhw, CROP_SIZE)
    landmarks_cropped_dhw = [p - origin for p in pts_dhw]

    roi_sitk = sitk.GetImageFromArray(roi)
    roi_sitk.SetSpacing([RESAMPLE_SPACING] * 3)
    sitk.WriteImage(roi_sitk, str(IMAGES_TS / f"{pid}_0000.nii.gz"))

    label_arr = np.zeros(roi.shape, dtype=np.uint8)
    case_landmarks_voxel = {}
    print(f"\n{pid}:")
    for i, p in enumerate(landmarks_cropped_dhw, start=1):
        d, h, w = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        in_bounds = (0 <= d < roi.shape[0]) and (0 <= h < roi.shape[1]) and (0 <= w < roi.shape[2])
        print(f"  annulus_{i} ({LABEL_ORDER[i-1]}): crop voxel (d,h,w)=({d},{h},{w})  in_bounds={in_bounds}")
        case_landmarks_voxel[f"annulus_{i}"] = [w, h, d]  # x=w,y=h,z=d -- matches Dataset742's existing (X,Y,Z) convention, verified earlier this project
        d0, d1 = max(0, d - half), min(roi.shape[0], d + half + 1)
        h0, h1 = max(0, h - half), min(roi.shape[1], h + half + 1)
        w0, w1 = max(0, w - half), min(roi.shape[2], w + half + 1)
        label_arr[d0:d1, h0:h1, w0:w1] = i

    label_sitk = sitk.GetImageFromArray(label_arr)
    label_sitk.CopyInformation(roi_sitk)
    sitk.WriteImage(label_sitk, str(LABELS_TS / f"{pid}.nii.gz"))

    all_landmarks_voxel[pid] = case_landmarks_voxel
    spacing_map[pid] = {"image_spacing": [RESAMPLE_SPACING, RESAMPLE_SPACING, RESAMPLE_SPACING], "annotation_spacing": None}
    n_added += 1

with open(OUT_RAW / "all_landmarks_voxel.json", "w") as f:
    json.dump(all_landmarks_voxel, f, indent=2)
with open(OUT_RAW / "spacing.json", "w") as f:
    json.dump(spacing_map, f, indent=2)

print(f"\nDone. Added {n_added}/3 cropped test cases to Dataset742.")
print("CHECK: verify all 'in_bounds=True' above before trusting predictions on these cases.")