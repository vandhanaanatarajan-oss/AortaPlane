#!/usr/bin/env python3
"""
Checks whether the 128mm crop built for K10/R10 (in add_dkr_cropped_test_cases.py)
actually contains the aorta, using each patient's own clinical segmentation
mask as independent ground truth for where the aorta really is.

Applies the IDENTICAL resample + crop pipeline to the segmentation mask
that was applied to the image, then checks what fraction of the
segmentation's total volume ends up inside the crop region.
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW_BASE = Path("/gpfs/DERI-ecgai/001_CTA_Segmention/data/Laura_shared/AC_cases_AS_OF_17042025")
RESAMPLE_SPACING = 1.0
CROP_SIZE = 128

PATIENTS = {
    "D8":  (RAW_BASE / "Completed__confirmed" / "D8", "D8_NN_LB.seg.nrrd"),
    "K10": (RAW_BASE / "Completed__uncertain__to_be_checked" / "K10", "K10_LB.seg.nrrd"),
    "R10": (RAW_BASE / "Completed__uncertain__to_be_checked" / "R10", "R10_seg_LB.nrrd"),
}
LABEL_ORDER = ["RC", "NC", "LC"]


def resample_to_isotropic(sitk_img, target_spacing=1.0, interpolator=sitk.sitkLinear, default_value=0.0):
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
    cd, ch, cw = int(np.clip(cd,0,D-1)), int(np.clip(ch,0,H-1)), int(np.clip(cw,0,W-1))
    crop_size = int(min(crop_size, D, H, W))
    half_c = crop_size // 2
    d0,h0,w0 = max(cd-half_c,0), max(ch-half_c,0), max(cw-half_c,0)
    d1,h1,w1 = min(d0+crop_size,D), min(h0+crop_size,H), min(w0+crop_size,W)
    d0,h0,w0 = max(d1-crop_size,0), max(h1-crop_size,0), max(w1-crop_size,0)
    return volume[d0:d1, h0:h1, w0:w1]


for pid, (pdir, seg_name) in PATIENTS.items():
    img = sitk.ReadImage(str(pdir / f"{pid}.nrrd"))
    img_iso = resample_to_isotropic(img, RESAMPLE_SPACING, sitk.sitkLinear)

    seg = sitk.ReadImage(str(pdir / seg_name))
    seg_iso = resample_to_isotropic(seg, RESAMPLE_SPACING, sitk.sitkNearestNeighbor)  # nearest-neighbor for label data
    seg_arr = sitk.GetArrayFromImage(seg_iso)  # (D,H,W)

    with open(pdir / f"{pid}_LB.mrk.json") as f:
        lm_data = json.load(f)
    cps = {cp["label"]: cp["position"] for cp in lm_data["markups"][0]["controlPoints"]}

    pts_dhw = []
    for name in LABEL_ORDER:
        idx = img_iso.TransformPhysicalPointToContinuousIndex(tuple(cps[name]))
        pts_dhw.append(np.array([idx[2], idx[1], idx[0]]))
    centre_dhw = (pts_dhw[0] + pts_dhw[1] + pts_dhw[2]) / 3.0

    total_seg_voxels = int((seg_arr > 0).sum())
    cropped_seg = crop_around_centre(seg_arr, centre_dhw, CROP_SIZE)
    voxels_in_crop = int((cropped_seg > 0).sum())
    pct_captured = 100.0 * voxels_in_crop / total_seg_voxels if total_seg_voxels > 0 else 0.0

    full_nz = np.argwhere(seg_arr > 0)
    print(f"=== {pid} ===")
    print(f"  Full segmentation: {total_seg_voxels} voxels, "
          f"bbox (D,H,W) min={full_nz.min(axis=0)} max={full_nz.max(axis=0)}")
    print(f"  Crop centre (D,H,W): {centre_dhw.round(1)}")
    print(f"  Segmentation voxels captured inside 128^3 crop: {voxels_in_crop} / {total_seg_voxels} "
          f"({pct_captured:.1f}%)")
    print()