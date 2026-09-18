#!/usr/bin/env python3
"""
Recovers the two files convert_annulus_cropped.py should have written but
didn't: all_landmarks_voxel.json and spacing.json for Dataset742. Neither
requires touching the existing images/labels (which are already correct
and written) -- this only recomputes the landmark coordinates using the
IDENTICAL logic convert_annulus_cropped.py already used (same resample,
same crop-centering), so the images on disk and the coordinates in these
two files stay consistent.

AXIS ORDER: convert_annulus_cropped.py's landmarks_cropped are in
(D,H,W) = (Z,Y,X) order (numpy array convention). Dataset741's
all_landmarks_voxel.json (which nnLM_evaluate reads) uses (X,Y,Z) order
(SimpleITK image convention, matching GetSize()). This script converts
between the two -- CHECK this is right by spot-checking one case's output
against the training log line
"landmarks (crop-local, D/H/W): [...]" you already have from the
original run (should match after reversing to X,Y,Z).
"""
import json, glob, os
import numpy as np
import SimpleITK as sitk

BASE = "/data/DERI-ecgai/__users/Vandhanaa/data_processed"
OUT_RAW = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped"

RESAMPLE_SPACING = 1.0
CROP_SIZE = 128
TEST_PATIENTS = {"CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg"}


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


def crop_around_centre_origin(volume_shape, centre_vox, crop_size=128):
    D, H, W = volume_shape
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
    return np.array([d0, h0, w0], dtype=np.float32)


all_landmarks_voxel = {}
spacing_map = {}
n_done = 0

patient_dirs = sorted(glob.glob(os.path.join(BASE, "CONTCT_R_*")))
for pdir in patient_dirs:
    pname = os.path.basename(pdir)
    proc_dir = os.path.join(pdir, "visualisation", "processed_data")
    img_path = os.path.join(proc_dir, f"{pname}_processed_image.nii.gz")
    lm_path = os.path.join(proc_dir, f"{pname}_processed_landmarks.json")
    if not (os.path.exists(img_path) and os.path.exists(lm_path)):
        continue

    img = sitk.ReadImage(img_path)
    img_iso = resample_to_isotropic(img, RESAMPLE_SPACING)
    size_iso = img_iso.GetSize()  # (X,Y,Z)
    D, H, W = size_iso[2], size_iso[1], size_iso[0]

    with open(lm_path) as f:
        lm = json.load(f)

    pts_dhw = []
    for key in ["controlPoint_1_image", "controlPoint_2_image", "controlPoint_3_image"]:
        x, y, z = lm["landmarks"][key]
        phys = img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
        idx_resampled = img_iso.TransformPhysicalPointToContinuousIndex(phys)
        pts_dhw.append(np.array([idx_resampled[2], idx_resampled[1], idx_resampled[0]]))  # (D,H,W)

    centre_dhw = (pts_dhw[0] + pts_dhw[1] + pts_dhw[2]) / 3.0
    origin = crop_around_centre_origin((D, H, W), centre_dhw, CROP_SIZE)
    landmarks_cropped_dhw = [p - origin for p in pts_dhw]

    # Convert (D,H,W) -> (X,Y,Z) to match Dataset741's all_landmarks_voxel.json convention
    case_landmarks = {}
    for i, p in enumerate(landmarks_cropped_dhw, start=1):
        d, h, w = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        case_landmarks[f"annulus_{i}"] = [w, h, d]  # x=w, y=h, z=d

    all_landmarks_voxel[pname] = case_landmarks
    spacing_map[pname] = {"image_spacing": [RESAMPLE_SPACING, RESAMPLE_SPACING, RESAMPLE_SPACING],
                           "annotation_spacing": None}
    n_done += 1
    print(f"{pname}: annulus_1={case_landmarks['annulus_1']} "
          f"annulus_2={case_landmarks['annulus_2']} annulus_3={case_landmarks['annulus_3']}")

with open(os.path.join(OUT_RAW, "all_landmarks_voxel.json"), "w") as f:
    json.dump(all_landmarks_voxel, f, indent=2)
with open(os.path.join(OUT_RAW, "spacing.json"), "w") as f:
    json.dump(spacing_map, f, indent=2)

print(f"\nDone. Wrote all_landmarks_voxel.json and spacing.json for {n_done} cases to {OUT_RAW}")
print("CHECK: compare a case's annulus_1/2/3 above against that same case's "
      "\"landmarks (crop-local, D/H/W)\" line in your original convert_annulus_cropped.py "
      "run log -- after reversing D/H/W to X/Y/Z (i.e. w,h,d), the numbers should match exactly.")