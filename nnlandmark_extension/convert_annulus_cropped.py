#!/usr/bin/env python3
"""
Build Dataset742: same 128mm ground-truth-centered crop that M1/M2 use,
but fed into nnLandmark at full crop resolution (no resize to 64^3).
This isolates the "refinement within a known ROI" task from the
"full-scan localization" task, for a fair nnLandmark-vs-M1/M2 comparison.
"""
import json, glob, os
import numpy as np
import SimpleITK as sitk

BASE = "/data/DERI-ecgai/__users/Vandhanaa/data_processed"
OUT_RAW = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped"
IMAGES_TR = os.path.join(OUT_RAW, "imagesTr")
LABELS_TR = os.path.join(OUT_RAW, "labelsTr")
IMAGES_TS = os.path.join(OUT_RAW, "imagesTs")
LABELS_TS = os.path.join(OUT_RAW, "labelsTs")
for d in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS]:
    os.makedirs(d, exist_ok=True)

RESAMPLE_SPACING = 1.0
CROP_SIZE = 128
half = 1  # 3x3x3 landmark cube, same as convert_annulus.py

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


n_written = 0
n_test = 0
patient_dirs = sorted(glob.glob(os.path.join(BASE, "CONTCT_R_*")))

for pdir in patient_dirs:
    pname = os.path.basename(pdir)
    proc_dir = os.path.join(pdir, "visualisation", "processed_data")
    img_path = os.path.join(proc_dir, f"{pname}_processed_image.nii.gz")
    lm_path = os.path.join(proc_dir, f"{pname}_processed_landmarks.json")
    if not (os.path.exists(img_path) and os.path.exists(lm_path)):
        print(f"SKIP {pname}: missing files")
        continue

    img = sitk.ReadImage(img_path)
    img_iso = resample_to_isotropic(img, RESAMPLE_SPACING)
    arr = sitk.GetArrayFromImage(img_iso).astype(np.float32)  # (Z,Y,X) = (D,H,W)

    with open(lm_path) as f:
        lm = json.load(f)

    # landmarks are voxel-space (x,y,z) in the ORIGINAL (pre-resample) image;
    # map into the resampled image's voxel grid via physical coordinates
    pts_dhw = []
    for key in ["controlPoint_1_image", "controlPoint_2_image", "controlPoint_3_image"]:
        x, y, z = lm["landmarks"][key]
        phys = img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
        idx_resampled = img_iso.TransformPhysicalPointToContinuousIndex(phys)  # (x,y,z) in resampled voxel space
        pts_dhw.append(np.array([idx_resampled[2], idx_resampled[1], idx_resampled[0]]))  # -> (D,H,W)

    centre_dhw = (pts_dhw[0] + pts_dhw[1] + pts_dhw[2]) / 3.0

    roi, origin = crop_around_centre(arr, centre_dhw, CROP_SIZE)

    # landmark coords relative to the crop
    landmarks_cropped = [p - origin for p in pts_dhw]  # each is (d,h,w) relative to crop

    is_test = pname in TEST_PATIENTS
    img_out_dir = IMAGES_TS if is_test else IMAGES_TR
    lbl_out_dir = LABELS_TS if is_test else LABELS_TR

    roi_sitk = sitk.GetImageFromArray(roi)
    roi_sitk.SetSpacing([RESAMPLE_SPACING] * 3)
    sitk.WriteImage(roi_sitk, os.path.join(img_out_dir, f"{pname}_0000.nii.gz"))

    label_arr = np.zeros(roi.shape, dtype=np.uint8)
    for i, p in enumerate(landmarks_cropped, start=1):
        d, h, w = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        d0, d1 = max(0, d - half), min(roi.shape[0], d + half + 1)
        h0, h1 = max(0, h - half), min(roi.shape[1], h + half + 1)
        w0, w1 = max(0, w - half), min(roi.shape[2], w + half + 1)
        label_arr[d0:d1, h0:h1, w0:w1] = i

    label_sitk = sitk.GetImageFromArray(label_arr)
    label_sitk.CopyInformation(roi_sitk)
    sitk.WriteImage(label_sitk, os.path.join(lbl_out_dir, f"{pname}.nii.gz"))

    if is_test:
        n_test += 1
    else:
        n_written += 1
    print(f"{'TEST' if is_test else 'TRAIN'}: {pname} -> crop shape {roi.shape}, "
          f"landmarks (crop-local, D/H/W): {[p.tolist() for p in landmarks_cropped]}")

print(f"\nDone. Train: {n_written}, Test: {n_test}")

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "annulus_1": 1, "annulus_2": 2, "annulus_3": 3},
    "numTraining": n_written,
    "file_ending": ".nii.gz",
    "name": "Dataset742_AnnulusCropped"
}
with open(os.path.join(OUT_RAW, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)
print(f"Wrote dataset.json with numTraining={n_written}")
