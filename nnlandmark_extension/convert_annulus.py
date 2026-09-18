#!/usr/bin/env python3
import json, glob, os
import numpy as np
import SimpleITK as sitk

BASE = "/data/DERI-ecgai/__users/Vandhanaa/data_processed"
OUT_RAW = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT"
IMAGES_TR = os.path.join(OUT_RAW, "imagesTr")
LABELS_TR = os.path.join(OUT_RAW, "labelsTr")
os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)

name_to_label = {"annulus_1": 1, "annulus_2": 2, "annulus_3": 3}
all_landmarks_voxel = {}
spacing_map = {}
half = 1  # 3x3x3 cube

patient_dirs = sorted(glob.glob(os.path.join(BASE, "CONTCT_R_*")))
n_written = 0

for pdir in patient_dirs:
    pname = os.path.basename(pdir)
    proc_dir = os.path.join(pdir, "visualisation", "processed_data")
    img_path = os.path.join(proc_dir, f"{pname}_processed_image.nii.gz")
    lm_path = os.path.join(proc_dir, f"{pname}_processed_landmarks.json")
    if not (os.path.exists(img_path) and os.path.exists(lm_path)):
        print(f"SKIP {pname}: missing files")
        continue

    img = sitk.ReadImage(img_path)
    size = img.GetSize()  # (x,y,z)
    spacing = img.GetSpacing()

    with open(lm_path) as f:
        lm = json.load(f)

    case_id = pname
    out_img_path = os.path.join(IMAGES_TR, f"{case_id}_0000.nii.gz")
    sitk.WriteImage(img, out_img_path)

    label_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)  # z,y,x for array indexing
    case_landmarks_voxel = {}
    for i, key in enumerate(["controlPoint_1_image", "controlPoint_2_image", "controlPoint_3_image"], start=1):
        x, y, z = lm["landmarks"][key]
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        case_landmarks_voxel[f"annulus_{i}"] = [xi, yi, zi]
        x0, x1 = max(0, xi-half), min(size[0], xi+half+1)
        y0, y1 = max(0, yi-half), min(size[1], yi+half+1)
        z0, z1 = max(0, zi-half), min(size[2], zi+half+1)
        label_arr[z0:z1, y0:y1, x0:x1] = i

    label_img = sitk.GetImageFromArray(label_arr)
    label_img.CopyInformation(img)
    sitk.WriteImage(label_img, os.path.join(LABELS_TR, f"{case_id}.nii.gz"))

    all_landmarks_voxel[case_id] = case_landmarks_voxel
    spacing_map[case_id] = {"image_spacing": list(spacing), "annotation_spacing": None}
    n_written += 1
    print(f"Processed {case_id}")

with open(os.path.join(OUT_RAW, "all_landmarks_voxel.json"), "w") as f:
    json.dump(all_landmarks_voxel, f, indent=2)
with open(os.path.join(OUT_RAW, "spacing.json"), "w") as f:
    json.dump(spacing_map, f, indent=2)
with open(os.path.join(OUT_RAW, "name_to_label.json"), "w") as f:
    json.dump(name_to_label, f, indent=2)

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, **name_to_label},
    "numTraining": n_written,
    "file_ending": ".nii.gz",
    "name": "Dataset741_AnnulusCT"
}
with open(os.path.join(OUT_RAW, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\nDone. {n_written} cases written to {OUT_RAW}")
