#!/usr/bin/env python3
import json
import numpy as np
import SimpleITK as sitk

cases = {
    "CONTCT_R_18_FBA": "CONTCT_R_18_Pre_CT_20230724_FBA_landmarks_LB.mrk.json",
    "CONTCT_R_28_FBA_no_extended_seg": "CONTCT_R_28_Pre_CT_20201012_FBA_landmarks_LB2.mrk.json",
}

CENTRAL_DIR = "/data/DERI-ecgai/001_CTA_Segmention/data/Clipping/landmarks/annulus"
OWN_DIR = "/data/DERI-ecgai/__users/Vandhanaa/data_processed"

def read_central_mrk(path):
    with open(path) as f:
        d = json.load(f)
    pts = d["markups"][0]["controlPoints"]
    return np.array([p["position"] for p in pts])  # LPS mm, 3x3

def read_own_landmarks(patient_dir, img_path):
    lm_path = f"{OWN_DIR}/{patient_dir}/visualisation/processed_data/{patient_dir}_processed_landmarks.json"
    with open(lm_path) as f:
        d = json.load(f)
    voxel_pts = np.array([
        d["landmarks"]["controlPoint_1_image"],
        d["landmarks"]["controlPoint_2_image"],
        d["landmarks"]["controlPoint_3_image"],
    ])
    img = sitk.ReadImage(img_path)
    # convert voxel (x,y,z) to physical (LPS) mm using the image's transform
    mm_pts = np.array([img.TransformContinuousIndexToPhysicalPoint(pt.tolist()) for pt in voxel_pts])
    return mm_pts

for patient_dir, mrk_filename in cases.items():
    print(f"\n=== {patient_dir} ===")
    central_path = f"{CENTRAL_DIR}/{mrk_filename}"
    central_pts = read_central_mrk(central_path)
    print("Central (Clipping/landmarks/annulus) LPS mm:")
    print(central_pts)

    img_path = f"{OWN_DIR}/{patient_dir}/visualisation/processed_data/{patient_dir}_processed_image.nii.gz"
    own_pts_mm = read_own_landmarks(patient_dir, img_path)
    print("Own project (converted to LPS mm):")
    print(own_pts_mm)

    dist_to_central = np.linalg.norm(own_pts_mm - central_pts, axis=1)
    print(f"Distance own-vs-central per point (mm): {dist_to_central}")
    print(f"Mean distance to Central: {dist_to_central.mean():.2f} mm")
