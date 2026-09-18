#!/usr/bin/env python3
import json, glob, os
import numpy as np
import SimpleITK as sitk

CENTRAL_DIR = "/data/DERI-ecgai/001_CTA_Segmention/data/Clipping/landmarks/annulus"
OWN_DIR = "/data/DERI-ecgai/__users/Vandhanaa/data_processed"

patient_ids = ["R_20", "R_22", "R_24", "R_25", "R_26", "R_28", "R_29"]

def read_central_mrk(path):
    with open(path) as f:
        d = json.load(f)
    pts = d["markups"][0]["controlPoints"]
    return np.array([p["position"] for p in pts])

def read_own_landmarks_mm(patient_dir, img_path):
    lm_path = f"{OWN_DIR}/{patient_dir}/visualisation/processed_data/{patient_dir}_processed_landmarks.json"
    with open(lm_path) as f:
        d = json.load(f)
    voxel_pts = np.array([
        d["landmarks"]["controlPoint_1_image"],
        d["landmarks"]["controlPoint_2_image"],
        d["landmarks"]["controlPoint_3_image"],
    ])
    img = sitk.ReadImage(img_path)
    mm_pts = np.array([img.TransformContinuousIndexToPhysicalPoint(pt.tolist()) for pt in voxel_pts])
    return mm_pts

print(f"{'Patient':<10} {'Mean dist to Central (mm)':>28}")
print("-" * 40)

for pid in patient_ids:
    matches = glob.glob(f"{CENTRAL_DIR}/CONTCT_{pid}_*.mrk.json")
    if not matches:
        print(f"{pid:<10} NO CENTRAL FILE FOUND")
        continue
    central_path = matches[0]
    central_pts = read_central_mrk(central_path)

    own_dirs = glob.glob(f"{OWN_DIR}/CONTCT_{pid}_*")
    if not own_dirs:
        print(f"{pid:<10} NO OWN PROJECT DATA FOUND")
        continue
    patient_dir = os.path.basename(own_dirs[0])

    img_path = f"{OWN_DIR}/{patient_dir}/visualisation/processed_data/{patient_dir}_processed_image.nii.gz"
    if not os.path.exists(img_path):
        print(f"{pid:<10} NO PROCESSED IMAGE FOUND")
        continue

    own_pts_mm = read_own_landmarks_mm(patient_dir, img_path)
    dist = np.linalg.norm(own_pts_mm - central_pts, axis=1)
    print(f"{pid:<10} {dist.mean():>28.2f}")
