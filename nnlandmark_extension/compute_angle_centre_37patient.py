#!/usr/bin/env python3
"""
Computes centroid-averaged centre error + plane-angle error (M1/M2's
exact metric definitions) for the 37-patient-trained nnLandmark model,
on the full 5-patient locked test set (Dataset741, full-scan, not the
separate cropped Dataset742).

Note: this is NOT the same as the "fair comparison vs M1/M2" table
(which needs Dataset742's cropped preprocessing, matching M1/M2's
easier task exactly) -- that comparison hasn't been rebuilt on 37
patients yet. This script instead gives the honest angle/centre error
for nnLandmark's own full-scan task, on the 37-patient model, using
the same underlying metric formulas for a like-for-like reading.
"""
import json
import numpy as np

PRED_DIR = ("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results/"
            "Dataset741_AnnulusCT/test_predictions_matched37_5fold")
RAW_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT"

TEST_PATIENTS = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8", "K10", "R10"]

with open(f"{PRED_DIR}/prediction_all_landmark_voxel.json") as f:
    pred = json.load(f)
with open(f"{RAW_DIR}/all_landmarks_voxel.json") as f:
    gt_all = json.load(f)
with open(f"{RAW_DIR}/spacing.json") as f:
    spacing_map = json.load(f)

LM_ORDER = ["annulus_1", "annulus_2", "annulus_3"]


def centroid_mm(case_landmarks, spacing):
    pts_vox = np.array([case_landmarks[name] for name in LM_ORDER], dtype=float)
    pts_mm = pts_vox * spacing
    return pts_mm.mean(axis=0), pts_mm


def plane_normal(pts_mm):
    v1 = pts_mm[1] - pts_mm[0]
    v2 = pts_mm[2] - pts_mm[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return None if norm == 0 else n / norm


centre_errors, angle_errors = [], []
print(f"{'case':40s} {'centre_err_mm':>14s} {'angle_err_deg':>14s}")
for case_id in TEST_PATIENTS:
    if case_id not in pred or case_id not in gt_all:
        print(f"SKIP {case_id}: missing")
        continue
    spacing = np.array(spacing_map[case_id]["image_spacing"])

    pred_centroid, pred_pts = centroid_mm(pred[case_id], spacing)
    gt_centroid, gt_pts = centroid_mm(gt_all[case_id], spacing)
    centre_err = float(np.linalg.norm(pred_centroid - gt_centroid))

    n_pred = plane_normal(pred_pts)
    n_gt = plane_normal(gt_pts)
    angle_err = float("nan")
    if n_pred is not None and n_gt is not None:
        cos_angle = np.clip(abs(np.dot(n_pred, n_gt)), -1.0, 1.0)
        angle_err = float(np.degrees(np.arccos(cos_angle)))

    centre_errors.append(centre_err)
    angle_errors.append(angle_err)
    print(f"{case_id:40s} {centre_err:14.3f} {angle_err:14.3f}")

centre_errors = np.array(centre_errors)
angle_errors = np.array(angle_errors)
print(f"\nnnLandmark, 37-patient model, full 5-patient test set (n={len(centre_errors)}):")
print(f"Centre error: {centre_errors.mean():.2f}mm +/- {centre_errors.std():.2f}mm")
print(f"Angle error:  {angle_errors.mean():.2f} deg +/- {angle_errors.std():.2f} deg")
print(f"\nFor comparison -- M1: 9.36 +/- 5.72mm centre, 11.33 +/- 7.34 deg angle")
print(f"                  M2: 16.64 +/- 8.24mm centre, 13.28 +/- 9.19 deg angle")
print("\nNote: this is nnLandmark's full-scan task (harder, unbounded search), not the")
print("same cropped task M1/M2 solve -- not a like-for-like 'fair comparison' number.")