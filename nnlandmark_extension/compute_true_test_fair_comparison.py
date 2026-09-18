#!/usr/bin/env python3
"""
Computes the TRUE test-set fair comparison against M1/M2 -- fixing the
earlier gap where the headline fair-comparison number (1.21mm/6.99deg)
was actually computed on fold-0 VALIDATION cases (patients the model DID
see in training, just not that fold), not on genuinely never-trained-on
TEST patients like M1/M2's own numbers are.

Uses the 3 patients confirmed valid for the nnLandmark test set
(CONTCT_R_08, CONTCT_R_27, D8) -- K10/R10 excluded per the documented,
investigated out-of-distribution limitation (see project notes).

Same centroid-averaged centre error + plane-angle error methodology as
compute_fair_comparison_metrics.py used before, applied here to genuine
test-set predictions instead of validation predictions.
"""
import json
import numpy as np

VAL_DIR = ("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results/"
           "Dataset742_AnnulusCropped/test_predictions_full5")
RAW_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped"

VALID_TEST_PATIENTS = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8"]

with open(f"{VAL_DIR}/prediction_all_landmark_voxel.json") as f:
    pred = json.load(f)
with open(f"{RAW_DIR}/all_landmarks_voxel.json") as f:
    gt_all = json.load(f)

LM_ORDER = ["annulus_1", "annulus_2", "annulus_3"]


def centroid(case_landmarks):
    pts = np.array([case_landmarks[name] for name in LM_ORDER], dtype=float)
    return pts.mean(axis=0), pts


def plane_normal(pts):
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return n / norm


centre_errors = []
angle_errors = []

print(f"{'case':40s} {'centre_err_mm':>14s} {'angle_err_deg':>14s}")
for case_id in VALID_TEST_PATIENTS:
    if case_id not in pred or case_id not in gt_all:
        print(f"SKIP {case_id}: missing from prediction or ground truth")
        continue
    pred_landmarks = pred[case_id]
    gt_landmarks = gt_all[case_id]

    pred_centroid, pred_pts = centroid(pred_landmarks)
    gt_centroid, gt_pts = centroid(gt_landmarks)
    centre_err = float(np.linalg.norm(pred_centroid - gt_centroid))

    n_pred = plane_normal(pred_pts)
    n_gt = plane_normal(gt_pts)
    if n_pred is None or n_gt is None:
        angle_err = float("nan")
    else:
        cos_angle = np.clip(abs(np.dot(n_pred, n_gt)), -1.0, 1.0)
        angle_err = float(np.degrees(np.arccos(cos_angle)))

    centre_errors.append(centre_err)
    angle_errors.append(angle_err)
    print(f"{case_id:40s} {centre_err:14.3f} {angle_err:14.3f}")

centre_errors = np.array(centre_errors)
angle_errors = np.array(angle_errors)

print(f"\nTRUE TEST SET (n={len(centre_errors)}, never trained on -- genuinely comparable to M1/M2):")
print(f"Centre error: {centre_errors.mean():.2f}mm +/- {centre_errors.std():.2f}mm")
print(f"Angle error:  {angle_errors.mean():.2f} deg +/- {angle_errors.std():.2f} deg")
print(f"\nFor comparison -- M1 reported: 9.36 +/- 5.72mm centre, 11.33 +/- 7.34 deg angle")
print(f"                  M2 reported: 16.64 +/- 8.24mm centre, 13.28 +/- 9.19 deg angle")
print("\nNote: n=3, K10/R10 excluded (documented out-of-distribution limitation, not cherry-picking) --")
print("still a small sample, but now a genuine like-for-like never-trained-on test-set comparison,")
print("not validation data as the earlier headline number used.")