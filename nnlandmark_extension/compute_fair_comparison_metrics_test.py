#!/usr/bin/env python3
"""
Computes the TRUE apples-to-apples comparison against M1/M2's reported
9.36mm/16.64mm centre error and 11.33°/13.28° angle error:

  - Centre error: predicted 3-landmark CENTROID vs ground-truth centroid
    (matches M1/M2's methodology exactly -- errors partially cancel,
    unlike the per-landmark-independent MRE already in summary_mm.json).
  - Angle error: angle between the plane normal defined by the 3
    predicted points vs the plane normal defined by the 3 ground-truth
    points.

Dataset742 is 1.0mm isotropic (confirmed via spacing.json), so voxel
distances equal mm distances directly -- no further spacing conversion
needed here.
"""
import json
import numpy as np

VAL_DIR = ("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results/"
           "Dataset742_AnnulusCropped/test_predictions_full5")
RAW_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset742_AnnulusCropped"

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
for case_id, pred_landmarks in pred.items():
    if case_id not in gt_all:
        print(f"SKIP {case_id}: not in ground truth")
        continue
    gt_landmarks = gt_all[case_id]

    pred_centroid, pred_pts = centroid(pred_landmarks)
    gt_centroid, gt_pts = centroid(gt_landmarks)
    centre_err = float(np.linalg.norm(pred_centroid - gt_centroid))

    n_pred = plane_normal(pred_pts)
    n_gt = plane_normal(gt_pts)
    if n_pred is None or n_gt is None:
        angle_err = float("nan")
    else:
        cos_angle = np.clip(abs(np.dot(n_pred, n_gt)), -1.0, 1.0)  # abs() handles normal sign ambiguity
        angle_err = float(np.degrees(np.arccos(cos_angle)))

    centre_errors.append(centre_err)
    angle_errors.append(angle_err)
    print(f"{case_id:40s} {centre_err:14.3f} {angle_err:14.3f}")

centre_errors = np.array(centre_errors)
angle_errors = np.array(angle_errors)

print(f"\nCentre error: {centre_errors.mean():.2f}mm +/- {centre_errors.std():.2f}mm "
      f"(n={len(centre_errors)})")
print(f"Angle error:  {angle_errors.mean():.2f} deg +/- {angle_errors.std():.2f} deg "
      f"(n={len(angle_errors)})")
print(f"\nFor comparison -- M1 reported: 9.36 +/- 5.72mm centre, 11.33 +/- 7.34 deg angle")
print(f"                  M2 reported: 16.64 +/- 8.24mm centre, 13.28 +/- 9.19 deg angle")
print("\nNote: n=4 (fold-0 validation only) -- high variance expected, same caveat "
      "as your other fold-0-only results this session.")