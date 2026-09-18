#!/usr/bin/env python3
"""
Computes MAE (Mean Absolute Error) for nnLandmark's PartialUnfreeze
5-fold ensemble test result, using the EXACT same definition confirmed
in the M1/M2 project's own eval_cv.py: mean of absolute PER-AXIS
differences (np.mean(np.abs(diff_mm))) -- NOT the same as MRE, which is
Euclidean/L2 distance. These are genuinely different metrics that
happen to both be reported in mm.

Uses the already-generated 5-fold ensemble test predictions
(test_predictions_PartialUnfreeze_5fold), no new inference needed.
"""
import json
import numpy as np

PRED_DIR = ("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results/"
            "Dataset741_AnnulusCT/test_predictions_matched37_5fold")
RAW_DIR = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw/Dataset741_AnnulusCT"

VALID_TEST_PATIENTS = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8", "K10", "R10"]  # all 5, matching M1/M2's own full locked test set -- 37-patient model handles all 5 reasonably now

with open(PRED_DIR + "/prediction_all_landmark_voxel.json") as f:
    pred = json.load(f)
with open(RAW_DIR + "/all_landmarks_voxel.json") as f:
    gt_all = json.load(f)
with open(RAW_DIR + "/spacing.json") as f:
    spacing_map = json.load(f)

LM_ORDER = ["annulus_1", "annulus_2", "annulus_3"]

all_abs_diffs_mm = []  # every |axis difference| in mm, across all landmarks/cases -- for the pooled MAE
per_case_mae = {}

for case_id in VALID_TEST_PATIENTS:
    if case_id not in pred or case_id not in gt_all:
        print(f"SKIP {case_id}: missing from prediction or ground truth")
        continue
    pred_landmarks = pred[case_id]
    gt_landmarks = gt_all[case_id]
    spacing = np.array(spacing_map[case_id]["image_spacing"])  # (3,) mm per voxel, per axis

    case_diffs = []
    for name in LM_ORDER:
        pred_pt = np.array(pred_landmarks[name], dtype=float)
        gt_pt = np.array(gt_landmarks[name], dtype=float)
        diff_voxel = pred_pt - gt_pt
        diff_mm = diff_voxel * spacing  # per-axis difference, voxels -> mm
        abs_diff_mm = np.abs(diff_mm)
        all_abs_diffs_mm.extend(abs_diff_mm.tolist())
        case_diffs.extend(abs_diff_mm.tolist())

    per_case_mae[case_id] = float(np.mean(case_diffs))

pooled_mae = float(np.mean(all_abs_diffs_mm))
pooled_std = float(np.std(all_abs_diffs_mm))

print(f"{'case':40s} {'MAE (mm)':>12s}")
for case_id, mae in per_case_mae.items():
    print(f"{case_id:40s} {mae:12.3f}")

print(f"\nPooled MAE (all axes, all landmarks, all {len(per_case_mae)} test cases): "
      f"{pooled_mae:.2f}mm +/- {pooled_std:.2f}mm (n={len(all_abs_diffs_mm)} axis-measurements)")
print("\nNote: MAE (mean absolute per-axis error) is a DIFFERENT metric from the MRE "
      "(mean Euclidean/radial error) reported elsewhere for this result (1.99mm) -- "
      "MAE is typically smaller since it doesn't compound errors across 3 axes via "
      "the Euclidean norm the way MRE does.")