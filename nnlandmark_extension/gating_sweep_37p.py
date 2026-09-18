#!/usr/bin/env python3
"""
Gating-rule precision/recall/F1 sweep for nnLandmark's own 5-fold
ensemble disagreement (37-patient model), mirroring the M1/M2 gating
table methodology (comparison_table.md, cross-model disagreement).

Reuses the exact disagreement_mm / true_error_mm computation from
disagreement_correlation_37p.py (same source data as the rho=0.898
correlation). For each "bad" error definition, sweeps every observed
disagreement value as a candidate threshold and reports the threshold
that maximizes F1 (same spirit as the original table's threshold
selection), plus flagged/TP/precision/recall/F1 at that point.

NOTE: original M1/M2 table thresholds were defined in degrees (angle
error); nnLandmark's disagreement/error here are in mm (landmark
position, not plane angle) -- thresholds below are chosen from this
dataset's own error distribution (mean ~4.86mm, so >5/7/10mm loosely
mirrors the original table's >10/15/20 degree spacing relative to its
own mean), not a direct unit-for-unit match. State this explicitly
when reporting.
"""
import json
from pathlib import Path
import numpy as np

from nnlandmark.evaluation.nnLandmark.evaluate_landmark_predictions import aggregate_predictions_voxel

RESULTS_ROOT = Path("nnLM_data/results/Dataset741_AnnulusCT")
DISAGREEMENT_DIR = RESULTS_ROOT / "disagreement_analysis_37p"
VALIDATION_ROOT = RESULTS_ROOT / "nnLandmark_PartialUnfreeze__nnUNetPlansV1Match__3d_fullres"
SPLITS_PATH = Path("nnLM_data/preprocessed/Dataset741_AnnulusCT/splits_final.json")
label_to_name = {"1": "annulus_1", "2": "annulus_2", "3": "annulus_3"}

fold_predictions = {}
for fold_idx in range(5):
    fold_dir = DISAGREEMENT_DIR / f"fold_{fold_idx}"
    fold_predictions[fold_idx] = aggregate_predictions_voxel(fold_dir, label_to_name)

all_case_ids = sorted(fold_predictions[0].keys())
spacing = np.array([1.0, 0.877, 0.877])
disagreement_mm = {}
for case_id in all_case_ids:
    per_fold_coords = []
    for fold_idx in range(5):
        landmarks = fold_predictions[fold_idx][case_id]
        pts = np.array([landmarks["annulus_1"], landmarks["annulus_2"], landmarks["annulus_3"]])
        per_fold_coords.append(pts)
    stacked = np.stack(per_fold_coords)
    std_mm = stacked.std(axis=0) * spacing
    disagreement_mm[case_id] = float(std_mm.mean())

with open(SPLITS_PATH) as f:
    splits = json.load(f)

true_error_mm = {}
for fold_idx in range(5):
    val_path = VALIDATION_ROOT / f"fold_{fold_idx}" / "validation" / "summary_mm.json"
    with open(val_path) as f:
        val_data = json.load(f)
    summary_cases = set(val_data["detailed_results"].keys())
    split_cases = set(splits[fold_idx]["val"])
    if summary_cases != split_cases:
        raise RuntimeError(f"fold_{fold_idx}: summary_mm.json/splits_final.json mismatch -- fix before trusting this.")
    for case_id, per_landmark in val_data["detailed_results"].items():
        true_error_mm[case_id] = float(np.mean(list(per_landmark.values())))

common_cases = sorted(set(disagreement_mm) & set(true_error_mm))
print(f"n={len(common_cases)} cases loaded")
print(f"true error distribution: mean={np.mean([true_error_mm[c] for c in common_cases]):.2f}mm, "
      f"std={np.std([true_error_mm[c] for c in common_cases]):.2f}mm, "
      f"min={min(true_error_mm[c] for c in common_cases):.2f}mm, "
      f"max={max(true_error_mm[c] for c in common_cases):.2f}mm")

bad_definitions_mm = [5.0, 7.0, 10.0]
candidate_thresholds = sorted(set(disagreement_mm.values()))

print(f"\n{'bad defined as':>18s} {'threshold':>10s} {'flagged':>8s} {'TP':>4s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
for bad_thresh in bad_definitions_mm:
    is_bad = {c: true_error_mm[c] > bad_thresh for c in common_cases}
    n_bad = sum(is_bad.values())
    if n_bad == 0:
        print(f"error > {bad_thresh:>4.1f}mm: no cases exceed this threshold, skipping")
        continue

    best = None
    for thresh in candidate_thresholds:
        flagged = {c: disagreement_mm[c] >= thresh for c in common_cases}
        n_flagged = sum(flagged.values())
        if n_flagged == 0:
            continue
        tp = sum(1 for c in common_cases if flagged[c] and is_bad[c])
        precision = tp / n_flagged
        recall = tp / n_bad
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if best is None or f1 > best[0]:
            best = (f1, thresh, n_flagged, tp, precision, recall)

    f1, thresh, n_flagged, tp, precision, recall = best
    print(f"error > {bad_thresh:>4.1f}mm {thresh:>9.3f}mm {n_flagged:>8d} {tp:>4d} {precision:>10.2f} {recall:>8.2f} {f1:>6.2f}")

print("\nNOTE: thresholds are in mm (landmark disagreement/error), not degrees like the "
      "original M1/M2 table (plane angle) -- not a unit-for-unit comparison, report as "
      "nnLandmark's own analogous gating-rule result, not a direct replication.")
