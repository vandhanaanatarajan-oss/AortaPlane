#!/usr/bin/env python3
"""
Extends the project's strongest existing finding (cross-model disagreement
correlating with error, Spearman rho=0.358, p=0.0297 in the M1/M2 work) to
the new V1 PartialUnfreeze model: does 5-fold ensemble disagreement predict
which cases have high true error?

Disagreement: for each of the 18 real patients, spread (std, converted to
mm) across all 5 folds' independent predictions -- computed here.
True error: each patient's error from whichever fold's OWN validation set
included them (the standard, unbiased CV error) -- pulled from the 5 folds'
already-computed summary_mm.json files, same data used for the pooled CV
number (2.53mm).

CAVEAT (important, must be stated when reporting this): the disagreement
measure mixes 1 genuinely-held-out fold with 4 folds that saw each case
during training (in-sample), for practical/time reasons -- this is an
approximate/exploratory uncertainty signal, not a fully unbiased ensemble
disagreement estimate. Report accordingly, same honesty standard as
elsewhere in this project.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from nnlandmark.evaluation.nnLandmark.evaluate_landmark_predictions import aggregate_predictions_voxel

RESULTS_ROOT = Path("nnLM_data/results/Dataset741_AnnulusCT")
DISAGREEMENT_DIR = RESULTS_ROOT / "disagreement_analysis"
label_to_name = {"1": "annulus_1", "2": "annulus_2", "3": "annulus_3"}

# 1. Load all 5 folds' predictions on all 18 cases, aggregate to voxel coords
fold_predictions = {}  # {fold_idx: {case_id: {landmark_name: [x,y,z]}}}
for fold_idx in range(5):
    fold_dir = DISAGREEMENT_DIR / f"fold_{fold_idx}"
    fold_predictions[fold_idx] = aggregate_predictions_voxel(fold_dir, label_to_name)

all_case_ids = sorted(fold_predictions[0].keys())
print(f"Loaded predictions for {len(all_case_ids)} cases across 5 folds")

# 2. Compute per-case disagreement (mm), same spacing conversion as the
#    semi-supervised script's filter_by_agreement.
spacing = np.array([1.0, 0.877, 0.877])
disagreement_mm = {}
for case_id in all_case_ids:
    per_fold_coords = []
    for fold_idx in range(5):
        landmarks = fold_predictions[fold_idx][case_id]
        pts = np.array([landmarks["annulus_1"], landmarks["annulus_2"], landmarks["annulus_3"]])
        per_fold_coords.append(pts)
    stacked = np.stack(per_fold_coords)  # (5, 3, 3)
    std_voxel = stacked.std(axis=0)       # (3, 3) landmarks x xyz
    std_mm = std_voxel * spacing
    disagreement_mm[case_id] = float(std_mm.mean())

# 3. Pull each case's TRUE error from whichever fold's own validation
#    summary_mm.json included it (matches the pooled-CV data exactly).
true_error_mm = {}
for fold_idx in range(5):
    val_path = (RESULTS_ROOT / "nnLandmark_PartialUnfreeze__nnUNetPlansV1Match__3d_fullres"
                / f"fold_{fold_idx}" / "validation" / "summary_mm.json")
    with open(val_path) as f:
        val_data = json.load(f)
    for case_id, per_landmark in val_data["detailed_results"].items():
        true_error_mm[case_id] = float(np.mean(list(per_landmark.values())))

print(f"True error available for {len(true_error_mm)} cases")

# 4. Correlate
common_cases = sorted(set(disagreement_mm) & set(true_error_mm))
print(f"\n{'case':40s} {'disagreement_mm':>16s} {'true_error_mm':>16s}")
for c in common_cases:
    print(f"{c:40s} {disagreement_mm[c]:16.3f} {true_error_mm[c]:16.3f}")

disagreement_vals = [disagreement_mm[c] for c in common_cases]
error_vals = [true_error_mm[c] for c in common_cases]

rho, p = spearmanr(disagreement_vals, error_vals)
print(f"\nSpearman correlation (disagreement vs true error): rho={rho:.3f}, p={p:.4f}, n={len(common_cases)}")
if p < 0.05:
    print("Result: SIGNIFICANT -- ensemble disagreement correlates with actual error, "
          "extending the project's disagreement-based uncertainty finding to nnLandmark.")
else:
    print("Result: not significant at alpha=0.05.")

print("\nCAVEAT: disagreement mixes 1 held-out fold with 4 in-sample folds per case "
      "(practical compromise, not a fully unbiased ensemble) -- report as an "
      "exploratory/supporting result, n=18.")