#!/usr/bin/env python3
"""
37-patient extension of disagreement_correlation.py.

Recomputes the ensemble-disagreement vs true-CV-error correlation
(originally rho=0.830, p<0.0001, n=18) on the full 37-patient
matched-training-data model, using the same method exactly.

Disagreement: for each of the 37 patients, spread (std, converted to
mm) across all 5 folds' independent predictions on that patient
-- computed here, from disagreement_analysis_37p/.

True error: each patient's error from whichever fold's OWN validation
set included them (the standard, unbiased CV error) -- pulled from the
5 folds' summary_mm.json files (regenerated 11 Aug after a stale-data
bug was found and fixed at the source; each now contains exactly its
splits_final.json val set, verified zero extra/missing).

Same CAVEAT as the 18-patient version: the disagreement measure mixes
1 genuinely-held-out fold with 4 folds that saw each case during
training (in-sample), for practical/time reasons -- this is an
approximate/exploratory uncertainty signal, not a fully unbiased
ensemble disagreement estimate. Report accordingly.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from nnlandmark.evaluation.nnLandmark.evaluate_landmark_predictions import aggregate_predictions_voxel

RESULTS_ROOT = Path("nnLM_data/results/Dataset741_AnnulusCT")
DISAGREEMENT_DIR = RESULTS_ROOT / "disagreement_analysis_37p"
VALIDATION_ROOT = RESULTS_ROOT / "nnLandmark_PartialUnfreeze__nnUNetPlansV1Match__3d_fullres"
SPLITS_PATH = Path("nnLM_data/preprocessed/Dataset741_AnnulusCT/splits_final.json")
label_to_name = {"1": "annulus_1", "2": "annulus_2", "3": "annulus_3"}

# 1. Load all 5 folds' predictions on all 37 cases, aggregate to voxel coords
fold_predictions = {}  # {fold_idx: {case_id: {landmark_name: [x,y,z]}}}
for fold_idx in range(5):
    fold_dir = DISAGREEMENT_DIR / f"fold_{fold_idx}"
    fold_predictions[fold_idx] = aggregate_predictions_voxel(fold_dir, label_to_name)

all_case_ids = sorted(fold_predictions[0].keys())
print(f"Loaded predictions for {len(all_case_ids)} cases across 5 folds")

# 2. Compute per-case disagreement (mm), same spacing conversion as the
#    18-patient script / semi-supervised script's filter_by_agreement.
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
#    summary_mm.json included it. Sanity-check each fold's contents
#    against splits_final.json before trusting it (guards against the
#    append-not-overwrite bug found and fixed on 11 Aug recurring).
with open(SPLITS_PATH) as f:
    splits = json.load(f)

true_error_mm = {}
for fold_idx in range(5):
    val_path = VALIDATION_ROOT / f"fold_{fold_idx}" / "validation" / "summary_mm.json"
    with open(val_path) as f:
        val_data = json.load(f)

    summary_cases = set(val_data["detailed_results"].keys())
    split_cases = set(splits[fold_idx]["val"])
    extra = summary_cases - split_cases
    missing = split_cases - summary_cases
    if extra or missing:
        raise RuntimeError(
            f"fold_{fold_idx}: summary_mm.json does not match splits_final.json "
            f"(extra={extra}, missing={missing}) -- regenerate validation for "
            f"this fold before trusting these numbers."
        )

    for case_id, per_landmark in val_data["detailed_results"].items():
        true_error_mm[case_id] = float(np.mean(list(per_landmark.values())))

print(f"True error available for {len(true_error_mm)} cases (all folds verified clean)")

# 4. Correlate
common_cases = sorted(set(disagreement_mm) & set(true_error_mm))
missing_from_either = (set(disagreement_mm) | set(true_error_mm)) - set(common_cases)
if missing_from_either:
    print(f"WARNING: {len(missing_from_either)} case(s) present in only one source, excluded: {sorted(missing_from_either)}")

print(f"\n{'case':40s} {'disagreement_mm':>16s} {'true_error_mm':>16s}")
for c in common_cases:
    print(f"{c:40s} {disagreement_mm[c]:16.3f} {true_error_mm[c]:16.3f}")

disagreement_vals = [disagreement_mm[c] for c in common_cases]
error_vals = [true_error_mm[c] for c in common_cases]

rho, p = spearmanr(disagreement_vals, error_vals)
print(f"\nSpearman correlation (disagreement vs true error): rho={rho:.3f}, p={p:.4f}, n={len(common_cases)}")
if p < 0.05:
    print("Result: SIGNIFICANT -- ensemble disagreement correlates with actual error, "
          "extending the disagreement-based uncertainty finding to the 37-patient "
          "matched-training-data nnLandmark model.")
else:
    print("Result: not significant at alpha=0.05.")

print("\nCAVEAT: disagreement mixes 1 held-out fold with 4 in-sample folds per case "
      "(practical compromise, not a fully unbiased ensemble) -- report as an "
      f"exploratory/supporting result, n={len(common_cases)}.")

mask = [c != "CONTCT_R_15_FBA" for c in common_cases]
rho_excl, p_excl = spearmanr([d for d, m in zip(disagreement_vals, mask) if m],
                              [e for e, m in zip(error_vals, mask) if m])
print(f"\nExcluding R_15: rho={rho_excl:.3f}, p={p_excl:.4f}, n={sum(mask)}")
