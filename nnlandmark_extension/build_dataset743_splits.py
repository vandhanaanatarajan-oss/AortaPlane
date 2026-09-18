#!/usr/bin/env python3
"""
Builds splits_final.json for Dataset743_AnnulusSemiSup_iter1, preserving
Dataset741's exact 5-fold validation structure (only real, ground-truth
CONTCT_R_* cases ever appear in a validation set) while adding all
pseudo-labelled cases to the TRAINING portion of every fold. This keeps
validation results comparable across iterations and meaningful (measured
against real ground truth, never against pseudo-labels).
"""
import json
from pathlib import Path

ORIGINAL_SPLITS = Path("nnLM_data/preprocessed/Dataset741_AnnulusCT/splits_final.json")
NEW_SPLITS = Path("nnLM_data/preprocessed/Dataset743_AnnulusSemiSup_iter1/splits_final.json")
NEW_DATASET_JSON = Path("nnLM_data/raw/Dataset743_AnnulusSemiSup_iter1/dataset.json")

with open(ORIGINAL_SPLITS) as f:
    original_folds = json.load(f)

# Pseudo-labelled case IDs = every case in the new dataset that ISN'T one
# of the 18 original CONTCT_R_* cases (derived from imagesTr, not hardcoded,
# so this stays correct even if the pseudo-label count changes).
images_tr = Path("nnLM_data/raw/Dataset743_AnnulusSemiSup_iter1/imagesTr")
all_case_ids = sorted(p.name[: -len("_0000.nii.gz")] for p in images_tr.glob("*_0000.nii.gz"))
original_case_ids = {case for fold in original_folds for case in fold["train"] + fold["val"]}
pseudo_case_ids = sorted(c for c in all_case_ids if c not in original_case_ids)

print(f"Found {len(pseudo_case_ids)} pseudo-labelled cases to add to training in every fold")
assert len(all_case_ids) == len(original_case_ids) + len(pseudo_case_ids), (
    "Case count mismatch -- some case in imagesTr doesn't match either the "
    "original cases or the derived pseudo-label list. Stop and check manually."
)

new_folds = []
for fold in original_folds:
    new_folds.append({
        "train": fold["train"] + pseudo_case_ids,  # pseudo-labels only ever in TRAIN
        "val": fold["val"],                         # val stays exactly as Dataset741's, real cases only
    })

NEW_SPLITS.parent.mkdir(parents=True, exist_ok=True)
with open(NEW_SPLITS, "w") as f:
    json.dump(new_folds, f, indent=2)

for i, fold in enumerate(new_folds):
    print(f"Fold {i}: {len(fold['train'])} train ({len(fold['train']) - len(pseudo_case_ids)} real + "
          f"{len(pseudo_case_ids)} pseudo), {len(fold['val'])} val (all real)")

print(f"\nWritten to {NEW_SPLITS}")