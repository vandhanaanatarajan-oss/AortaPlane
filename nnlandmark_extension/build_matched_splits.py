#!/usr/bin/env python3
"""
Builds Dataset741's splits_final.json by reusing fold_config.json's
EXACT own 5-fold train/val patient assignments (not an independently
regenerated split with the same patient count) -- so nnLandmark's
cross-validation genuinely matches M1/M2's fold-by-fold, patient-for-
patient.

Run from /data/DERI-ecgai/__users/Vandhanaa/ (where fold_config.json
lives).
"""
import json

FOLD_CONFIG_PATH = "/data/DERI-ecgai/__users/Vandhanaa/fold_config.json"
OUT_PATH = ("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed/"
            "Dataset741_AnnulusCT/splits_final.json")

with open(FOLD_CONFIG_PATH) as f:
    fc = json.load(f)

splits = []
for fold in fc["folds"]:
    splits.append({
        "train": fold["train"],
        "val": fold["val"],
    })

with open(OUT_PATH, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Wrote {OUT_PATH}")
for i, s in enumerate(splits):
    print(f"Fold {i}: train={len(s['train'])}, val={len(s['val'])}")