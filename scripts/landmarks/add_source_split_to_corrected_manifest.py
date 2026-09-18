"""
add_source_split_to_corrected_manifest.py

Merges 'source' and 'split' columns from the original dataset_manifest.csv
into dataset_manifest_128_corrected.csv, matching build_manifest_128.py's
existing merge pattern. Without this, the corrected 128^3 manifest is
missing the columns train.py / fold_config.json likely need for CV fold
assignment.
"""

import pandas as pd

corrected = pd.read_csv("dataset_manifest_128_corrected.csv")
orig = pd.read_csv("dataset_manifest.csv")[["patient_id", "source", "split"]]

merged = corrected.merge(orig, on="patient_id", how="left")

missing = merged[merged["split"].isna()]
if len(missing) > 0:
    print("WARNING — these patients have no matching row in the original manifest:")
    print(missing["patient_id"].tolist())
    print("(Expected if these are newly-added CONTCT_R patients not in the original")
    print(" 42-patient cohort -- you'll need to decide their source/split manually.)")

merged.to_csv("dataset_manifest_128_corrected.csv", index=False)
print(f"\nWrote dataset_manifest_128_corrected.csv — {len(merged)} patients, now with source/split")
