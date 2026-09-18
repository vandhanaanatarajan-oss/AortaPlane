import pandas as pd

rows_128 = pd.read_csv("data_processed_128/manifest_rows_128.csv")
orig     = pd.read_csv("dataset_manifest.csv")[["patient_id", "source", "split"]]

merged = rows_128.merge(orig, on="patient_id", how="left")

missing = merged[merged["split"].isna()]
if len(missing) > 0:
    print("WARNING — these patients have no matching row in the original manifest:")
    print(missing["patient_id"].tolist())

merged = merged[["patient_id", "image_path", "seg_path",
                  "centre_D", "centre_H", "centre_W",
                  "normal_D", "normal_H", "normal_W",
                  "source", "split"]]

merged.to_csv("dataset_manifest_128.csv", index=False)

print(f"Wrote dataset_manifest_128.csv — {len(merged)} patients")
print(merged["split"].value_counts())
