#!/usr/bin/env python3
import glob
import json
import os

manifest = []

# SEGA: PatientID.nrrd (raw), excluding PatientID.seg.nrrd (labels)
sega_files = glob.glob("/data/DERI-ecgai/001_CTA_Segmention/data/SEGA/SEGA/*/*.nrrd")
sega_files = [f for f in sega_files if not f.endswith(".seg.nrrd")]
for f in sega_files:
    manifest.append({"source": "SEGA", "path": f})

# CIS_UNet_Data: Volumes/*.nii.gz
cis_files = glob.glob("/data/DERI-ecgai/001_CTA_Segmention/data/CIS_UNet_Data/Volumes/*.nii.gz")
for f in cis_files:
    manifest.append({"source": "CIS_UNet_Data", "path": f})

# Barts_CT_01: all .nii.gz under both subfolders
barts_files = glob.glob("/data/DERI-ecgai/001_CTA_Segmention/data/Barts_CT_01/**/*.nii.gz", recursive=True)
for f in barts_files:
    manifest.append({"source": "Barts_CT_01", "path": f})

print(f"SEGA: {len(sega_files)} volumes")
print(f"CIS_UNet_Data: {len(cis_files)} volumes")
print(f"Barts_CT_01: {len(barts_files)} volumes")
print(f"TOTAL: {len(manifest)} volumes")

out_path = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWritten to {out_path}")
