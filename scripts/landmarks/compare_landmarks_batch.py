"""
compare_landmarks_batch.py

Runs compare_landmarks_v2.py's logic across all patients that have both:
  (a) a new _LB/_NN_LB/_LB2 landmark file in the annulus folder, and
  (b) an existing entry in dataset_manifest.csv

Summarises agreement (voxel distance, mm distance) per patient and overall,
to confirm whether the single-patient result generalises or was a fluke.

Run from project root (where src/ is importable), e.g.
/data/DERI-ecgai/__users/Vandhanaa/

Usage:
    python compare_landmarks_batch.py \
        --landmarks_dir "/data/DERI-ecgai/001_CTA_Segmention/data/Clipping/landmarks/annulus" \
        --data_processed_dir /data/DERI-ecgai/__users/Vandhanaa/data_processed \
        --manifest dataset_manifest.csv \
        --out_csv landmark_agreement_report.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocess import (
    resample_to_isotropic,
    world_to_voxel,
    prepare_centre_for_crop,
    crop_around_centre,
)


def extract_patient_id(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    return "_".join(parts[:3])  # e.g. CONTCT_R_08


def load_new_landmarks(json_path):
    with open(json_path) as f:
        data = json.load(f)
    coords = {}
    for cp in data["markups"][0]["controlPoints"]:
        label = cp["label"].strip().upper()
        if label in ("RC", "NC", "LC"):
            coords[label] = np.array(cp["position"], dtype=float)
    missing = {"RC", "NC", "LC"} - coords.keys()
    if missing:
        raise ValueError(f"Missing labels {missing}")
    return coords


def find_manifest_patient_id(short_id, manifest_ids):
    """short_id like 'CONTCT_R_08' -> matches 'CONTCT_R_08_FBA' etc. in manifest."""
    matches = [m for m in manifest_ids if m.startswith(short_id + "_")]
    return matches[0] if matches else None


def process_one(json_path, processed_image_path, manifest_row,
                 resample_spacing=1.0, crop_size=128, target_size=64):
    coords_mm = load_new_landmarks(json_path)
    centre_world = np.mean([coords_mm["RC"], coords_mm["NC"], coords_mm["LC"]], axis=0)

    sitk_img_native = sitk.ReadImage(processed_image_path)
    sitk_img_resampled = resample_to_isotropic(
        sitk_img_native, resample_spacing,
        interpolator=sitk.sitkLinear, default_value=-1000.0
    )

    centre_vox = world_to_voxel(centre_world, sitk_img_resampled, 'LPS')
    arr_shape = sitk.GetArrayFromImage(sitk_img_resampled).shape
    centre_vox = prepare_centre_for_crop(centre_vox, arr_shape)

    eff_crop_size = min(crop_size, arr_shape[0], arr_shape[1], arr_shape[2])
    _, crop_start = crop_around_centre(
        sitk.GetArrayFromImage(sitk_img_resampled), centre_vox, eff_crop_size
    )

    centre_crop = centre_vox - crop_start
    scale = target_size / eff_crop_size
    centre_resized = np.clip(centre_crop * scale, 0, target_size - 1)

    existing_centre = manifest_row[["centre_D", "centre_H", "centre_W"]].values[0]
    diff_vox = centre_resized - existing_centre
    dist_vox = float(np.linalg.norm(diff_vox))
    dist_mm = dist_vox * (resample_spacing * eff_crop_size / target_size)

    return centre_resized, existing_centre, dist_vox, dist_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_dir", required=True)
    parser.add_argument("--data_processed_dir", required=True)
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--out_csv", default="landmark_agreement_report.csv")
    args = parser.parse_args()

    df_manifest = pd.read_csv(args.manifest)
    manifest_ids = df_manifest["patient_id"].tolist()

    json_files = sorted(glob.glob(os.path.join(args.landmarks_dir, "*.json")))
    results = []

    for jf in json_files:
        short_id = extract_patient_id(jf)
        full_id = find_manifest_patient_id(short_id, manifest_ids)

        if full_id is None:
            print(f"SKIP {short_id}: not found in manifest")
            continue

        processed_image_path = os.path.join(
            args.data_processed_dir, full_id, "visualisation", "processed_data",
            f"{full_id}_processed_image.nii.gz"
        )
        if not os.path.exists(processed_image_path):
            print(f"SKIP {full_id}: no processed_image.nii.gz found")
            continue

        manifest_row = df_manifest[df_manifest["patient_id"] == full_id]

        try:
            centre_resized, existing_centre, dist_vox, dist_mm = process_one(
                jf, processed_image_path, manifest_row
            )
            print(f"{full_id}: dist = {dist_vox:.2f} vox / {dist_mm:.2f} mm")
            results.append({
                "patient_id": full_id,
                "source_file": os.path.basename(jf),
                "new_centre_D": centre_resized[0], "new_centre_H": centre_resized[1], "new_centre_W": centre_resized[2],
                "existing_centre_D": existing_centre[0], "existing_centre_H": existing_centre[1], "existing_centre_W": existing_centre[2],
                "dist_vox": dist_vox,
                "dist_mm": dist_mm,
            })
        except Exception as e:
            print(f"FAILED {full_id}: {e}")

    report = pd.DataFrame(results)
    report.to_csv(args.out_csv, index=False)

    print("\n=== SUMMARY ===")
    print(f"Patients compared: {len(report)}")
    if len(report) > 0:
        print(f"Mean distance: {report['dist_mm'].mean():.2f} mm  "
              f"(std {report['dist_mm'].std():.2f} mm)")
        print(f"Median distance: {report['dist_mm'].median():.2f} mm")
        print(f"Max distance: {report['dist_mm'].max():.2f} mm  "
              f"(patient: {report.loc[report['dist_mm'].idxmax(), 'patient_id']})")
        print(f"Min distance: {report['dist_mm'].min():.2f} mm")
    print(f"\nFull report written to {args.out_csv}")


if __name__ == "__main__":
    main()