"""
compare_landmarks_v2.py

Correct version: converts the new landmark centre into the SAME 64^3
resized voxel space as dataset_manifest.csv, by importing and reusing
your actual src/preprocess.py functions (resample_to_isotropic,
world_to_voxel, prepare_centre_for_crop, crop_around_centre) rather than
reimplementing the logic separately. This guarantees we're not
introducing a second, subtly-different coordinate conversion.

Run this from your project root (where src/ is a subfolder), e.g.:
    /data/DERI-ecgai/__users/Vandhanaa/

Usage:
    python compare_landmarks_v2.py \
        --patient_id CONTCT_R_08_FBA \
        --new_landmark_json /data/.../CONTCT_R_08_Pre_CT_20191129_FBA_landmarks_LB.mrk.json \
        --processed_image /data/.../CONTCT_R_08_FBA_processed_image.nii.gz \
        --manifest dataset_manifest.csv \
        --resample_spacing 1.0 \
        --crop_size 128 \
        --target_size 64
"""

import argparse
import json
import sys
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Make sure src/ is importable regardless of where this script is invoked from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocess import (
    resample_to_isotropic,
    world_to_voxel,
    prepare_centre_for_crop,
    crop_around_centre,
)


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
        raise ValueError(f"Missing labels {missing} in {json_path}")
    return coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--new_landmark_json", required=True)
    parser.add_argument("--processed_image", required=True)
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--resample_spacing", type=float, default=1.0)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=64)
    args = parser.parse_args()

    # --- 1. New landmarks: world mm (LPS) -> centre in WORLD space ---
    coords_mm = load_new_landmarks(args.new_landmark_json)
    centre_world = np.mean([coords_mm["RC"], coords_mm["NC"], coords_mm["LC"]], axis=0)
    print(f"New centre (world LPS mm): {centre_world.round(2)}")

    # --- 2. Load native image, RESAMPLE to isotropic (same as pipeline) ---
    sitk_img_native = sitk.ReadImage(args.processed_image)
    print(f"Native image size/spacing: {sitk_img_native.GetSize()} / "
          f"{sitk_img_native.GetSpacing()}")

    sitk_img_resampled = resample_to_isotropic(
        sitk_img_native, args.resample_spacing,
        interpolator=sitk.sitkLinear, default_value=-1000.0
    )
    print(f"Resampled image size/spacing: {sitk_img_resampled.GetSize()} / "
          f"{sitk_img_resampled.GetSpacing()}")

    # --- 3. World mm -> voxel index on the RESAMPLED image, via the same
    #        world_to_voxel() function the real pipeline uses ---
    #        coordinate_system='LPS' matches this file's declared header
    centre_vox = world_to_voxel(centre_world, sitk_img_resampled, 'LPS')
    print(f"New centre (resampled voxel space, D,H,W): {centre_vox.round(2)}")

    # --- 4. Crop + resize, exactly mirroring preprocess_patient() ---
    arr_shape = sitk.GetArrayFromImage(sitk_img_resampled).shape  # (D,H,W)
    centre_vox = prepare_centre_for_crop(centre_vox, arr_shape)

    eff_crop_size = min(args.crop_size, arr_shape[0], arr_shape[1], arr_shape[2])
    _, crop_start = crop_around_centre(
        sitk.GetArrayFromImage(sitk_img_resampled), centre_vox, eff_crop_size
    )

    centre_crop = centre_vox - crop_start
    scale = args.target_size / eff_crop_size
    centre_resized = np.clip(centre_crop * scale, 0, args.target_size - 1)

    print(f"\nNew centre (64^3 resized voxel space, D,H,W): {centre_resized.round(3)}")

    # --- 5. Compare against existing manifest value ---
    df = pd.read_csv(args.manifest)
    row = df[df["patient_id"] == args.patient_id]
    if row.empty:
        print(f"\nWARNING: {args.patient_id} not in {args.manifest}")
        return

    existing_centre = row[["centre_D", "centre_H", "centre_W"]].values[0]
    print(f"Existing manifest centre (64^3 voxel space):  {existing_centre.round(3)}")

    diff_vox = centre_resized - existing_centre
    diff_mm = diff_vox * (args.resample_spacing * eff_crop_size / args.target_size)
    dist_vox = np.linalg.norm(diff_vox)
    dist_mm = np.linalg.norm(diff_mm)

    print(f"\nDifference (voxels): {diff_vox.round(3)}  |  distance: {dist_vox:.2f} vox")
    print(f"Difference (mm):     {diff_mm.round(2)}  |  distance: {dist_mm:.2f} mm")
    print(f"\n(For reference: this pipeline's voxel-to-mm factor at 64^3 with "
          f"128mm crop is {128/64:.1f} mm/voxel.)")


if __name__ == "__main__":
    main()