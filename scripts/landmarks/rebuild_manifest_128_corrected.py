"""
rebuild_manifest_128_corrected.py

Rebuilds 128^3 image/seg arrays + manifest rows for the 17 confirmed-clean
CONTCT_R patients, using the new annulus landmark drop
(Clipping/landmarks/annulus) instead of whatever landmark source produced
the original (bugged, dead-centre) dataset_manifest_128.csv.

Excludes CONTCT_R_13 (non-standard point labels, pending supervisor
Slicer check) and CONTCT_R_18 (landmark z-coordinate appears sign-flipped,
pending supervisor confirmation).

Writes to a NEW output directory and NEW manifest CSV -- never overwrites
existing checkpoints, results, or manifests.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocess import (
    resample_to_isotropic,
    world_to_voxel,
    prepare_centre_for_crop,
    crop_around_centre,
    normalise_hu,
)

EXCLUDE = {'CONTCT_R_13'}


def load_rc_nc_lc(json_path, pid_short=None):
    with open(json_path) as f:
        data = json.load(f)
    coords = {}
    for cp in data['markups'][0]['controlPoints']:
        label = cp['label'].strip().upper()
        if label in ('RC', 'NC', 'LC'):
            coords[label] = np.array(cp['position'], dtype=float)
    missing = {'RC', 'NC', 'LC'} - coords.keys()
    if missing:
        raise ValueError(f"missing {missing}")

    # CONTCT_R_18: confirmed z-sign typo (Dr Sayed, July 2026) — flip z on
    # all three raw points before deriving centre/normal.
    if pid_short == 'CONTCT_R_18':
        for label in coords:
            coords[label] = coords[label] * np.array([1.0, 1.0, -1.0])

    centre = np.mean([coords['RC'], coords['NC'], coords['LC']], axis=0)
    normal = np.cross(coords['NC'] - coords['RC'], coords['LC'] - coords['RC'])
    normal = normal / np.linalg.norm(normal)
    return centre, normal


def find_patient_dir_and_files(raw_data_root, pid_short):
    matches = glob.glob(os.path.join(raw_data_root, f"{pid_short}_FBA*"))
    if not matches:
        return None, None, None
    patient_dir = matches[0]
    all_files = glob.glob(os.path.join(patient_dir, "*.nrrd"))
    img_file = None
    seg_file = None
    for f in all_files:
        fname = os.path.basename(f).lower()
        if 'seg' in fname:
            if seg_file is None:
                seg_file = f
        else:
            img_file = f
    return patient_dir, img_file, seg_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks_dir', required=True)
    ap.add_argument('--raw_data_root', required=True)
    ap.add_argument('--out_dir', default='./processed_output_128_corrected')
    ap.add_argument('--out_manifest', default='dataset_manifest_128_corrected.csv')
    ap.add_argument('--crop_size', type=int, required=True)
    ap.add_argument('--target_size', type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.landmarks_dir, '*.mrk.json')))

    rows = []
    errors = []
    skipped_excluded = []

    for jf in json_files:
        base = os.path.basename(jf)
        pid_short = '_'.join(base.split('_')[:3])

        if pid_short in EXCLUDE:
            skipped_excluded.append(pid_short)
            continue

        try:
            centre_world, normal_world = load_rc_nc_lc(jf, pid_short)
        except Exception as e:
            errors.append((pid_short, f"landmark parse failed: {e}"))
            continue

        patient_dir, img_file, seg_file = find_patient_dir_and_files(args.raw_data_root, pid_short)
        if img_file is None:
            errors.append((pid_short, f"no raw CT found under {args.raw_data_root}"))
            continue

        pid_full = os.path.basename(patient_dir)

        print(f"\n{'='*55}\nProcessing {pid_full}\n{'='*55}")

        sitk_img_native = sitk.ReadImage(img_file)
        sitk_img = resample_to_isotropic(
            sitk_img_native, 1.0, interpolator=sitk.sitkLinear, default_value=-1000.0
        )
        arr_raw = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        arr_norm = normalise_hu(arr_raw)

        centre_vox = world_to_voxel(centre_world, sitk_img, 'LPS')
        centre_vox = prepare_centre_for_crop(centre_vox, arr_norm.shape)

        eff_crop = min(args.crop_size, *arr_norm.shape)
        cropped, crop_start = crop_around_centre(arr_norm, centre_vox, eff_crop)
        centre_crop = centre_vox - crop_start

        scale = args.target_size / eff_crop
        if abs(scale - 1.0) > 1e-6:
            from scipy.ndimage import zoom
            cropped = zoom(cropped, scale, order=1)
        centre_resized = np.clip(centre_crop * scale, 0, args.target_size - 1)

        normal_dhw = normal_world[::-1]
        normal_dhw = normal_dhw / np.linalg.norm(normal_dhw)

        seg_out_path = ''
        out_dir_p = os.path.join(args.out_dir, pid_full)
        os.makedirs(out_dir_p, exist_ok=True)

        if seg_file is not None:
            sitk_seg_native = sitk.ReadImage(seg_file)
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(sitk_img)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            sitk_seg = resample.Execute(sitk_seg_native)
            seg_arr = sitk.GetArrayFromImage(sitk_seg).astype(np.float32)
            seg_cropped, _ = crop_around_centre(seg_arr, centre_vox, eff_crop)
            if abs(scale - 1.0) > 1e-6:
                from scipy.ndimage import zoom
                seg_cropped = zoom(seg_cropped, scale, order=0)
            seg_out_path = os.path.join(out_dir_p, f"{pid_full}_seg_resized_{args.target_size}.npy")
            np.save(seg_out_path, seg_cropped.astype(np.uint8))

        img_out_path = os.path.join(out_dir_p, f"{pid_full}_resized_{args.target_size}.npy")
        np.save(img_out_path, cropped.astype(np.float32))

        print(f"  centre_resized (D,H,W): {centre_resized.round(3)}")
        print(f"  saved: {img_out_path}")

        rows.append({
            'patient_id': pid_full,
            'image_path': img_out_path,
            'seg_path': seg_out_path,
            'centre_D': centre_resized[0], 'centre_H': centre_resized[1], 'centre_W': centre_resized[2],
            'normal_D': normal_dhw[0], 'normal_H': normal_dhw[1], 'normal_W': normal_dhw[2],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_manifest, index=False)

    print(f"\n{'='*55}")
    print(f"Wrote {len(df)} patients to {args.out_manifest}")
    if skipped_excluded:
        print(f"Excluded (pending supervisor check): {skipped_excluded}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for pid, err in errors:
            print(f"  {pid}: {err}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
