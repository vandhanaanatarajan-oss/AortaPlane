"""
preprocess_and_visualise.py
============================
Run this locally RIGHT NOW — no HPC needed.

What this does:
  1. Loads your raw NIfTI CT scan + segmentation + landmark JSON
  2. Applies the same preprocessing as AutoDataProcessor
     (resampling, normalisation, resizing to 64x64x64)
  3. Saves the processed files to a local output folder
  4. Generates rich visualisation figures you can show your supervisor

Usage:
  python src/preprocess.py \
      --raw_data_path /path/to/raw/data \
      --out_path ./processed_output \
      --patient_ids CONTCT_R_08_FBA CONTCT_R_16_FBA D1 \
      --target_size 64

Outputs per patient (saved to ./processed_output/<patient_id>/):
  - <id>_resized_64.npy              (normalised, resampled, cropped, resized CT)
  - <id>_seg_resized_64.npy          (matching resized segmentation, if available)
  - <id>_fig1_preprocessing_steps.png
  - <id>_fig2_slices_with_plane.png
  - <id>_fig3_hu_histogram.png
  - <id>_fig4_3d_aorta.png
  - <id>_fig7_roi_overlay_quality_check.png
  - fig5_dataset_overview.png
  - fig6_label_statistics.png

============================================================
FIX LOG (this version)
============================================================
BUG: ROI crop / 3D plot / plane overlay were landing on the wrong anatomy
(e.g. shoulder/ribs instead of the aortic valve) for patients whose data
came from the *_processed_landmarks.json path (the `using_processed`
branch in preprocess_patient()).

ROOT CAUSE: `geometric_properties.center_point` and `.normal_vector` in
*_processed_landmarks.json are stored in (X, Y, Z) order — i.e.
(column, row, slice) — the SAME order as the raw controlPoint_*_image
voxel coordinates they were derived from. Every other part of this
pipeline (numpy arrays from SimpleITK, prepare_centre_for_crop,
crop_around_centre, draw_plane_line, the 3D plotting code) works in
(D, H, W) = (Z, Y, X) array order. The old code read center_point /
normal_vector straight out of the JSON and used them AS-IS in
(D, H, W) position, silently swapping the depth (slice) axis with the
width (column) axis. That put the crop centre on a completely
different slice than intended, which is exactly the "wrong anatomy"
symptom (e.g. CONTCT_R_08_FBA centre [268.8, 235.3, 641.5] in X,Y,Z
was being read as D=268.8, H=235.3, W=641.5 instead of the correct
D=641.5, H=235.3, W=268.8).

FIX: reverse the axis order ([::-1]) the moment center_point /
normal_vector are read out of the processed-landmarks JSON, so they
match (D, H, W) like everything downstream expects. Added an explicit
sanity print + an automatic "is centre roughly central / inside the
volume" check so future axis-order regressions surface immediately
instead of silently producing a wrong crop.
============================================================
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════
# PLANE LINE HELPER
# ══════════════════════════════════════════════

def draw_plane_line(ax, normal, centre_2d, slice_size, view='axial'):
    """
    Draw the correct plane intersection line on a 2D orthogonal slice.

    The intersection of a plane with a slice is:
        line_direction = cross(plane_normal, slice_view_axis)
    """
    n = np.array(normal, dtype=float)
    norm_len = np.linalg.norm(n)
    if norm_len < 1e-8:
        return
    n = n / norm_len

    view_axes = {
        'axial':    np.array([1, 0, 0], dtype=float),
        'coronal':  np.array([0, 1, 0], dtype=float),
        'sagittal': np.array([0, 0, 1], dtype=float),
    }
    row_col_axes = {
        'axial':    (1, 2),
        'coronal':  (0, 2),
        'sagittal': (0, 1),
    }

    view_dir    = view_axes[view]
    line_dir_3d = np.cross(n, view_dir)
    ld_len      = np.linalg.norm(line_dir_3d)

    if ld_len < 1e-8:
        _, c = centre_2d
        ax.axvline(c, color='yellow', linestyle='--',
                   linewidth=1.8, alpha=0.85, label='Plane')
        return

    line_dir_3d = line_dir_3d / ld_len
    row_ax, col_ax = row_col_axes[view]
    d_col = line_dir_3d[col_ax]
    d_row = line_dir_3d[row_ax]

    r0, c0 = centre_2d
    t_vals = np.linspace(-slice_size * 2, slice_size * 2, 800)
    cols = c0 + t_vals * d_col
    rows = r0 + t_vals * d_row

    mask = (cols >= 0) & (cols < slice_size) & (rows >= 0) & (rows < slice_size)
    if mask.sum() < 2:
        return

    ax.plot(cols[mask], rows[mask], 'y--',
            linewidth=1.8, alpha=0.85, label='Plane')


# ══════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════

def find_patient_files(raw_data_path: str, patient_id: str) -> dict:
    search_dirs = [
        os.path.join(raw_data_path, patient_id),
        os.path.join(raw_data_path, 'BRC_retrospectives', patient_id),
        os.path.join(raw_data_path, 'AC_cases_AS_OF_17042025', 'Completed__confirmed', patient_id),
        os.path.join(raw_data_path, 'AC_cases_AS_OF_17042025', 'Completed__uncertain__to_be_checked', patient_id),
        raw_data_path,
    ]

    files = {'img': None, 'seg': None, 'landmark': None, 'patient_dir': None}

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        all_files = glob.glob(os.path.join(search_dir, '*'))
        img_candidates = []
        seg_candidates = []
        lm_candidates  = []

        for f in all_files:
            fname = Path(f).name.lower()
            if '.mrk.json' in fname or ('landmark' in fname and fname.endswith('.json')):
                lm_candidates.append(f)
            elif '.nrrd' in fname or '.nii.gz' in fname or '.nii' in fname:
                if 'seg' in fname:
                    seg_candidates.append(f)
                else:
                    img_candidates.append(f)

        if img_candidates:
            files['img']         = img_candidates[0]
            files['seg']         = seg_candidates[0] if seg_candidates else None
            files['landmark']    = lm_candidates[0]  if lm_candidates  else None
            files['patient_dir'] = search_dir
            break

    processed_base = os.path.join(raw_data_path, patient_id,
                                   'visualisation', 'processed_data')
    if os.path.exists(processed_base):
        proc_img = os.path.join(processed_base, f'{patient_id}_processed_image.nii.gz')
        proc_seg = os.path.join(processed_base, f'{patient_id}_processed_seg.nii.gz')
        proc_lm  = os.path.join(processed_base, f'{patient_id}_processed_landmarks.json')
        if os.path.exists(proc_img) and os.path.exists(proc_lm):
            files['processed_img'] = proc_img
            files['processed_lm']  = proc_lm
            if os.path.exists(proc_seg):
                files['processed_seg'] = proc_seg

    return files


def load_sitk_image(path: str):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = np.array(img.GetSpacing())
    origin  = np.array(img.GetOrigin())
    return img, arr, spacing, origin


def load_landmark_mrk_json(path: str):
    """
    Returns (points_dict, coordinate_system, format).

    Supports TWO landmark file formats, auto-detected:

    1. Slicer .mrk.json "markups" format — points are RAS or LPS world
       mm, declared via a "coordinateSystem" key.

    2. This project's custom format — points are VOXEL INDICES in the
       *original, pre-resampling* image grid, declared via
       "coordinate_system": "image_voxel_space" and keys like
       "controlPoint_1_image". These are stored in (X, Y, Z) order
       (column, row, slice) — NOT world mm and must NOT be run through
       the RAS/LPS flip — they need to be converted via the original
       image's own TransformContinuousIndexToPhysicalPoint.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # ── Format 2: this project's custom voxel-space schema ──
    if data.get('coordinate_system') == 'image_voxel_space' or 'landmarks' in data:
        lm = data.get('landmarks', {})
        points = {}
        for i, key in enumerate(sorted(k for k in lm if k.startswith('controlPoint_')), 1):
            points[f'controlPoint_{i}'] = np.array(lm[key], dtype=np.float64)
        return points, None, 'image_voxel_space'

    # ── Format 1: Slicer markups (RAS/LPS world mm) ──
    points = {}
    coord_system = None
    if 'markups' in data:
        for markup in data['markups']:
            cs = markup.get('coordinateSystem')
            if cs is not None:
                coord_system = cs
            if 'controlPoints' in markup:
                for i, cp in enumerate(markup['controlPoints'][:3], 1):
                    pos = cp.get('position', cp.get('Position', [0, 0, 0]))
                    points[f'controlPoint_{i}'] = np.array(pos, dtype=np.float64)

    if coord_system is None:
        print("  ⚠ WARNING: no 'coordinateSystem' field found in .mrk.json — "
              "assuming RAS. If the crop lands on the wrong anatomy, check "
              "this file manually and pass the correct system.")
        coord_system = 'RAS'

    return points, coord_system, 'markups'


def load_processed_landmarks(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════
# SECTION 2 — PREPROCESSING
# ══════════════════════════════════════════════

def is_already_normalised(arr: np.ndarray) -> bool:
    """
    Detect whether an array is raw HU (large dynamic range, negative values
    typical of air/lung) or already normalised to roughly [0, 1].

    Raw CT HU data always has min well below -200 (air ≈ -1000 HU) and a
    range far larger than 1. Pre-normalised data sits inside ~[0, 1].
    """
    amin, amax = float(arr.min()), float(arr.max())
    return amin >= -1e-3 and amax <= 1.0 + 1e-3


def normalise_hu(arr: np.ndarray,
                  clip_min: float = -200.0,
                  clip_max: float = 1000.0) -> np.ndarray:
    arr = np.clip(arr, clip_min, clip_max)
    arr = (arr - clip_min) / (clip_max - clip_min)
    return arr.astype(np.float32)


def resample_to_isotropic(sitk_img, target_spacing: float = 1.0,
                           interpolator=sitk.sitkLinear,
                           default_value: float = -1000.0):
    original_spacing = np.array(sitk_img.GetSpacing())
    original_size    = np.array(sitk_img.GetSize())

    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing))
                for i in range(3)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([target_spacing] * 3)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(default_value)
    resample.SetInterpolator(interpolator)

    return resample.Execute(sitk_img)


def resize_volume(arr: np.ndarray, target_size: int = 64, order: int = 1) -> np.ndarray:
    return resize(arr, (target_size, target_size, target_size),
                  order=order, preserve_range=True,
                  anti_aliasing=(order > 0)).astype(np.float32)


def compute_plane_from_3_points(p1, p2, p3):
    """p1=RC, p2=NC, p3=LC  ->  v1 = NC-RC, v2 = LC-RC, normal = normalise(cross(v1,v2))"""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-8:
        normal = normal / norm_len
    centre = (p1 + p2 + p3) / 3.0
    return normal, centre


def original_voxel_to_resampled_voxel(voxel_xyz, sitk_img_original, sitk_img_resampled):
    """
    Convert a voxel index (X, Y, Z order) in the ORIGINAL (pre-resampling)
    image grid to a PHYSICAL point (mm), via the image's own geometry.
    No RAS/LPS guessing needed: this stays in the image's native (LPS)
    world frame throughout.
    """
    physical_point = sitk_img_original.TransformContinuousIndexToPhysicalPoint(
        [float(voxel_xyz[0]), float(voxel_xyz[1]), float(voxel_xyz[2])]
    )
    return np.array(physical_point, dtype=np.float64)  # physical mm


def world_to_voxel(centre_world, sitk_img, coordinate_system='RAS'):
    """
    Convert world coordinates (mm) from .mrk.json to voxel indices in
    (D, H, W) = (Z, Y, X) order.

    SimpleITK's physical space is always LPS. Only flip X/Y if the source
    points are actually RAS — flipping LPS points again would shift the
    centre by tens of millimetres and crop the wrong anatomy entirely.
    """
    if coordinate_system.upper() == 'RAS':
        centre_lps = np.array([-centre_world[0],
                                -centre_world[1],
                                 centre_world[2]])
    else:  # already LPS
        centre_lps = np.array(centre_world, dtype=np.float64)

    vox_xyz = sitk_img.TransformPhysicalPointToContinuousIndex(centre_lps.tolist())
    centre_vox = np.array([vox_xyz[2], vox_xyz[1], vox_xyz[0]], dtype=np.float32)
    return centre_vox


def prepare_centre_for_crop(centre, image_shape):
    """
    `centre` MUST already be in (D, H, W) voxel order by the time it
    reaches this function (callers are responsible for axis-reordering
    out of whatever order the source file used — see the axis-order fix
    in preprocess_patient() for the *_processed_landmarks.json path).
    Handles the case where centre is stored normalised to [0,1] instead
    of raw voxel coordinates.
    """
    centre = np.asarray(centre, dtype=np.float32)
    D, H, W = image_shape

    if centre.max() <= 1.0:
        centre_vox = np.array([centre[0]*D, centre[1]*H, centre[2]*W], dtype=np.float32)
    else:
        centre_vox = centre.astype(np.float32)

    centre_vox[0] = np.clip(centre_vox[0], 0, D - 1)
    centre_vox[1] = np.clip(centre_vox[1], 0, H - 1)
    centre_vox[2] = np.clip(centre_vox[2], 0, W - 1)

    return centre_vox


def crop_around_centre(volume, centre_vox, crop_size=128):
    D, H, W = volume.shape
    cd, ch, cw = np.asarray(centre_vox, dtype=np.float32).astype(int)

    cd = int(np.clip(cd, 0, D - 1))
    ch = int(np.clip(ch, 0, H - 1))
    cw = int(np.clip(cw, 0, W - 1))

    crop_size = int(min(crop_size, D, H, W))
    half = crop_size // 2

    d0 = max(cd - half, 0)
    h0 = max(ch - half, 0)
    w0 = max(cw - half, 0)

    d1 = min(d0 + crop_size, D)
    h1 = min(h0 + crop_size, H)
    w1 = min(w0 + crop_size, W)

    d0 = max(d1 - crop_size, 0)
    h0 = max(h1 - crop_size, 0)
    w0 = max(w1 - crop_size, 0)

    roi = volume[d0:d1, h0:h1, w0:w1]

    if roi.size == 0 or 0 in roi.shape:
        raise ValueError(
            f"Empty ROI crop: volume={volume.shape}, "
            f"centre_vox={(cd,ch,cw)}, "
            f"start={(d0,h0,w0)}, end={(d1,h1,w1)}"
        )

    return roi, np.array([d0, h0, w0], dtype=np.float32)


def preprocess_patient(raw_data_path, patient_id, out_path,
                        target_size=64, resample_spacing=1.0,
                        crop_size=128):
    print(f"\n{'='*55}")
    print(f"  Processing: {patient_id}")
    print(f"{'='*55}")

    out_dir = os.path.join(out_path, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    files  = find_patient_files(raw_data_path, patient_id)
    result = {'patient_id': patient_id, 'success': False}
    using_processed = False
    seg_arr_full = None  # full-resolution segmentation, aligned to arr_raw

    if 'processed_img' in files and os.path.exists(files['processed_img']):
        using_processed = True
        print(f"  Loading already-processed data...")
        sitk_img, arr_raw, spacing, origin = load_sitk_image(files['processed_img'])
        lm_data = load_processed_landmarks(files['processed_lm'])

        geo = lm_data['geometric_properties']

        # ── AXIS-ORDER FIX ──────────────────────────────────────────
        # `normal_vector` and `center_point` in *_processed_landmarks.json
        # are stored in (X, Y, Z) order (column, row, slice) — the SAME
        # order as the raw controlPoint_*_image voxel coordinates they
        # were derived from (verify: the 3rd component is the largest /
        # matches the slice-count scale, e.g. ~640 for a tall chest CT,
        # confirming it's the slice index Z, not a row/col index).
        #
        # Every downstream consumer (prepare_centre_for_crop,
        # crop_around_centre, draw_plane_line, the 3D scatter plot) works
        # in (D, H, W) = (Z, Y, X) numpy array order. Using the JSON
        # values as-is silently swaps the depth (slice) axis with the
        # width (column) axis, which sends the crop to a totally
        # different slice than the aortic valve — this was the root
        # cause of crops landing on the shoulder/ribs instead of the
        # heart. Reversing the order here fixes it.
        normal_xyz = np.array(geo['normal_vector'])
        centre_xyz = np.array(geo['center_point'])

        normal = normal_xyz[::-1]   # (X,Y,Z) -> (Z,Y,X) == (D,H,W)
        centre = centre_xyz[::-1]   # (X,Y,Z) -> (Z,Y,X) == (D,H,W)
        # ────────────────────────────────────────────────────────────

        if 'processed_seg' in files and os.path.exists(files['processed_seg']):
            _, seg_arr_full, _, _ = load_sitk_image(files['processed_seg'])

        print(f"  Image shape      : {arr_raw.shape}")
        print(f"  Image value range: [{arr_raw.min():.3f}, {arr_raw.max():.3f}]")
        print(f"  Normal (D,H,W)   : {normal.round(4)}  (raw X,Y,Z was {normal_xyz.round(4)})")
        print(f"  Centre (D,H,W)   : {centre.round(2)}  (raw X,Y,Z was {centre_xyz.round(2)})")

    elif files['img'] is not None:
        print(f"  Loading raw image from: {files['img']}")
        sitk_img_raw, arr_raw_orig, spacing_orig, origin = load_sitk_image(files['img'])

        print(f"  Raw image shape    : {arr_raw_orig.shape}")
        print(f"  Raw spacing (mm)   : {spacing_orig.round(3)}")
        print(f"  HU range           : [{arr_raw_orig.min():.0f}, {arr_raw_orig.max():.0f}]")

        print(f"  Resampling image to {resample_spacing}mm isotropic...")
        sitk_img = resample_to_isotropic(sitk_img_raw, resample_spacing,
                                          interpolator=sitk.sitkLinear,
                                          default_value=-1000.0)
        arr_raw  = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        spacing  = np.array(sitk_img.GetSpacing())
        print(f"  Resampled shape    : {arr_raw.shape}")

        if files['seg'] is not None:
            print(f"  Loading + resampling segmentation (nearest-neighbour)...")
            sitk_seg_raw, _, _, _ = load_sitk_image(files['seg'])
            # Resample seg onto the SAME grid as the resampled image so the
            # two stay pixel-aligned. Nearest-neighbour preserves label values.
            sitk_seg_raw = sitk.Cast(sitk_seg_raw, sitk.sitkUInt8)
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(sitk_img)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resample.SetDefaultPixelValue(0)
            sitk_seg = resample.Execute(sitk_seg_raw)
            seg_arr_full = sitk.GetArrayFromImage(sitk_seg).astype(np.float32)
            print(f"  Segmentation shape : {seg_arr_full.shape} (aligned to image)")

        normal, centre = None, None
        coord_system = 'RAS'
        lm_format = None
        if files['landmark'] is not None:
            print(f"  Loading landmarks from: {files['landmark']}")
            lm_pts, coord_system, lm_format = load_landmark_mrk_json(files['landmark'])
            print(f"  Landmark format: {lm_format}"
                  + (f" ({coord_system})" if coord_system else ""))
            if len(lm_pts) >= 3:
                pts = list(lm_pts.values())

                if lm_format == 'image_voxel_space':
                    # Points are voxel indices (X, Y, Z order) in the
                    # ORIGINAL pre-resampling image grid — convert each to
                    # a physical world point via the ORIGINAL image's own
                    # geometry first, then compute the plane in that
                    # physical (mm) space. No RAS/LPS flip applies here.
                    phys_pts = [
                        original_voxel_to_resampled_voxel(p, sitk_img_raw, None)
                        for p in pts
                    ]
                    normal, centre = compute_plane_from_3_points(*phys_pts)
                    print(f"  (points converted: original voxel index → physical mm "
                          f"via the original image's own spacing/origin/direction)")
                else:
                    normal, centre = compute_plane_from_3_points(pts[0], pts[1], pts[2])

                print(f"  Normal vector    : {normal.round(4)}")
                print(f"  Centre (world mm): {centre.round(2)}")
    else:
        print(f"  ✗ No data found for {patient_id}")
        return result

    if normal is None or centre is None:
        print(f"  ✗ No valid landmarks for {patient_id} — skipping")
        return result

    # ── only run HU-clip normalisation on RAW HU data.
    # Already-processed data loaded from *_processed_image.nii.gz is typically
    # already normalised to [0,1]; re-running normalise_hu on it collapses
    # the whole volume into a near-constant value, which produces a flat,
    # low-contrast image. ──
    if using_processed:

        if is_already_normalised(arr_raw):

            sat_pct = float((arr_raw >= 0.999).mean())

            if sat_pct < 0.20:
                print("  Detected already-normalised data")
                arr_norm = arr_raw.astype(np.float32)

            elif files.get('img') is not None:

                print(
                    f"  ⚠ Processed image {sat_pct*100:.1f}% saturated "
                    "→ reloading raw image"
                )

                sitk_img_raw_fb, arr_raw_fb, _, _ = load_sitk_image(files['img'])

                sitk_img = resample_to_isotropic(
                    sitk_img_raw_fb,
                    resample_spacing
                )

                arr_raw = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

                tissue = arr_raw[arr_raw > -1500]

                p1 = np.percentile(tissue, 1)
                p99 = np.percentile(tissue, 99)

                arr_clipped = np.clip(arr_raw, p1, p99)

                arr_norm = (
                    (arr_clipped - p1) /
                    (p99 - p1 + 1e-8)
                ).astype(np.float32)

            else:
                arr_norm = arr_raw.astype(np.float32)

        else:

            print("  Processed image contains HU values")

            tissue = arr_raw[arr_raw > -1500]

            p1 = np.percentile(tissue, 1)
            p99 = np.percentile(tissue, 99)

            arr_clipped = np.clip(arr_raw, p1, p99)

            arr_norm = (
                (arr_clipped - p1) /
                (p99 - p1 + 1e-8)
            ).astype(np.float32)

    else:

        arr_norm = normalise_hu(arr_raw)

    # ── coordinate conversion ──
    if using_processed:
        # `centre` was already reordered to (D, H, W) above — safe to pass
        # straight through.
        centre_vox = prepare_centre_for_crop(centre, arr_norm.shape)
    elif lm_format == 'image_voxel_space':
        # `centre` here is already a physical (mm) point in the original
        # image's world frame (computed above). Map it straight into the
        # RESAMPLED image's voxel grid — same world frame, just a
        # different voxel spacing, so no RAS/LPS flip is involved.
        vox_xyz = sitk_img.TransformPhysicalPointToContinuousIndex(centre.tolist())
        centre_vox = np.array([vox_xyz[2], vox_xyz[1], vox_xyz[0]], dtype=np.float32)
    else:
        centre_vox = world_to_voxel(centre, sitk_img, coord_system)

        # Safety net: a centre landing right at the volume edge almost
        # always means the RAS/LPS convention was wrong for this file.
        # Try the opposite convention and keep whichever is more central.
        D0, H0, W0 = arr_norm.shape
        frac0 = np.array([centre_vox[0]/D0, centre_vox[1]/H0, centre_vox[2]/W0])
        edge0 = np.minimum(frac0, 1 - frac0).min()
        if edge0 < 0.05:
            alt_system = 'LPS' if coord_system.upper() == 'RAS' else 'RAS'
            centre_vox_alt = world_to_voxel(centre, sitk_img, alt_system)
            frac1 = np.array([centre_vox_alt[0]/D0, centre_vox_alt[1]/H0, centre_vox_alt[2]/W0])
            edge1 = np.minimum(frac1, 1 - frac1).min()
            if edge1 > edge0:
                print(f"  ⚠ Centre was at the volume edge using {coord_system} "
                      f"convention — auto-switching to {alt_system}, which lands "
                      f"more centrally. PLEASE VERIFY against your actual "
                      f".mrk.json 'coordinateSystem' field.")
                centre_vox = centre_vox_alt
                coord_system = alt_system

    D, H, W = arr_norm.shape
    centre_vox[0] = np.clip(centre_vox[0], 0, D - 1)
    centre_vox[1] = np.clip(centre_vox[1], 0, H - 1)
    centre_vox[2] = np.clip(centre_vox[2], 0, W - 1)

    # ── Quality-check guard: catch axis-order / coordinate-system bugs
    # automatically instead of only discovering them by eyeballing PNGs.
    # The aortic valve should never land in the outer 10% of any axis
    # for any of these datasets — if it does, something upstream
    # (coordinate system, axis order) is almost certainly wrong. ──
    edge_warnings = []
    for dim, val, size, name in [(0, centre_vox[0], D, 'D (depth/slice)'),
                                   (1, centre_vox[1], H, 'H (row)'),
                                   (2, centre_vox[2], W, 'W (column)')]:
        frac = val / size
        if frac < 0.1 or frac > 0.9:
            msg = (f"  ⚠ WARNING: centre {name}={val:.1f} near edge "
                   f"(size={size}, frac={frac:.2f}) — crop may be off. "
                   f"Check coordinate system / axis order for this patient.")
            print(msg)
            edge_warnings.append(msg)

    eff_crop_size = min(crop_size, arr_norm.shape[0], arr_norm.shape[1], arr_norm.shape[2])
    roi, crop_start = crop_around_centre(arr_norm, centre_vox, eff_crop_size)
    arr_resized = resize_volume(roi, target_size, order=1)

    # ── crop + resize the segmentation the same way as the image
    # (nearest-neighbour resize to preserve label IDs) and carry it
    # through, instead of discarding it. ──
    seg_resized = None
    if seg_arr_full is not None and seg_arr_full.shape == arr_norm.shape:
        seg_roi, _ = crop_around_centre(seg_arr_full, centre_vox, eff_crop_size)
        seg_resized = resize_volume(seg_roi, target_size, order=0)  # nearest-neighbour

    centre_crop    = centre_vox - crop_start
    scale          = np.array([target_size / roi.shape[0],
                                target_size / roi.shape[1],
                                target_size / roi.shape[2]], dtype=np.float32)
    centre_resized = np.clip(centre_crop * scale, 0, target_size - 1)

    print(f"  Centre world/vox : {centre.round(2)}")
    print(f"  Centre DHW voxel : {centre_vox}")
    print(f"  Centre crop      : {centre_crop}")
    print(f"  Centre resized   : {centre_resized}")
    print(f"  ROI crop shape   : {roi.shape}")
    print(f"  Resized ROI shape: {arr_resized.shape}")
    print(f"  Resized ROI range: [{arr_resized.min():.3f}, {arr_resized.max():.3f}]")
    if seg_resized is not None:
        print(f"  Real segmentation carried through: {(seg_resized > 0).sum()} fg voxels")
    else:
        print(f"  No real segmentation available — fig7 will fall back to an intensity threshold")

    result.update({
        'success'       : True,
        'arr_raw'       : arr_raw,
        'arr_norm'      : arr_norm,
        'arr_resized'   : arr_resized,
        'seg_resized'   : seg_resized,
        'spacing'       : spacing,
        'normal'        : normal,
        'centre'        : centre_resized / target_size,
        'centre_vox'    : centre_vox,
        'centre_crop'   : centre_crop,
        'centre_resized': centre_resized,
        'out_dir'       : out_dir,
        'target_size'   : target_size,
        'edge_warnings' : edge_warnings,
        'qc_pass'       : len(edge_warnings) == 0,
    })

    proc_img_path = os.path.join(out_dir, f'{patient_id}_resized_{target_size}.npy')
    np.save(proc_img_path, arr_resized)
    print(f"  Saved: {proc_img_path}")

    if seg_resized is not None:
        proc_seg_path = os.path.join(out_dir, f'{patient_id}_seg_resized_{target_size}.npy')
        np.save(proc_seg_path, seg_resized)
        print(f"  Saved: {proc_seg_path}")

    qc_status = "✓ PASS" if result['qc_pass'] else "✗ FAIL (centre near volume edge)"
    print(f"  Quality check    : {qc_status}")
    print(f"  ✓ Preprocessing complete")
    return result


# ══════════════════════════════════════════════
# SECTION 3 — VISUALISATIONS
# ══════════════════════════════════════════════

def fig1_preprocessing_steps(result: dict):
    pid   = result['patient_id']
    raw   = result['arr_raw']
    norm  = result['arr_norm']
    small = result['arr_resized']
    S     = result['target_size']

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(f'{pid} — Preprocessing pipeline: raw → normalised → resized',
                 fontsize=13, fontweight='bold')

    titles_row = ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)']

    for col, title in enumerate(titles_row):
        d0, d1, d2 = raw.shape
        slices_raw = [raw[d0//2, :, :], raw[:, d1//2, :], raw[:, :, d2//2]]
        vmin_raw = raw.min() if raw.min() < -1 else 0
        vmax_raw = raw.max() if raw.max() > 1 else 1
        im = axes[0, col].imshow(slices_raw[col], cmap='gray',
                                  vmin=vmin_raw, vmax=vmax_raw, origin='lower')
        axes[0, col].set_title(f'Loaded image — {title}', fontsize=9)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        d0, d1, d2 = norm.shape
        slices_norm = [norm[d0//2, :, :], norm[:, d1//2, :], norm[:, :, d2//2]]
        axes[1, col].imshow(slices_norm[col], cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[1, col].set_title(f'Normalised [0,1] — {title}', fontsize=9)
        axes[1, col].axis('off')

        slices_sm = [small[S//2, :, :], small[:, S//2, :], small[:, :, S//2]]
        axes[2, col].imshow(slices_sm[col], cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[2, col].set_title(f'Resized {S}³ — {title}', fontsize=9)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Step 1: Loaded',         fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Step 2: HU normalised',  fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel(f'Step 3: Resized {S}³',  fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(result['out_dir'], f'{pid}_fig1_preprocessing_steps.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig2_slices_with_plane(result: dict):
    pid    = result['patient_id']
    arr    = result['arr_resized']
    S      = result['target_size']
    normal = result['normal']
    centre = result['centre']   # normalised [0,1]

    cx = int(np.clip(centre[0] * S, 0, S - 1))
    cy = int(np.clip(centre[1] * S, 0, S - 1))
    cz = int(np.clip(centre[2] * S, 0, S - 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'{pid} — Orthogonal slices with annulus plane overlay',
                 fontsize=12, fontweight='bold')

    views = [
        (arr[cx, :, :],  f'Axial  (slice {cx})',    cy, cz, 'axial'),
        (arr[:, cy, :],  f'Coronal (slice {cy})',   cx, cz, 'coronal'),
        (arr[:, :, cz],  f'Sagittal (slice {cz})',  cx, cy, 'sagittal'),
    ]

    for ax, (slc, title, r, c, view_name) in zip(axes, views):
        ax.imshow(slc, cmap='gray', vmin=0, vmax=1, origin='lower')
        ax.plot(c, r, 'r*', markersize=14, label='Annulus centre', zorder=5)

        if normal is not None:
            draw_plane_line(ax, normal, (r, c), S, view=view_name)

        ax.set_title(title, fontsize=10)
        ax.axis('off')

    axes[0].legend(loc='upper right', fontsize=8)

    if normal is not None:
        info = (f"Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n"
                f"Centre slice: ({cx}, {cy}, {cz})")
        fig.text(0.02, 0.02, info, fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    path = os.path.join(result['out_dir'], f'{pid}_fig2_slices_with_plane.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig7_roi_overlay_quality_check(result: dict):
    """
    ROI quality check. Uses the REAL cropped+resized segmentation mask
    (result['seg_resized']) when available, instead of intensity-
    thresholding the image itself, which over-highlights almost the
    entire heart.
    """
    pid    = result['patient_id']
    arr    = result['arr_resized']
    S      = result['target_size']
    centre = result['centre']   # normalised [0,1]
    normal = result['normal']
    seg    = result.get('seg_resized')

    cx = int(np.clip(centre[0] * S, 0, S - 1))
    cy = int(np.clip(centre[1] * S, 0, S - 1))
    cz = int(np.clip(centre[2] * S, 0, S - 1))

    if seg is not None:
        mask = seg > 0.5
        overlay_label = "Real segmentation"
    else:
        # Fallback only — clearly labelled as approximate, and a higher
        # threshold so it doesn't swallow the whole heart.
        mask = arr > 0.55
        overlay_label = "Intensity threshold (no real seg available)"

    views = [
        (arr[cx, :, :], mask[cx, :, :], "Axial",    cy, cz, 'axial'),
        (arr[:, cy, :], mask[:, cy, :], "Coronal",   cx, cz, 'coronal'),
        (arr[:, :, cz], mask[:, :, cz], "Sagittal",  cx, cy, 'sagittal'),
    ]

    qc_pass = result.get('qc_pass', True)
    border_color = 'green' if qc_pass else 'red'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    qc_label = "QC PASS" if qc_pass else "QC FAIL — centre near volume edge"
    fig.suptitle(f"{pid} — Annulus-centred ROI quality check ({overlay_label}) [{qc_label}]",
                 fontsize=13, fontweight="bold", color=border_color)

    for ax, (img, msk, title, r, c, view_name) in zip(axes, views):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.imshow(np.ma.masked_where(msk == 0, msk),
                  cmap="autumn", alpha=0.35, origin="lower")
        ax.plot(c, r, "r*", markersize=16, label="Annulus centre")

        if normal is not None:
            draw_plane_line(ax, normal, (r, c), S, view=view_name)

        ax.set_title(title, fontsize=11)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

    axes[0].legend(loc="upper right", fontsize=8)

    info = (
        f"Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n"
        f"Centre: ({cx}, {cy}, {cz})"
        if normal is not None else f"Centre: ({cx}, {cy}, {cz})"
    )
    if result.get('edge_warnings'):
        info += "\n" + "\n".join(w.strip() for w in result['edge_warnings'])
    fig.text(0.02, 0.02, info, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(result['out_dir'], f'{pid}_fig7_roi_overlay_quality_check.png')
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig3_hu_histogram(result: dict):
    pid  = result['patient_id']
    raw  = result['arr_raw']
    norm = result['arr_norm']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{pid} — Intensity distribution analysis',
                 fontsize=12, fontweight='bold')

    axes[0].hist(raw.flatten(), bins=200, color='steelblue', alpha=0.8, edgecolor='none')
    axes[0].set_xlabel('Raw loaded value')
    axes[0].set_ylabel('Voxel count')
    axes[0].set_title('Loaded data distribution')
    axes[0].set_yscale('log')

    if raw.min() < -1:  # raw HU data
        axes[0].axvline(-200, color='red',   linestyle='--', linewidth=2, label='Clip min (-200 HU)')
        axes[0].axvline(1000, color='green', linestyle='--', linewidth=2, label='Clip max (1000 HU)')
        axes[0].legend(fontsize=8)
        clipped = np.clip(raw, -200, 1000)
        axes[1].hist(clipped.flatten(), bins=200, color='coral', alpha=0.8, edgecolor='none')
        axes[1].set_xlabel('Hounsfield Units (HU)')
        axes[1].set_title('After clipping [-200, 1000] HU')
    else:
        axes[1].hist(raw.flatten(), bins=200, color='coral', alpha=0.8, edgecolor='none')
        axes[1].set_xlabel('Value')
        axes[1].set_title('(already normalised — no clip applied)')
    axes[1].set_yscale('log')

    axes[2].hist(norm.flatten(), bins=200, color='teal', alpha=0.8, edgecolor='none')
    axes[2].set_xlabel('Normalised intensity [0, 1]')
    axes[2].set_title('Final normalised [0, 1]')
    axes[2].set_yscale('log')

    stats = (f"Min: {raw.min():.3f}\nMax: {raw.max():.3f}\n"
             f"Mean: {raw.mean():.3f}\nStd: {raw.std():.3f}")
    axes[0].text(0.98, 0.98, stats, transform=axes[0].transAxes, fontsize=8,
                  va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    path = os.path.join(result['out_dir'], f'{pid}_fig3_hu_histogram.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig4_3d_aorta(result: dict):
    pid    = result['patient_id']
    arr    = result['arr_resized']
    S      = result['target_size']
    normal = result['normal']
    centre = result['centre']   # normalised [0,1]
    seg    = result.get('seg_resized')

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f'{pid} — 3D view of resized CT with annulus plane',
                 fontsize=12, fontweight='bold')

    # Prefer the real segmentation for the point cloud if available —
    # much more anatomically meaningful than a raw intensity threshold.
    if seg is not None and seg.max() > 0:
        coords = np.argwhere(seg > 0.5)
    else:
        threshold = 0.30
        coords = np.argwhere(arr > threshold)

    n_points = min(8000, len(coords))
    if len(coords) > n_points:
        idx    = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[idx]

    ax1 = fig.add_subplot(121, projection='3d')

    if len(coords) > 0:
        intensities = arr[coords[:, 0], coords[:, 1], coords[:, 2]]
        ax1.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                    c=intensities, cmap='gray', s=1.0, alpha=0.15, linewidths=0)
    else:
        ax1.text(0.5, 0.5, 0.5, 'No voxels above threshold',
                 ha='center', fontsize=9)

    cd = centre[0] * S
    ch = centre[1] * S
    cw = centre[2] * S

    ax1.scatter([cw], [ch], [cd], c='red', s=100, marker='*',
                zorder=10, label='Annulus centre')

    if normal is not None:
        u  = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(normal, u);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        t  = np.linspace(-15, 15, 30)
        T1, T2 = np.meshgrid(t, t)
        c_pt = np.array([cd, ch, cw])
        Xp = c_pt[2] + T1*v1[2] + T2*v2[2]
        Yp = c_pt[1] + T1*v1[1] + T2*v2[1]
        Zp = c_pt[0] + T1*v1[0] + T2*v2[0]
        ax1.plot_surface(Xp, Yp, Zp, alpha=0.3, color='yellow')
        ax1.quiver(c_pt[2], c_pt[1], c_pt[0],
                   normal[2]*15, normal[1]*15, normal[0]*15,
                   color='red', linewidth=2)

    ax1.set_xlabel('X (W)'); ax1.set_ylabel('Y (H)'); ax1.set_zlabel('Z (D)')
    ax1.set_xlim(0, S); ax1.set_ylim(0, S); ax1.set_zlim(0, S)
    ax1.set_title('3D view — ROI volume')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(122)
    mip = arr.max(axis=1)
    ax2.imshow(mip, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax2.set_title('Max intensity projection (coronal)')
    ax2.axis('off')
    ax2.plot(cw, cd, 'r*', markersize=14)

    plt.tight_layout()
    path = os.path.join(result['out_dir'], f'{pid}_fig4_3d_aorta.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def fig5_dataset_overview(results: list, out_path: str):
    successful = [r for r in results if r['success']]
    if not successful:
        print("  No successful patients — skipping dataset overview")
        return None

    n      = len(results)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3, n_rows * 3.2))
    fig.suptitle(f'Dataset overview — {len(successful)}/{n} patients preprocessed',
                 fontsize=13, fontweight='bold')

    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_1d(axes).flatten().tolist()

    for i, result in enumerate(results):
        if not result['success']:
            axes_flat[i].text(0.5, 0.5, f"{result['patient_id']}\nFailed",
                               ha='center', va='center', fontsize=9,
                               transform=axes_flat[i].transAxes)
            axes_flat[i].axis('off')
            continue

        arr    = result['arr_resized']
        S      = result['target_size']
        centre = result['centre']
        qc_ok  = result.get('qc_pass', True)

        axes_flat[i].imshow(arr[S//2, :, :], cmap='gray',
                             vmin=0, vmax=1, origin='lower')

        cy = int(centre[1] * S)
        cz = int(centre[2] * S)
        marker_color = 'lime' if qc_ok else 'red'
        axes_flat[i].plot(cz, cy, '+', color=marker_color, markersize=10, markeredgewidth=2)
        title_suffix = '' if qc_ok else ' ⚠'
        axes_flat[i].set_title(result['patient_id'] + title_suffix, fontsize=7,
                                color='black' if qc_ok else 'red')
        axes_flat[i].axis('off')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    path = os.path.join(out_path, 'fig5_dataset_overview.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nDataset overview saved → {path}")
    return path


def fig6_label_statistics(results: list, out_path: str):
    normals = []
    centres = []

    for r in results:
        if r['success'] and r['normal'] is not None and r['centre'] is not None:
            normals.append(r['normal'])
            centres.append(r['centre'])  # already stored normalised [0,1]

    if not normals:
        print("  No valid labels found for statistics")
        return None

    normals = np.array(normals)
    centres = np.array(centres)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Label statistics across {len(normals)} patients',
                 fontsize=12, fontweight='bold')

    for i, (comp, name) in enumerate(zip([normals[:,0], normals[:,1], normals[:,2]],
                                          ['nx', 'ny', 'nz'])):
        axes[0, i].hist(comp, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0, i].axvline(comp.mean(), color='red', linewidth=2,
                            label=f'Mean={comp.mean():.3f}')
        axes[0, i].set_title(f'Normal component — {name}')
        axes[0, i].set_xlabel('Value'); axes[0, i].set_ylabel('Count')
        axes[0, i].legend(fontsize=8)

    for i, (comp, name) in enumerate(zip([centres[:,0], centres[:,1], centres[:,2]],
                                          ['depth (D)', 'height (H)', 'width (W)'])):
        axes[1, i].hist(comp, bins=20, color='coral', edgecolor='white', alpha=0.8)
        axes[1, i].axvline(comp.mean(), color='darkred', linewidth=2,
                            label=f'Mean={comp.mean():.3f}')
        axes[1, i].set_title(f'Centre position — {name}')
        axes[1, i].set_xlabel('Normalised [0,1]'); axes[1, i].set_ylabel('Count')
        axes[1, i].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_path, 'fig6_label_statistics.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Label statistics saved → {path}")
    return path


# ══════════════════════════════════════════════
# SECTION 4 — MAIN
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Local preprocessing and visualisation pipeline'
    )
    parser.add_argument('--raw_data_path',    type=str, required=True)
    parser.add_argument('--out_path',         type=str, default='./processed_output')
    parser.add_argument('--patient_ids',      nargs='+', default=None)
    parser.add_argument('--target_size',      type=int,   default=64)
    parser.add_argument('--resample_spacing', type=float, default=1.0)
    parser.add_argument('--crop_size',        type=int,   default=128)
    args = parser.parse_args()

    if args.patient_ids is None or len(args.patient_ids) == 0:
        args.patient_ids = sorted([
            d for d in os.listdir(args.raw_data_path)
            if os.path.isdir(os.path.join(args.raw_data_path, d))
        ])

    os.makedirs(args.out_path, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  LOCAL PREPROCESSING AND VISUALISATION")
    print(f"  Patients : {args.patient_ids}")
    print(f"  Target   : {args.target_size}³ voxels")
    print(f"  Spacing  : {args.resample_spacing} mm isotropic")
    print(f"  Output   : {args.out_path}")
    print(f"{'='*55}")

    all_results   = []
    all_fig_paths = []

    for pid in args.patient_ids:
        result = preprocess_patient(
            args.raw_data_path, pid, args.out_path,
            args.target_size, args.resample_spacing, args.crop_size
        )
        all_results.append(result)

        if result['success']:
            manifest_row_path = os.path.join(args.out_path, 'manifest_rows_128.csv')
            write_header = not os.path.exists(manifest_row_path)
            img_path = os.path.join(result['out_dir'], f"{pid}_resized_{result['target_size']}.npy")
            seg_path = (os.path.join(result['out_dir'], f"{pid}_seg_resized_{result['target_size']}.npy")
                        if result.get('seg_resized') is not None else '')
            cr = result['centre_resized']
            nm = result['normal']
            with open(manifest_row_path, 'a') as mf:
                if write_header:
                    mf.write('patient_id,image_path,seg_path,centre_D,centre_H,centre_W,normal_D,normal_H,normal_W\n')
                mf.write(f"{pid},{img_path},{seg_path},{cr[0]},{cr[1]},{cr[2]},{nm[0]},{nm[1]},{nm[2]}\n")
            print(f"\n  Generating visualisations for {pid}...")
            all_fig_paths.append(fig1_preprocessing_steps(result))
            all_fig_paths.append(fig2_slices_with_plane(result))
            all_fig_paths.append(fig3_hu_histogram(result))
            all_fig_paths.append(fig4_3d_aorta(result))
            all_fig_paths.append(fig7_roi_overlay_quality_check(result))

    print(f"\n  Generating dataset-wide figures...")
    fig5_dataset_overview(all_results, args.out_path)
    fig6_label_statistics(all_results, args.out_path)

    n_success = sum(1 for r in all_results if r['success'])
    n_qc_pass = sum(1 for r in all_results if r.get('qc_pass'))
    print(f"\n{'='*55}")
    print(f"  COMPLETE")
    print(f"  Processed     : {n_success}/{len(args.patient_ids)} patients")
    print(f"  Quality check : {n_qc_pass}/{n_success} passed (centre not near volume edge)")
    print(f"  Figures       : {len(all_fig_paths) + 2} PNG files")
    print(f"  Output        : {args.out_path}")
    if n_qc_pass < n_success:
        failed = [r['patient_id'] for r in all_results if r['success'] and not r.get('qc_pass')]
        print(f"  ⚠ QC FAILED for: {failed}")
        print(f"    Inspect their fig7_roi_overlay_quality_check.png (red border) "
              f"and check coordinate_system / axis order for those specific files.")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()