"""
preprocess_and_visualise_combined.py
====================================
Combined preprocessing + fixed supervisor-quality visualisation script.

What it does:
  1. Loads processed/raw CT, segmentation, and landmark JSON.
  2. Applies HU clipping and normalisation.
  3. Converts annulus centre into CT array order (D,H,W).
  4. Crops an annulus-centred ROI before resizing.
  5. Resizes CT ROI to target_size^3.
  6. Resizes aligned segmentation ROI when available.
  7. Saves processed ROI .npy.
  8. Generates improved figures:
       - preprocessing overview
       - HU histogram
       - true angled plane line overlays
       - improved 3D aorta/annulus plane view
       - ROI quality check with segmentation overlay
       - dataset overview and label statistics

Run:
  python preprocess_and_visualise_combined.py \
      --raw_data_path ./data_processed \
      --out_path ./combined_output \
      --patient_ids CONTCT_R_08_FBA CONTCT_R_09_FBA CONTCT_R_16_FBA \
      --target_size 64 \
      --crop_size 160
"""

import os
import json
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# DATA LOADING
# ============================================================

def find_patient_files(raw_data_path: str, patient_id: str) -> dict:
    """Find CT, segmentation, and landmark files for one patient."""
    search_dirs = [
        os.path.join(raw_data_path, patient_id),
        os.path.join(raw_data_path, "BRC_retrospectives", patient_id),
        os.path.join(raw_data_path, "AC_cases_AS_OF_17042025", "Completed__confirmed", patient_id),
        os.path.join(raw_data_path, "AC_cases_AS_OF_17042025", "Completed__uncertain__to_be_checked", patient_id),
        raw_data_path,
    ]

    files = {
        "img": None,
        "seg": None,
        "landmark": None,
        "patient_dir": None,
        "processed_img": None,
        "processed_seg": None,
        "processed_lm": None,
    }

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # Processed AutoDataProcessor format
        processed_base = os.path.join(search_dir, "visualisation", "processed_data")
        proc_img = os.path.join(processed_base, f"{patient_id}_processed_image.nii.gz")
        proc_seg = os.path.join(processed_base, f"{patient_id}_processed_seg.nii.gz")
        proc_lm = os.path.join(processed_base, f"{patient_id}_processed_landmarks.json")
        if os.path.exists(proc_img) and os.path.exists(proc_lm):
            files.update({
                "processed_img": proc_img,
                "processed_seg": proc_seg if os.path.exists(proc_seg) else None,
                "processed_lm": proc_lm,
                "patient_dir": search_dir,
            })
            return files

        # Raw/simple format
        all_files = glob.glob(os.path.join(search_dir, "*"))
        img_candidates, seg_candidates, lm_candidates = [], [], []
        for f in all_files:
            fname = Path(f).name.lower()
            if ".mrk.json" in fname:
                lm_candidates.append(f)
            elif ".nrrd" in fname or ".nii.gz" in fname or ".nii" in fname:
                if "seg" in fname:
                    seg_candidates.append(f)
                else:
                    img_candidates.append(f)

        if img_candidates:
            files.update({
                "img": img_candidates[0],
                "seg": seg_candidates[0] if seg_candidates else None,
                "landmark": lm_candidates[0] if lm_candidates else None,
                "patient_dir": search_dir,
            })
            return files

    return files


def load_sitk_image(path: str):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    return img, arr, spacing, origin


def load_landmark_mrk_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    points = {}
    if "markups" in data:
        for markup in data["markups"]:
            for i, cp in enumerate(markup.get("controlPoints", [])[:3], 1):
                pos = cp.get("position", cp.get("Position", [0, 0, 0]))
                points[f"controlPoint_{i}"] = np.array(pos, dtype=np.float64)
    return points


def load_processed_landmarks(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def normalise_hu(arr: np.ndarray, clip_min: float = -200.0, clip_max: float = 1000.0) -> np.ndarray:
    arr = np.clip(arr, clip_min, clip_max)
    arr = (arr - clip_min) / (clip_max - clip_min)
    return arr.astype(np.float32)


def resize_volume(arr: np.ndarray, target_size: int = 64, order: int = 1) -> np.ndarray:
    return resize(
        arr,
        (target_size, target_size, target_size),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def resample_to_isotropic(sitk_img, target_spacing: float = 1.0):
    original_spacing = np.array(sitk_img.GetSpacing())
    original_size = np.array(sitk_img.GetSize())
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing)) for i in range(3)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([target_spacing] * 3)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1000)
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(sitk_img)


def compute_plane_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    v1, v2 = p2 - p1, p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    centre = (p1 + p2 + p3) / 3.0
    return normal.astype(np.float32), centre.astype(np.float32)


def prepare_centre_for_crop(centre, image_shape):
    """Convert centre to CT array voxel order (D,H,W)."""
    centre = np.asarray(centre, dtype=np.float32)
    D, H, W = image_shape

    if centre.max() <= 1.0:
        centre_vox = np.array([centre[0] * D, centre[1] * H, centre[2] * W], dtype=np.float32)
    else:
        # Most landmark JSON/processed centres are x,y,z. CT array is z,y,x=(D,H,W).
        centre_vox = centre[[2, 1, 0]].astype(np.float32)

    centre_vox[0] = np.clip(centre_vox[0], 0, D - 1)
    centre_vox[1] = np.clip(centre_vox[1], 0, H - 1)
    centre_vox[2] = np.clip(centre_vox[2], 0, W - 1)
    return centre_vox


def crop_around_centre(volume, centre_vox, crop_size=160):
    """Crop cubic ROI around centre_vox in (D,H,W)."""
    D, H, W = volume.shape
    cd, ch, cw = np.asarray(centre_vox, dtype=np.float32).astype(int)
    cd = int(np.clip(cd, 0, D - 1))
    ch = int(np.clip(ch, 0, H - 1))
    cw = int(np.clip(cw, 0, W - 1))

    crop_size = int(min(crop_size, D, H, W))
    half = crop_size // 2

    d0, h0, w0 = max(cd - half, 0), max(ch - half, 0), max(cw - half, 0)
    d1, h1, w1 = min(d0 + crop_size, D), min(h0 + crop_size, H), min(w0 + crop_size, W)

    # Keep requested crop size near boundaries.
    d0, h0, w0 = max(d1 - crop_size, 0), max(h1 - crop_size, 0), max(w1 - crop_size, 0)
    roi = volume[d0:d1, h0:h1, w0:w1]
    if roi.size == 0 or 0 in roi.shape:
        raise ValueError(f"Empty ROI crop: volume={volume.shape}, centre={centre_vox}, start={(d0,h0,w0)}, end={(d1,h1,w1)}")
    return roi, np.array([d0, h0, w0], dtype=np.float32)


def masked_for_visualisation(arr, seg=None):
    """Return aorta-focused image for figures only. Training array remains unchanged."""
    if seg is None:
        return arr
    mask = seg > 0
    focused = arr.copy()
    focused[~mask] *= 0.25  # dim non-aorta rather than delete it; keeps anatomical context faintly visible
    return focused


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess_patient(raw_data_path: str, patient_id: str, out_path: str, target_size: int = 64,
                       crop_size: int = 160, resample_spacing: float = 1.0) -> dict:
    print(f"\n{'='*55}\n  Processing: {patient_id}\n{'='*55}")
    out_dir = os.path.join(out_path, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    files = find_patient_files(raw_data_path, patient_id)
    result = {"patient_id": patient_id, "success": False}

    arr_seg = None
    cp1 = cp2 = cp3 = None

    if files.get("processed_img"):
        print("  Loading already-processed data...")
        _, arr_raw, spacing, _ = load_sitk_image(files["processed_img"])
        lm_data = load_processed_landmarks(files["processed_lm"])
        geo = lm_data.get("geometric_properties", {})
        normal = np.array(geo.get("normal_vector", [0, 0, 1]), dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        centre = np.array(geo.get("center_point", [0, 0, 0]), dtype=np.float32)
        lms = lm_data.get("landmarks", {})
        cp1 = np.array(lms.get("controlPoint_1_image", [0, 0, 0]), dtype=np.float32)
        cp2 = np.array(lms.get("controlPoint_2_image", [0, 0, 0]), dtype=np.float32)
        cp3 = np.array(lms.get("controlPoint_3_image", [0, 0, 0]), dtype=np.float32)
        if files.get("processed_seg"):
            _, arr_seg, _, _ = load_sitk_image(files["processed_seg"])
            arr_seg = (arr_seg > 0).astype(np.uint8)
    elif files.get("img"):
        print(f"  Loading raw image: {files['img']}")
        sitk_img_raw, arr_raw_orig, spacing_orig, _ = load_sitk_image(files["img"])
        print(f"  Raw shape: {arr_raw_orig.shape}, HU: [{arr_raw_orig.min():.0f}, {arr_raw_orig.max():.0f}]")
        sitk_img = resample_to_isotropic(sitk_img_raw, resample_spacing)
        arr_raw = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        spacing = np.array(sitk_img.GetSpacing())

        if files.get("seg"):
            _, arr_seg_raw, _, _ = load_sitk_image(files["seg"])
            # For raw path, this is approximate unless segmentation is already aligned.
            arr_seg = (resize(arr_seg_raw.astype(np.float32), arr_raw.shape, order=0, preserve_range=True, anti_aliasing=False) > 0).astype(np.uint8)

        normal, centre = None, None
        if files.get("landmark"):
            lm_pts = load_landmark_mrk_json(files["landmark"])
            if len(lm_pts) >= 3:
                pts = list(lm_pts.values())
                normal, centre = compute_plane_from_3_points(pts[0], pts[1], pts[2])
    else:
        print(f"  ✗ No data found for {patient_id}")
        return result

    if centre is None or normal is None:
        print("  ✗ Missing landmarks/plane label")
        return result

    print(f"  Image shape      : {arr_raw.shape}")
    print(f"  Normal vector    : {normal.round(4)} |n|={np.linalg.norm(normal):.4f}")
    print(f"  Centre original  : {np.asarray(centre).round(2)}")

    arr_norm = normalise_hu(arr_raw)
    centre_vox = prepare_centre_for_crop(centre, arr_norm.shape)
    roi, crop_start = crop_around_centre(arr_norm, centre_vox, crop_size=crop_size)
    arr_resized = resize_volume(roi, target_size, order=1)

    seg_roi = None
    seg_resized = None
    if arr_seg is not None and arr_seg.shape == arr_norm.shape:
        d0, h0, w0 = crop_start.astype(int)
        d1, h1, w1 = d0 + roi.shape[0], h0 + roi.shape[1], w0 + roi.shape[2]
        seg_roi = arr_seg[d0:d1, h0:h1, w0:w1]
        seg_resized = (resize(seg_roi.astype(np.float32), (target_size,)*3, order=0, preserve_range=True, anti_aliasing=False) > 0).astype(np.uint8)

    centre_crop = centre_vox - crop_start
    scale = np.array([target_size / roi.shape[0], target_size / roi.shape[1], target_size / roi.shape[2]], dtype=np.float32)
    centre_resized = np.clip(centre_crop * scale, 0, target_size - 1)
    centre_norm = centre_resized / target_size

    print(f"  ROI crop shape   : {roi.shape}   crop_size={crop_size}")
    print(f"  Crop start       : {crop_start.astype(int)}")
    print(f"  Centre DHW voxel : {centre_vox.round(2)}")
    print(f"  Centre resized   : {centre_resized.round(2)}")
    print(f"  Resized ROI      : {arr_resized.shape}")

    proc_img_path = os.path.join(out_dir, f"{patient_id}_resized_{target_size}.npy")
    np.save(proc_img_path, arr_resized)

    result.update({
        "success": True,
        "arr_raw": arr_raw,
        "arr_norm": arr_norm,
        "arr_roi": roi,
        "arr_resized": arr_resized,
        "arr_seg_resized": seg_resized,
        "spacing": spacing,
        "normal": normal,
        "centre": centre_norm,
        "centre_vox": centre_vox,
        "centre_crop": centre_crop,
        "centre_resized": centre_resized,
        "crop_start": crop_start,
        "crop_size": crop_size,
        "cp1": cp1,
        "cp2": cp2,
        "cp3": cp3,
        "out_dir": out_dir,
        "target_size": target_size,
    })
    print(f"  Saved: {proc_img_path}\n  ✓ Preprocessing complete")
    return result


# ============================================================
# TRUE PLANE LINE PROJECTION
# ============================================================

def compute_plane_line_on_slice(normal_3d, centre_3d_vox, slice_axis, slice_idx, img_size):
    n = normal_3d / (np.linalg.norm(normal_3d) + 1e-8)
    c = centre_3d_vox
    if slice_axis == 0:      # axial: display H,W
        a1, a2, fixed = 1, 2, 0
    elif slice_axis == 1:    # coronal: display D,W
        a1, a2, fixed = 0, 2, 1
    else:                    # sagittal: display D,H
        a1, a2, fixed = 0, 1, 2

    pts = []
    const = n[fixed] * (slice_idx - c[fixed])
    if abs(n[a2]) > 1e-6:
        for v1 in [0, img_size - 1]:
            v2 = c[a2] - (n[a1] * (v1 - c[a1]) + const) / n[a2]
            if 0 <= v2 <= img_size - 1:
                pts.append((float(v1), float(v2)))
    if abs(n[a1]) > 1e-6:
        for v2 in [0, img_size - 1]:
            v1 = c[a1] - (n[a2] * (v2 - c[a2]) + const) / n[a1]
            if 0 <= v1 <= img_size - 1:
                pts.append((float(v1), float(v2)))
    if len(pts) < 2:
        return None

    unique = []
    for p in pts:
        if not any(np.hypot(p[0]-q[0], p[1]-q[1]) < 1e-3 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        best = max(((i, j, np.hypot(unique[i][0]-unique[j][0], unique[i][1]-unique[j][1]))
                    for i in range(len(unique)) for j in range(i+1, len(unique))), key=lambda x: x[2])
        p0, p1 = unique[best[0]], unique[best[1]]
    else:
        p0, p1 = unique
    return p0[1], p0[0], p1[1], p1[0]  # x1,y1,x2,y2


# ============================================================
# FIGURES
# ============================================================

def fig1_preprocessing_steps(result):
    pid, raw, norm, small, S = result["patient_id"], result["arr_raw"], result["arr_norm"], result["arr_resized"], result["target_size"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(f"{pid} — Preprocessing: raw → normalised → annulus-centred ROI {S}³", fontsize=13, fontweight="bold")
    for col, title in enumerate(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]):
        d0, d1, d2 = raw.shape
        raw_slices = [raw[d0//2], raw[:, d1//2, :], raw[:, :, d2//2]]
        im = axes[0, col].imshow(raw_slices[col], cmap="gray", vmin=-200, vmax=800, origin="lower")
        axes[0, col].set_title(f"Raw CT — {title}", fontsize=9); axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        d0, d1, d2 = norm.shape
        norm_slices = [norm[d0//2], norm[:, d1//2, :], norm[:, :, d2//2]]
        axes[1, col].imshow(norm_slices[col], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[1, col].set_title(f"Normalised [0,1] — {title}", fontsize=9); axes[1, col].axis("off")

        slices = [small[S//2], small[:, S//2, :], small[:, :, S//2]]
        axes[2, col].imshow(slices[col], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[2, col].set_title(f"Annulus-centred ROI {S}³ — {title}", fontsize=9); axes[2, col].axis("off")
    axes[0,0].set_ylabel("Step 1: Raw HU", fontsize=10, fontweight="bold")
    axes[1,0].set_ylabel("Step 2: HU normalised", fontsize=10, fontweight="bold")
    axes[2,0].set_ylabel(f"Step 3: ROI {S}³", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(result["out_dir"], f"{pid}_fig1_preprocessing_steps.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig2_slices_fixed(result):
    pid, arr, S, normal = result["patient_id"], result["arr_resized"], result["target_size"], result["normal"]
    cx, cy, cz = np.clip(result["centre_resized"].astype(int), 0, S-1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"{pid} — Orthogonal slices with true annulus plane intersection", fontsize=12, fontweight="bold")
    views = [(arr[cx], f"Axial (slice {cx})", cy, cz, 0), (arr[:, cy, :], f"Coronal (slice {cy})", cx, cz, 1), (arr[:, :, cz], f"Sagittal (slice {cz})", cx, cy, 2)]
    for ax, (slc, title, row, col, axis_idx) in zip(axes, views):
        ax.imshow(slc, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        ax.plot(col, row, "r*", markersize=16, label="Annulus centre", zorder=10)
        pts = compute_plane_line_on_slice(normal, np.array([cx, cy, cz], dtype=float), axis_idx, [cx, cy, cz][axis_idx], S)
        if pts is not None:
            x1,y1,x2,y2 = pts
            ax.plot([x1,x2], [y1,y2], "y--", linewidth=2.5, label="Annulus plane")
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            ax.text(0.02, 0.04, f"Plane angle: {angle:.1f}°", transform=ax.transAxes, fontsize=8, color="yellow", bbox=dict(facecolor="black", alpha=0.6))
        ax.set_title(title, fontsize=10); ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
    fig.text(0.01, 0.04, f"Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\nCentre slice: ({cx}, {cy}, {cz})", fontsize=8.5, fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#FFF9E6", alpha=0.95))
    plt.tight_layout(rect=[0,0.08,1,1])
    path = os.path.join(result["out_dir"], f"{pid}_fig2_slices_FIXED.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig3_hu_histogram(result):
    pid, raw, norm = result["patient_id"], result["arr_raw"], result["arr_norm"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{pid} — HU distribution analysis", fontsize=12, fontweight="bold")
    axes[0].hist(raw.flatten(), bins=200, alpha=0.8, edgecolor="none")
    axes[0].axvline(-200, color="red", linestyle="--", linewidth=2, label="Clip min (-200 HU)")
    axes[0].axvline(1000, color="green", linestyle="--", linewidth=2, label="Clip max (1000 HU)")
    axes[0].set_xlabel("Hounsfield Units (HU)"); axes[0].set_ylabel("Voxel count"); axes[0].set_title("Raw HU distribution"); axes[0].legend(fontsize=8); axes[0].set_yscale("log")
    clipped = np.clip(raw, -200, 1000)
    axes[1].hist(clipped.flatten(), bins=200, alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("Hounsfield Units (HU)"); axes[1].set_title("After clipping [-200, 1000]"); axes[1].set_yscale("log")
    axes[2].hist(norm.flatten(), bins=200, alpha=0.8, edgecolor="none")
    axes[2].set_xlabel("Normalised intensity [0,1]"); axes[2].set_title("After normalisation [0,1]"); axes[2].set_yscale("log")
    plt.tight_layout()
    path = os.path.join(result["out_dir"], f"{pid}_fig3_hu_histogram.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig4_3d_fixed(result):
    pid, arr, S, normal = result["patient_id"], result["arr_resized"], result["target_size"], result["normal"]
    seg = result.get("arr_seg_resized")
    mask = seg > 0 if seg is not None else arr > 0.45
    coords = np.argwhere(mask)
    if len(coords) > 7000:
        idx = np.random.choice(len(coords), 7000, replace=False); coords = coords[idx]
    intensities = arr[coords[:,0], coords[:,1], coords[:,2]]
    cx, cy, cz = np.clip(result["centre_resized"].astype(int), 0, S-1)
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"{pid} — 3D aorta segmentation with annulus plane\nLeft: full volume | Right: zoomed annulus region", fontsize=12, fontweight="bold")
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(coords[:,2], coords[:,1], coords[:,0], c=intensities, cmap="Blues", s=10, alpha=0.55, label="Aorta")
    plt.colorbar(sc, ax=ax1, fraction=0.03, pad=0.1, label="Normalised intensity")
    ax1.scatter([cz], [cy], [cx], c="red", s=300, marker="*", edgecolors="darkred", zorder=10, label="Annulus centre")
    u = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    v1 = np.cross(normal, u); v1 /= np.linalg.norm(v1) + 1e-8
    v2 = np.cross(normal, v1)
    R = 12
    th = np.linspace(0, 2*np.pi, 80)
    disc_x = cz + R*(np.cos(th)*v1[2] + np.sin(th)*v2[2])
    disc_y = cy + R*(np.cos(th)*v1[1] + np.sin(th)*v2[1])
    disc_z = cx + R*(np.cos(th)*v1[0] + np.sin(th)*v2[0])
    ax1.plot(disc_x, disc_y, disc_z, color="gold", lw=2.5, label="Plane boundary")
    t = np.linspace(-14, 14, 20); T1, T2 = np.meshgrid(t, t)
    Xp = cz + T1*v1[2] + T2*v2[2]; Yp = cy + T1*v1[1] + T2*v2[1]; Zp = cx + T1*v1[0] + T2*v2[0]
    ax1.plot_surface(Xp, Yp, Zp, alpha=0.22, color="yellow")
    ax1.quiver(cz, cy, cx, normal[2]*20, normal[1]*20, normal[0]*20, color="red", linewidth=3, arrow_length_ratio=0.2)
    ax1.set_xlabel("X (W)"); ax1.set_ylabel("Y (H)"); ax1.set_zlabel("Z (D)"); ax1.set_title("Full volume view", fontsize=10); ax1.legend(fontsize=8, loc="upper left")
    ax2 = fig.add_subplot(122, projection="3d")
    zoom = 20
    zm = ((coords[:,0] >= cx-zoom) & (coords[:,0] <= cx+zoom) & (coords[:,1] >= cy-zoom) & (coords[:,1] <= cy+zoom) & (coords[:,2] >= cz-zoom) & (coords[:,2] <= cz+zoom))
    zc = coords[zm]
    if len(zc): ax2.scatter(zc[:,2], zc[:,1], zc[:,0], c=arr[zc[:,0], zc[:,1], zc[:,2]], cmap="Blues", s=12, alpha=0.65)
    ax2.scatter([cz], [cy], [cx], c="red", s=400, marker="*", edgecolors="darkred", linewidth=1.5, zorder=10)
    ax2.plot(disc_x, disc_y, disc_z, color="gold", lw=3); ax2.plot_surface(Xp, Yp, Zp, alpha=0.30, color="yellow")
    ax2.quiver(cz, cy, cx, normal[2]*15, normal[1]*15, normal[0]*15, color="red", linewidth=3.5, arrow_length_ratio=0.25)
    ax2.set_xlim(cz-zoom, cz+zoom); ax2.set_ylim(cy-zoom, cy+zoom); ax2.set_zlim(cx-zoom, cx+zoom)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z"); ax2.set_title("Zoomed annulus region", fontsize=10)
    fig.text(0.01, 0.12, f"Normal vector: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\nCentre (vox): ({cx}, {cy}, {cz})\n|n| = {np.linalg.norm(normal):.5f}", fontsize=9, va="bottom", bbox=dict(boxstyle="round", facecolor="#FFF9E6", alpha=0.95), fontfamily="monospace")
    plt.tight_layout()
    path = os.path.join(result["out_dir"], f"{pid}_fig4_3d_FIXED.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig5_roi_quality(result, focus_aorta=True):
    pid, arr, S, normal = result["patient_id"], result["arr_resized"], result["target_size"], result["normal"]
    seg = result.get("arr_seg_resized")
    img = masked_for_visualisation(arr, seg) if focus_aorta else arr
    cx, cy, cz = np.clip(result["centre_resized"].astype(int), 0, S-1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"{pid} — Annulus-centred ROI quality check\nRed overlay = aorta segmentation | Red star = annulus centre", fontsize=12, fontweight="bold")
    views = [("Axial", img[cx], (cy,cz), 0), ("Coronal", img[:,cy,:], (cx,cz), 1), ("Sagittal", img[:,:,cz], (cx,cy), 2)]
    for ax, (name, slc, (r,c), axis_idx) in zip(axes, views):
        ax.imshow(slc, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        if seg is not None:
            seg_slc = seg[cx] if axis_idx == 0 else (seg[:,cy,:] if axis_idx == 1 else seg[:,:,cz])
            overlay = np.zeros((*seg_slc.shape, 4)); overlay[...,0] = 1.0; overlay[...,3] = seg_slc.astype(float) * 0.35
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.plot(c, r, "r*", markersize=16, label="Annulus centre", zorder=10)
        pts = compute_plane_line_on_slice(normal, np.array([cx,cy,cz], dtype=float), axis_idx, [cx,cy,cz][axis_idx], S)
        if pts is not None:
            x1,y1,x2,y2 = pts
            ax.plot([x1,x2], [y1,y2], "y--", linewidth=2.5, alpha=0.95, label="Plane")
        ax.set_title(name, fontsize=10); ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
    fig.text(0.01, 0.04, f"Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\nCentre: ({cx}, {cy}, {cz})", fontsize=8.5, fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#FFF9E6", alpha=0.95))
    plt.tight_layout(rect=[0,0.08,1,1])
    path = os.path.join(result["out_dir"], f"{pid}_fig5_roi_quality.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig6_label_statistics(results, out_path):
    normals, centres = [], []
    for r in results:
        if r.get("success") and r.get("normal") is not None and r.get("centre") is not None:
            normals.append(r["normal"]); centres.append(r["centre"])
    if not normals:
        return None
    normals, centres = np.array(normals), np.array(centres)
    fig, axes = plt.subplots(2,3, figsize=(14,8))
    fig.suptitle(f"Label statistics across {len(normals)} patients", fontsize=12, fontweight="bold")
    for i, name in enumerate(["nx","ny","nz"]):
        axes[0,i].hist(normals[:,i], bins=20, edgecolor="white", alpha=0.8); axes[0,i].axvline(normals[:,i].mean(), color="red", lw=2); axes[0,i].set_title(f"Normal component — {name}")
    for i, name in enumerate(["depth (D)", "height (H)", "width (W)"]):
        axes[1,i].hist(centres[:,i], bins=20, edgecolor="white", alpha=0.8); axes[1,i].axvline(centres[:,i].mean(), color="red", lw=2); axes[1,i].set_title(f"Centre position — {name}"); axes[1,i].set_xlabel("Normalised [0,1]")
    plt.tight_layout()
    path = os.path.join(out_path, "fig6_label_statistics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


def fig7_dataset_overview(results, out_path):
    ok = [r for r in results if r.get("success")]
    if not ok:
        return None
    n = len(ok); n_cols = min(5, n); n_rows = (n + n_cols - 1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.2))
    if n_rows == 1 and n_cols == 1: axes_flat = [axes]
    elif n_rows == 1: axes_flat = list(axes)
    else: axes_flat = axes.ravel()
    fig.suptitle(f"Dataset overview — {n} patients annulus-centred ROI", fontsize=13, fontweight="bold")
    for ax, r in zip(axes_flat, ok):
        arr = r["arr_resized"]; cx, cy, cz = np.clip(r["centre_resized"].astype(int), 0, r["target_size"]-1)
        ax.imshow(arr[cx], cmap="gray", vmin=0, vmax=1, origin="lower"); ax.plot(cz, cy, "r+", markersize=10, markeredgewidth=2); ax.set_title(r["patient_id"], fontsize=7); ax.axis("off")
    for ax in axes_flat[len(ok):]: ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_path, "fig7_dataset_overview.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Combined annulus preprocessing and fixed visualisation pipeline")
    p.add_argument("--raw_data_path", type=str, required=True, help="Folder containing patient folders/data_processed")
    p.add_argument("--out_path", type=str, default="./combined_output")
    p.add_argument("--patient_ids", nargs="*", help="Patient IDs. If omitted, process all folders under raw_data_path")
    p.add_argument("--target_size", type=int, default=64)
    p.add_argument("--crop_size", type=int, default=160, help="Annulus-centred crop size before resizing. Try 160 or 192.")
    p.add_argument("--resample_spacing", type=float, default=1.0)
    p.add_argument("--no_focus_aorta", action="store_true", help="Do not dim non-aorta tissue in ROI quality figure")
    args = p.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    if args.patient_ids:
        pids = args.patient_ids
    else:
        pids = sorted([d for d in os.listdir(args.raw_data_path) if os.path.isdir(os.path.join(args.raw_data_path, d))])

    print(f"\n{'='*60}\n COMBINED PREPROCESSING + FIXED VISUALISATION\n Patients : {len(pids)}\n Target   : {args.target_size}³\n Crop     : {args.crop_size}\n Output   : {args.out_path}\n{'='*60}")

    all_results = []
    for pid in pids:
        r = preprocess_patient(args.raw_data_path, pid, args.out_path, args.target_size, args.crop_size, args.resample_spacing)
        all_results.append(r)
        if not r.get("success"):
            continue
        print(f"\n  Generating figures for {pid}...")
        fig1_preprocessing_steps(r)
        fig2_slices_fixed(r)
        fig3_hu_histogram(r)
        fig4_3d_fixed(r)
        fig5_roi_quality(r, focus_aorta=not args.no_focus_aorta)

    print("\n  Generating dataset-wide figures...")
    fig6_label_statistics(all_results, args.out_path)
    fig7_dataset_overview(all_results, args.out_path)

    n_success = sum(1 for r in all_results if r.get("success"))
    print(f"\n{'='*60}\n COMPLETE\n Processed : {n_success}/{len(pids)} patients\n Output    : {args.out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
