"""
fixed_visualise.py
==================
FIXES based on supervisor feedback:

FIX 1: 3D view — voxel points are now LARGER (s=8 not s=1) and
        segmentation is filtered to aorta-relevant threshold only,
        so points are clearly visible in the 3D plot.

FIX 2: Plane line in 2D slices is now CORRECTLY ANGLED (not always horizontal).
        The bug was: normal[in_plane_axes] was picking wrong components.
        Now uses proper 3D-to-2D projection of the normal vector.
        The line rotates according to the actual plane tilt for each patient.

FIX 3: Added a new fig_roi_quality figure that shows a zoomed crop
        around the annulus so the supervisor can verify the centre is
        sitting on the correct anatomical landmark.

Run:
  python fixed_visualise.py \
      --data_processed_path /path/to/data_processed \
      --out_path ./fixed_output \
      --patient_ids CONTCT_R_08_FBA CONTCT_R_09_FBA D1
"""

import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════

def load_patient(data_processed_path, patient_id):
    base = os.path.join(data_processed_path, patient_id,
                        'visualisation', 'processed_data')
    img_path = os.path.join(base, f'{patient_id}_processed_image.nii.gz')
    seg_path = os.path.join(base, f'{patient_id}_processed_seg.nii.gz')
    lm_path  = os.path.join(base, f'{patient_id}_processed_landmarks.json')

    if not os.path.exists(img_path):
        print(f"  ✗ Not found: {img_path}")
        return {'success': False, 'patient_id': patient_id}

    sitk_img = sitk.ReadImage(img_path)
    arr_raw  = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    spacing  = np.array(sitk_img.GetSpacing())

    result = {
        'success': True, 'patient_id': patient_id,
        'arr_raw': arr_raw, 'spacing': spacing,
        'img_shape': arr_raw.shape,
        'has_seg': False, 'has_landmarks': False,
    }

    if os.path.exists(seg_path):
        sitk_seg = sitk.ReadImage(seg_path)
        result['arr_seg'] = sitk.GetArrayFromImage(sitk_seg).astype(np.uint8)
        result['has_seg'] = True

    if os.path.exists(lm_path):
        with open(lm_path) as f:
            lm = json.load(f)
        geo    = lm.get('geometric_properties', {})
        normal = np.array(geo.get('normal_vector', [0,0,1]), dtype=np.float32)
        centre = np.array(geo.get('center_point',  [0,0,0]), dtype=np.float32)
        lm_pts = lm.get('landmarks', {})
        result.update({
            'has_landmarks': True,
            'normal': normal / (np.linalg.norm(normal) + 1e-8),
            'centre': centre,
            'cp1': np.array(lm_pts.get('controlPoint_1_image', [0,0,0])),
            'cp2': np.array(lm_pts.get('controlPoint_2_image', [0,0,0])),
            'cp3': np.array(lm_pts.get('controlPoint_3_image', [0,0,0])),
        })

    print(f"  ✓ {patient_id}  shape={arr_raw.shape}  "
          f"seg={'Y' if result['has_seg'] else 'N'}  "
          f"lm={'Y' if result['has_landmarks'] else 'N'}")
    return result


def preprocess(result, target_size=64):
    if not result['success']:
        return result
    raw = result['arr_raw']
    arr_norm    = np.clip(raw, -200, 1000)
    arr_norm    = (arr_norm + 200) / 1200.0
    arr_resized = resize(arr_norm, (target_size,)*3,
                         order=1, preserve_range=True,
                         anti_aliasing=False).astype(np.float32)
    S = target_size

    arr_seg_resized = None
    if result['has_seg']:
        arr_seg_resized = resize(
            result['arr_seg'].astype(np.float32), (S,)*3,
            order=0, preserve_range=True,
            anti_aliasing=False).astype(np.uint8)

    centre_resized = None
    if result['has_landmarks']:
        D, H, W = result['img_shape']
        c = result['centre'].astype(np.float32)

        # Convert from JSON/processed landmark order into CT array order
        # If centre is stored as (x,y,z), convert to (z,y,x)
        if c.max() > 1.0:
            c_dhw = c[[2, 1, 0]]
        else:
            c_dhw = c

        centre_resized = np.array([
            c_dhw[0] / D * S,
            c_dhw[1] / H * S,
            c_dhw[2] / W * S
        ], dtype=np.float32)

        centre_resized = np.clip(centre_resized, 0, S - 1)

        print(f"  Centre original: {c}")
        print(f"  Centre DHW     : {c_dhw}")
        print(f"  Centre resized : {centre_resized}")

    result.update({
        'arr_norm': arr_norm,
        'arr_resized': arr_resized,
        'arr_seg_resized': arr_seg_resized,
        'centre_resized': centre_resized,
        'target_size': S,
    })
    return result


# ══════════════════════════════════════
# FIX 2 — CORRECT PLANE LINE DRAWING
# ══════════════════════════════════════

def compute_plane_line_on_slice(normal_3d, centre_3d_vox, slice_axis,
                                 slice_idx, img_size):
    """
    Correctly project the 3D annulus plane onto a 2D orthogonal slice.

    The plane equation is: normal · (p - centre) = 0
    On a slice at position slice_idx along slice_axis, we find the
    intersection of this plane with the 2D image.

    slice_axis: 0=axial(depth), 1=coronal(height), 2=sagittal(width)

    Returns: (x1,y1,x2,y2) in image pixel coordinates, or None.
    """
    n = normal_3d / (np.linalg.norm(normal_3d) + 1e-8)
    c = centre_3d_vox  # (cx, cy, cz)

    # The two display axes on this slice
    if slice_axis == 0:   # axial:    display = (H, W) → (y=row, x=col)
        a1, a2 = 1, 2    # H, W
        fixed_axis = 0
    elif slice_axis == 1: # coronal:  display = (D, W)
        a1, a2 = 0, 2    # D, W
        fixed_axis = 1
    else:                 # sagittal: display = (D, H)
        a1, a2 = 0, 1    # D, H
        fixed_axis = 2

    n_fixed = n[fixed_axis]
    if abs(n_fixed) < 1e-6:
        # Plane nearly parallel to slice — skip drawing
        return None

    # Parametric: walk along a1 from 0 to S, solve for a2
    # n[a1]*(p[a1]-c[a1]) + n[a2]*(p[a2]-c[a2]) + n[fixed]*(slice_idx-c[fixed]) = 0
    # → p[a2] = c[a2] - (n[a1]*(p[a1]-c[a1]) + n[fixed]*(slice_idx-c[fixed])) / n[a2]
    pts = []
    const = n_fixed * (slice_idx - c[fixed_axis])

    if abs(n[a2]) > 1e-6:
        for v_a1 in [0, img_size-1]:
            v_a2 = c[a2] - (n[a1]*(v_a1 - c[a1]) + const) / n[a2]
            if 0 <= v_a2 <= img_size-1:
                pts.append((float(v_a1), float(v_a2)))

    if abs(n[a1]) > 1e-6:
        for v_a2 in [0, img_size-1]:
            v_a1 = c[a1] - (n[a2]*(v_a2 - c[a2]) + const) / n[a1]
            if 0 <= v_a1 <= img_size-1:
                pts.append((float(v_a1), float(v_a2)))

    if len(pts) < 2:
        return None

    # Remove duplicate or nearly identical intersection points
    unique_pts = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-3 for q in unique_pts):
            unique_pts.append(p)

    if len(unique_pts) < 2:
        return None

    if len(unique_pts) > 2:
        # Choose the two line endpoints farthest apart
        best = max(
            ((i, j, np.hypot(unique_pts[i][0] - unique_pts[j][0],
                            unique_pts[i][1] - unique_pts[j][1]))
             for i in range(len(unique_pts)) for j in range(i+1, len(unique_pts))),
            key=lambda x: x[2]
        )
        i, j, _ = best
        (a1_0, a2_0), (a1_1, a2_1) = unique_pts[i], unique_pts[j]
    else:
        (a1_0, a2_0), (a1_1, a2_1) = unique_pts

    # a1 → row (y), a2 → col (x)
    return a2_0, a1_0, a2_1, a1_1   # x1, y1, x2, y2


# ══════════════════════════════════════
# FIX 1 — IMPROVED 3D VIEW
# ══════════════════════════════════════

def fig4_fixed_3d(result, out_dir):
    """
    FIXED 3D view:
    - Point size increased from s=1 to s=8 (clearly visible)
    - Uses actual segmentation mask (not thresholding whole CT)
    - Separate full view and zoomed view side by side
    - Plane shown as a proper disc not just a mesh
    - Normal vector arrow is longer and thicker
    """
    pid    = result['patient_id']
    S      = result['target_size']
    normal = result.get('normal', None)

    # ── Get aorta point cloud ──
    if result.get('arr_seg_resized') is not None:
        # Use the actual segmentation mask — most accurate
        mask = result['arr_seg_resized'] > 0
    else:
        # Fallback: threshold CT at bright structures (aorta + calcification)
        mask = result['arr_resized'] > 0.45

    coords = np.argwhere(mask)
    print(f"    3D: {len(coords)} aorta voxels found")

    # Sample points for speed
    if len(coords) > 6000:
        idx    = np.random.choice(len(coords), 6000, replace=False)
        coords = coords[idx]

    # Intensity values for colouring by brightness
    intensities = result['arr_resized'][mask]
    if len(intensities) > 6000:
        intensities = intensities[idx]

    cx, cy, cz = (result['centre_resized'].astype(int)
                  if result['centre_resized'] is not None
                  else [S//2, S//2, S//2])

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f'{pid}  —  3D aorta segmentation with annulus plane\n'
        f'Left: full volume  |  Right: zoomed annulus region',
        fontsize=12, fontweight='bold'
    )

    # ─── LEFT: full 3D view ───
    ax1 = fig.add_subplot(121, projection='3d')

    # BIGGER points (s=8 not s=1), coloured by CT intensity
    sc = ax1.scatter(coords[:,2], coords[:,1], coords[:,0],
                     c=intensities, cmap='Blues', s=8,
                     alpha=0.5, label='Aorta')
    plt.colorbar(sc, ax=ax1, fraction=0.03, pad=0.1,
                 label='Normalised intensity')

    # Annulus centre — large red star
    ax1.scatter([cz], [cy], [cx], c='red', s=300, marker='*',
                zorder=10, label='Annulus centre', edgecolors='darkred')

    if normal is not None:
        # Plane disc (more realistic than a flat mesh)
        u  = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
        v1 = np.cross(normal, u);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        R  = 12   # radius in voxels
        th = np.linspace(0, 2*np.pi, 80)
        disc_x = cz + R*(np.cos(th)*v1[2] + np.sin(th)*v2[2])
        disc_y = cy + R*(np.cos(th)*v1[1] + np.sin(th)*v2[1])
        disc_z = cx + R*(np.cos(th)*v1[0] + np.sin(th)*v2[0])
        ax1.plot(disc_x, disc_y, disc_z,
                 color='gold', lw=2.5, label='Plane boundary')

        # Filled plane surface
        t = np.linspace(-14, 14, 20)
        T1, T2 = np.meshgrid(t, t)
        Xp = cz + T1*v1[2] + T2*v2[2]
        Yp = cy + T1*v1[1] + T2*v2[1]
        Zp = cx + T1*v1[0] + T2*v2[0]
        ax1.plot_surface(Xp, Yp, Zp, alpha=0.22, color='yellow')

        # Normal vector — longer and thicker arrow
        ax1.quiver(cz, cy, cx,
                   normal[2]*20, normal[1]*20, normal[0]*20,
                   color='red', linewidth=3, arrow_length_ratio=0.2)

    ax1.set_xlabel('X (W)'); ax1.set_ylabel('Y (H)'); ax1.set_zlabel('Z (D)')
    ax1.set_title('Full volume view', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')

    # ─── RIGHT: zoomed view ───
    ax2 = fig.add_subplot(122, projection='3d')
    zoom = 20
    zm = ((coords[:,0] >= cx-zoom) & (coords[:,0] <= cx+zoom) &
          (coords[:,1] >= cy-zoom) & (coords[:,1] <= cy+zoom) &
          (coords[:,2] >= cz-zoom) & (coords[:,2] <= cz+zoom))

    if zm.sum() > 0:
        zc  = coords[zm]
        zi  = intensities[zm]
        ax2.scatter(zc[:,2], zc[:,1], zc[:,0],
                    c=zi, cmap='Blues', s=12, alpha=0.6)

    ax2.scatter([cz], [cy], [cx], c='red', s=400, marker='*',
                zorder=10, edgecolors='darkred', linewidth=1.5)

    if normal is not None:
        ax2.plot(disc_x, disc_y, disc_z, color='gold', lw=3)
        ax2.plot_surface(Xp, Yp, Zp, alpha=0.30, color='yellow')
        ax2.quiver(cz, cy, cx,
                   normal[2]*15, normal[1]*15, normal[0]*15,
                   color='red', linewidth=3.5, arrow_length_ratio=0.25)

        # Mark 3 landmarks if available
        if result.get('cp1') is not None:
            D0, H0, W0 = result['img_shape']
            for cp, col, mk, lb in [
                (result['cp1'], 'red',   'o', 'RC'),
                (result['cp2'], 'blue',  's', 'NC'),
                (result['cp3'], 'green', '^', 'LC'),
            ]:
                px = cp[0]/D0*S; py = cp[1]/H0*S; pz = cp[2]/W0*S
                ax2.scatter([pz],[py],[px], c=col, s=120,
                            marker=mk, label=lb, zorder=8,
                            edgecolors='black', linewidth=1)
            ax2.legend(fontsize=8)

    ax2.set_xlim(cz-zoom, cz+zoom)
    ax2.set_ylim(cy-zoom, cy+zoom)
    ax2.set_zlim(cx-zoom, cx+zoom)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('Zoomed: annulus region', fontsize=10)

    # Normal vector info
    if normal is not None:
        fig.text(0.01, 0.12,
                 f'Normal vector: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n'
                 f'Centre (vox):  ({cx}, {cy}, {cz})\n'
                 f'|n| = {np.linalg.norm(normal):.5f} (should = 1.0)',
                 fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='#FFF9E6', alpha=0.95),
                 fontfamily='monospace')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{pid}_fig4_3d_FIXED.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"    Saved: fig4_3d_FIXED.png")
    return path


# ══════════════════════════════════════
# FIX 2 — CORRECT ANGLED PLANE LINE
# ══════════════════════════════════════

def fig2_fixed_plane_line(result, out_dir):
    """
    FIXED plane overlay:
    - Yellow line is now CORRECTLY ANGLED based on the actual normal vector
    - Was always horizontal because the 2D normal projection was wrong
    - Now uses proper 3D plane-slice intersection geometry
    - Shows angle of plane in degrees for each view
    """
    pid    = result['patient_id']
    arr    = result['arr_resized']
    S      = result['target_size']
    normal = result.get('normal', None)

    if result['centre_resized'] is not None:
        cx, cy, cz = np.clip(result['centre_resized'].astype(int), 0, S-1)
    else:
        cx = cy = cz = S//2

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f'{pid}  —  Orthogonal slices with annulus plane overlay\n'
        f'Yellow dashed line = true plane cross-section  (FIXED: correctly angled)',
        fontsize=11, fontweight='bold'
    )

    views = [
        (arr[cx,:,:],  f'Axial  (slice {cx})',    cy, cz, 0),
        (arr[:,cy,:],  f'Coronal (slice {cy})',    cx, cz, 1),
        (arr[:,:,cz],  f'Sagittal (slice {cz})',   cx, cy, 2),
    ]

    for ax, (slc, title, row, col_val, axis_idx) in zip(axes, views):
        ax.imshow(slc, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal')
        ax.plot(col_val, row, 'r*', markersize=16, zorder=10,
                label='Annulus centre')

        if normal is not None:
            # ── CORRECT plane line projection ──
            c3d = np.array([float(cx), float(cy), float(cz)])
            pts = compute_plane_line_on_slice(
                normal, c3d, axis_idx, [cx, cy, cz][axis_idx], S
            )
            if pts is not None:
                x1, y1, x2, y2 = pts
                x1 = np.clip(x1, 0, S-1)
                x2 = np.clip(x2, 0, S-1)
                y1 = np.clip(y1, 0, S-1)
                y2 = np.clip(y2, 0, S-1)
                ax.plot([x1, x2], [y1, y2],
                        color='yellow', linestyle='--', linewidth=2.5,
                        alpha=0.9, label='Annulus plane')

                # Show angle of the line in this view
                angle_deg = np.degrees(np.arctan2(y2-y1, x2-x1))
                ax.text(0.02, 0.04,
                        f'Plane angle: {angle_deg:.1f}°',
                        transform=ax.transAxes, fontsize=8,
                        color='yellow',
                        bbox=dict(facecolor='black', alpha=0.6))

        ax.set_title(title, fontsize=10)
        ax.axis('off')

    axes[0].legend(loc='upper right', fontsize=8,
                   facecolor='black', labelcolor='white')

    if normal is not None:
        t_info = (f'Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n'
                  f'Centre slice: ({cx}, {cy}, {cz})')
        fig.text(0.01, 0.04, t_info, fontsize=8.5,
                 bbox=dict(boxstyle='round', facecolor='#FFF9E6', alpha=0.95),
                 fontfamily='monospace')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(out_dir, f'{pid}_fig2_slices_FIXED.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"    Saved: fig2_slices_FIXED.png")
    return path


# ══════════════════════════════════════
# FIX 3 — ROI QUALITY CHECK (NEW)
# ══════════════════════════════════════

def fig_roi_quality(result, out_dir):
    """
    NEW FIGURE: Annulus-centred ROI quality check.
    Crops a 24x24 voxel window around the predicted annulus centre.
    Overlays the segmentation mask in semi-transparent red.
    This confirms the centre is on the correct anatomical landmark.
    """
    if not result['has_landmarks']:
        return None

    pid  = result['patient_id']
    arr  = result['arr_resized']
    S    = result['target_size']
    cx, cy, cz = np.clip(result['centre_resized'].astype(int), 2, S-3)
    normal = result.get('normal', None)
    seg    = result.get('arr_seg_resized', None)

    crop = 42# half-width of crop in voxels

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f'{pid}  —  Annulus-centred ROI quality check\n'
        f'Red overlay = aorta segmentation  |  Red star = annulus centre',
        fontsize=11, fontweight='bold'
    )

    views = [
        ('Axial',    arr[cx, :, :],    (cy, cz), 0),
        ('Coronal',  arr[:, cy, :],    (cx, cz), 1),
        ('Sagittal', arr[:, :, cz],    (cx, cy), 2),
    ]

    for ax, (view_name, slc, (r, c), axis_idx) in zip(axes, views):
        # Crop around centre
        r0, r1 = max(0, r-crop), min(S, r+crop)
        c0, c1 = max(0, c-crop), min(S, c+crop)
        slc_crop = slc[r0:r1, c0:c1]
        ax.imshow(
            slc_crop,
            cmap='gray',
            vmin=0,
            vmax=1,
            origin='lower',
            aspect='equal'
        )

        # Overlay segmentation
        if seg is not None:
            if axis_idx == 0:
                seg_slc = seg[cx, :, :][r0:r1, c0:c1]
            elif axis_idx == 1:
                seg_slc = seg[:, cy, :][r0:r1, c0:c1]
            else:
                seg_slc = seg[:, :, cz][r0:r1, c0:c1]

            seg_rgba = np.zeros((*seg_slc.shape, 4))
            seg_rgba[..., 0] = 1.0   # red
            seg_rgba[..., 3] = seg_slc.astype(float) * 0.45
            ax.imshow(seg_rgba, origin='lower', interpolation='nearest')

        # Mark centre in cropped coords
        centre_col = c - c0
        centre_row = r - r0
        ax.plot(centre_col, centre_row, 'r*', markersize=14, zorder=10,
                label='Annulus centre')

        # Plane line in crop
        if normal is not None:
            c3d = np.array([float(cx), float(cy), float(cz)])
            pts = compute_plane_line_on_slice(
                normal, c3d, axis_idx, [cx, cy, cz][axis_idx], S
            )
            if pts is not None:
                x1, y1, x2, y2 = pts
                ax.plot([x1 - c0, x2 - c0],
                        [y1 - r0, y2 - r0],
                        color='yellow', linestyle='--',
                        linewidth=2.5, alpha=0.9, label='Plane')

        ax.set_title(view_name, fontsize=10)
        ax.axis('off')

    axes[0].legend(loc='upper right', fontsize=8,
                   facecolor='black', labelcolor='white')

    if normal is not None:
        fig.text(0.01, 0.04,
                 f'Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n'
                 f'Centre: ({cx}, {cy}, {cz})',
                 fontsize=8.5, fontfamily='monospace',
                 bbox=dict(boxstyle='round',
                           facecolor='#FFF9E6', alpha=0.95))

    plt.tight_layout()
    path = os.path.join(out_dir, f'{pid}_fig5_roi_quality.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"    Saved: fig5_roi_quality.png")
    return path


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_processed_path', type=str, required=True)
    p.add_argument('--out_path', type=str, default='./fixed_output')
    p.add_argument('--patient_ids', nargs='*')
    p.add_argument('--target_size', type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    if args.patient_ids:
        pids = args.patient_ids
    else:
        pids = sorted([d for d in os.listdir(args.data_processed_path)
                       if os.path.isdir(os.path.join(args.data_processed_path, d))])

    print(f"\nProcessing {len(pids)} patients → {args.out_path}\n")

    for pid in pids:
        out_dir = os.path.join(args.out_path, pid)
        os.makedirs(out_dir, exist_ok=True)

        result = load_patient(args.data_processed_path, pid)
        if not result['success']:
            continue
        result = preprocess(result, args.target_size)

        fig4_fixed_3d(result, out_dir)        # Fix 1: bigger voxels
        fig2_fixed_plane_line(result, out_dir) # Fix 2: angled plane
        fig_roi_quality(result, out_dir)       # Fix 3: ROI check
        print()

    print(f"Done.  Open {args.out_path} to see all figures.")
    print(f"\nKey improvements made:")
    print(f"  fig4_3d_FIXED.png    — voxels are now s=8 (visible)")
    print(f"  fig2_slices_FIXED.png — plane line is now correctly angled")
    print(f"  fig5_roi_quality.png  — new: zoomed ROI with seg overlay")

if __name__ == '__main__':
    main()