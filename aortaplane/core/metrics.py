"""
metrics.py
----------
Individual metric functions for aortic annulus plane evaluation.
Supports both output types:
  - Type A: predicted landmarks (RC, NC, LC) → derive centre + normal
  - Type B: predicted centre + normal vector directly
"""

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_centre(rc, nc, lc):
    """Mean of three landmark points → annulus centre (x, y, z)."""
    return (np.array(rc) + np.array(nc) + np.array(lc)) / 3.0


def compute_normal(rc, nc, lc):
    """Cross product of (NC-RC) × (LC-RC), normalised to unit length."""
    rc, nc, lc = np.array(rc), np.array(nc), np.array(lc)
    v1 = nc - rc
    v2 = lc - rc
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("Degenerate landmark configuration: collinear points.")
    return normal / norm


def ensure_consistent_normal_direction(pred_normal, gt_normal):
    """
    Flip predicted normal if it points opposite to ground truth.
    Plane normals are direction-ambiguous (n and -n define the same plane).
    """
    pred = np.array(pred_normal)
    gt = np.array(gt_normal)
    if np.dot(pred, gt) < 0:
        pred = -pred
    return pred


# ---------------------------------------------------------------------------
# Landmark metrics  (Type A outputs)
# ---------------------------------------------------------------------------

def landmark_mae(pred_landmarks, gt_landmarks):
    """
    Mean Absolute Error per landmark and overall.

    Parameters
    ----------
    pred_landmarks : dict  {'RC': [x,y,z], 'NC': [x,y,z], 'LC': [x,y,z]}
    gt_landmarks   : dict  {'RC': [x,y,z], 'NC': [x,y,z], 'LC': [x,y,z]}

    Returns
    -------
    dict with per-landmark MAE (mm) and mean across all landmarks.
    """
    results = {}
    errors = []
    for key in ['RC', 'NC', 'LC']:
        pred = np.array(pred_landmarks[key])
        gt   = np.array(gt_landmarks[key])
        mae  = np.mean(np.abs(pred - gt))
        results[f'MAE_{key}_mm'] = float(mae)
        errors.append(mae)
    results['MAE_landmarks_mean_mm'] = float(np.mean(errors))
    return results


def landmark_euclidean_distance(pred_landmarks, gt_landmarks):
    """
    Euclidean (L2) distance per landmark and mean.

    Returns
    -------
    dict with per-landmark Euclidean distance (mm) and mean.
    """
    results = {}
    dists = []
    for key in ['RC', 'NC', 'LC']:
        pred = np.array(pred_landmarks[key])
        gt   = np.array(gt_landmarks[key])
        dist = np.linalg.norm(pred - gt)
        results[f'Euclidean_{key}_mm'] = float(dist)
        dists.append(dist)
    results['Euclidean_landmarks_mean_mm'] = float(np.mean(dists))
    return results


# ---------------------------------------------------------------------------
# Centre metrics
# ---------------------------------------------------------------------------

def centre_euclidean_distance(pred_centre, gt_centre):
    """
    Euclidean distance between predicted and ground-truth annulus centres.

    Returns
    -------
    float  distance in mm
    """
    pred = np.array(pred_centre)
    gt   = np.array(gt_centre)
    return float(np.linalg.norm(pred - gt))


def centre_mae(pred_centre, gt_centre):
    """
    Per-axis and overall MAE for annulus centre prediction.

    Returns
    -------
    dict with axis-wise MAE and mean MAE (mm).
    """
    pred = np.array(pred_centre)
    gt   = np.array(gt_centre)
    abs_err = np.abs(pred - gt)
    return {
        'centre_MAE_x_mm': float(abs_err[0]),
        'centre_MAE_y_mm': float(abs_err[1]),
        'centre_MAE_z_mm': float(abs_err[2]),
        'centre_MAE_mean_mm': float(np.mean(abs_err)),
    }


# ---------------------------------------------------------------------------
# Normal vector metrics
# ---------------------------------------------------------------------------

def normal_angle_error_deg(pred_normal, gt_normal):
    """
    Angular difference between predicted and ground-truth plane normals.
    Handles the sign ambiguity: the angle is always in [0°, 90°].

    Returns
    -------
    float  angle in degrees
    """
    pred = np.array(pred_normal, dtype=float)
    gt   = np.array(gt_normal,   dtype=float)

    # normalise defensively
    pred = pred / (np.linalg.norm(pred) + 1e-8)
    gt   = gt   / (np.linalg.norm(gt)   + 1e-8)

    cos_angle = np.clip(np.abs(np.dot(pred, gt)), 0.0, 1.0)  # abs → sign-agnostic
    angle_deg = np.degrees(np.arccos(cos_angle))
    return float(angle_deg)


def normal_cosine_similarity(pred_normal, gt_normal):
    """
    Absolute cosine similarity between predicted and ground-truth normals.
    1.0 = perfect alignment, 0.0 = perpendicular.

    Returns
    -------
    float in [0, 1]
    """
    pred = np.array(pred_normal, dtype=float)
    gt   = np.array(gt_normal,   dtype=float)
    pred = pred / (np.linalg.norm(pred) + 1e-8)
    gt   = gt   / (np.linalg.norm(gt)   + 1e-8)
    return float(np.abs(np.dot(pred, gt)))


def normal_component_mae(pred_normal, gt_normal):
    """
    Per-component MAE between (sign-aligned) predicted and ground-truth normals.

    Returns
    -------
    dict with per-component MAE and mean.
    """
    pred = np.array(pred_normal, dtype=float)
    gt   = np.array(gt_normal,   dtype=float)
    pred = pred / (np.linalg.norm(pred) + 1e-8)
    gt   = gt   / (np.linalg.norm(gt)   + 1e-8)

    # align sign before computing component error
    pred = ensure_consistent_normal_direction(pred, gt)

    abs_err = np.abs(pred - gt)
    return {
        'normal_MAE_nx': float(abs_err[0]),
        'normal_MAE_ny': float(abs_err[1]),
        'normal_MAE_nz': float(abs_err[2]),
        'normal_MAE_mean': float(np.mean(abs_err)),
    }


# ---------------------------------------------------------------------------
# Plane distance metric
# ---------------------------------------------------------------------------

def point_to_plane_distance(point, plane_centre, plane_normal):
    """
    Signed distance from `point` to the plane defined by
    (plane_centre, plane_normal).

    Useful for checking how far a predicted centre lies from the GT plane,
    or vice-versa.

    Returns
    -------
    float  signed distance (mm); absolute value is the geometric distance.
    """
    n = np.array(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-8)
    return float(np.dot(np.array(point) - np.array(plane_centre), n))


# ---------------------------------------------------------------------------
# Aggregate: compute all metrics given output type
# ---------------------------------------------------------------------------

def compute_all_metrics(pred, gt, output_type='landmarks'):
    """
    Compute the full metric suite for one patient.

    Parameters
    ----------
    pred : dict
        Type A ('landmarks'): {'RC': [...], 'NC': [...], 'LC': [...]}
        Type B ('centre_normal'): {'centre': [...], 'normal': [...]}

    gt : dict
        Same structure as pred, plus always contains:
        {'RC': [...], 'NC': [...], 'LC': [...],
         'centre': [...], 'normal': [...]}

    output_type : str  'landmarks' | 'centre_normal'

    Returns
    -------
    dict of all scalar metric values for this patient.
    """
    metrics = {}

    if output_type == 'landmarks':
        # ---- derive predicted centre + normal from predicted landmarks ----
        pred_centre = compute_centre(pred['RC'], pred['NC'], pred['LC'])
        pred_normal = compute_normal(pred['RC'], pred['NC'], pred['LC'])

        # landmark metrics
        metrics.update(landmark_mae(pred, gt))
        metrics.update(landmark_euclidean_distance(pred, gt))

    elif output_type == 'centre_normal':
        pred_centre = np.array(pred['centre'])
        pred_normal = np.array(pred['normal'])

    else:
        raise ValueError(f"Unknown output_type: {output_type!r}. "
                         "Choose 'landmarks' or 'centre_normal'.")

    # ---- GT centre + normal (always available in gt dict) ----
    gt_centre = np.array(gt['centre'])
    gt_normal = np.array(gt['normal'])

    # centre metrics
    metrics['centre_euclidean_mm'] = centre_euclidean_distance(pred_centre, gt_centre)
    metrics.update(centre_mae(pred_centre, gt_centre))

    # normal metrics
    metrics['normal_angle_error_deg']   = normal_angle_error_deg(pred_normal, gt_normal)
    metrics['normal_cosine_similarity'] = normal_cosine_similarity(pred_normal, gt_normal)
    metrics.update(normal_component_mae(pred_normal, gt_normal))

    # plane distance: how far is the predicted centre from GT plane?
    metrics['pred_centre_to_gt_plane_mm'] = abs(
        point_to_plane_distance(pred_centre, gt_centre, gt_normal)
    )

    return metrics
