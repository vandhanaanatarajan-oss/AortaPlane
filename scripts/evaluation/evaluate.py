"""
evaluate.py
-----------
Main evaluation entry-point.
Works for both output types:
  - 'landmarks'    : model predicts RC / NC / LC
  - 'centre_normal': model predicts annulus centre + plane normal vector

Usage
-----
From Python:
    from scripts.evaluation.evaluate import evaluate_patient, evaluate_dataset

From CLI:
    python evaluate.py --pred_dir /path/to/preds --gt_dir /path/to/gt \
                       --output_type landmarks --save_csv results/eval.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# NOTE: this file lives at scripts/evaluation/evaluate.py, so two levels up
# from its own directory is the repo root, where `src/` lives. Confirmed
# correct for this location (Part 7.8 of the project log) -- kept as-is.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.metrics import (
    compute_centre,
    compute_normal,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gt_json(json_path):
    """
    Load a ground-truth landmark JSON produced by the project pipeline.

    Expected keys (image-space coordinates):
        'controlPoint_1_image' → RC
        'controlPoint_2_image' → NC
        'controlPoint_3_image' → LC

    Returns
    -------
    dict with keys: RC, NC, LC, centre, normal
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rc = np.array(data['controlPoint_1_image'])  # RC
    nc = np.array(data['controlPoint_2_image'])  # NC
    lc = np.array(data['controlPoint_3_image'])  # LC

    centre = compute_centre(rc, nc, lc)
    normal = compute_normal(rc, nc, lc)

    return {
        'RC':     rc.tolist(),
        'NC':     nc.tolist(),
        'LC':     lc.tolist(),
        'centre': centre.tolist(),
        'normal': normal.tolist(),
    }


def load_pred_json(json_path, output_type):
    """
    Load model prediction JSON.

    Type A ('landmarks') expected format:
        {"RC": [x,y,z], "NC": [x,y,z], "LC": [x,y,z]}

    Type B ('centre_normal') expected format:
        {"centre": [cx,cy,cz], "normal": [nx,ny,nz]}

    Returns
    -------
    dict matching the expected structure for output_type.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if output_type == 'landmarks':
        required = ['RC', 'NC', 'LC']
    else:
        required = ['centre', 'normal']

    for key in required:
        if key not in data:
            raise KeyError(f"Prediction file {json_path} missing key '{key}'.")

    return {k: np.array(data[k]).tolist() for k in required}


# ---------------------------------------------------------------------------
# Single-patient evaluation
# ---------------------------------------------------------------------------

def evaluate_patient(patient_id, pred, gt, output_type='landmarks'):
    """
    Evaluate one patient and return a flat metric dict.

    Parameters
    ----------
    patient_id  : str
    pred        : dict  (landmark or centre_normal predictions)
    gt          : dict  (ground truth with RC/NC/LC + derived centre/normal)
    output_type : str   'landmarks' | 'centre_normal'

    Returns
    -------
    dict  {'patient_id': ..., 'output_type': ..., <metrics>}
    """
    metrics = compute_all_metrics(pred, gt, output_type=output_type)
    metrics['patient_id']  = patient_id
    metrics['output_type'] = output_type

    # move identifiers to front
    ordered = {k: metrics[k] for k in ['patient_id', 'output_type']}
    ordered.update({k: v for k, v in metrics.items()
                    if k not in ('patient_id', 'output_type')})
    return ordered


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(patient_ids, pred_dir, gt_dir,
                     output_type='landmarks',
                     gt_suffix='_labels.json',
                     pred_suffix='_pred.json'):
    """
    Evaluate a list of patients and return a DataFrame.

    Parameters
    ----------
    patient_ids  : list[str]
    pred_dir     : str | Path  directory containing prediction JSON files
    gt_dir       : str | Path  directory containing ground-truth JSON files
    output_type  : str         'landmarks' | 'centre_normal'
    gt_suffix    : str         filename pattern: <patient_id><gt_suffix>
    pred_suffix  : str         filename pattern: <patient_id><pred_suffix>

    Returns
    -------
    pd.DataFrame  one row per patient + a 'MEAN' summary row
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    rows = []
    for pid in patient_ids:
        pred_path = pred_dir / f"{pid}{pred_suffix}"
        gt_path   = gt_dir   / f"{pid}{gt_suffix}"

        if not pred_path.exists():
            print(f"[WARN] Prediction file not found: {pred_path}")
            continue
        if not gt_path.exists():
            print(f"[WARN] GT file not found: {gt_path}")
            continue

        try:
            pred = load_pred_json(pred_path, output_type)
            gt   = load_gt_json(gt_path)
            row  = evaluate_patient(pid, pred, gt, output_type)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] Patient {pid}: {e}")

    if not rows:
        raise RuntimeError("No patients were successfully evaluated.")

    df = pd.DataFrame(rows)

    # numeric columns only for mean row
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    mean_row = {col: df[col].mean() for col in numeric_cols}
    mean_row['patient_id']  = 'MEAN'
    mean_row['output_type'] = output_type

    summary_df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return summary_df


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

_METRIC_GROUPS = {
    'Landmark Errors (mm)': [
        'MAE_RC_mm', 'MAE_NC_mm', 'MAE_LC_mm', 'MAE_landmarks_mean_mm',
        'Euclidean_RC_mm', 'Euclidean_NC_mm', 'Euclidean_LC_mm',
        'Euclidean_landmarks_mean_mm',
    ],
    'Centre Errors (mm)': [
        'centre_euclidean_mm',
        'centre_MAE_x_mm', 'centre_MAE_y_mm', 'centre_MAE_z_mm',
        'centre_MAE_mean_mm',
        'pred_centre_to_gt_plane_mm',
    ],
    'Normal Vector Errors': [
        'normal_angle_error_deg',
        'normal_cosine_similarity',
        'normal_MAE_nx', 'normal_MAE_ny', 'normal_MAE_nz', 'normal_MAE_mean',
    ],
}


def print_summary(df, output_type):
    """Print a formatted per-group metric summary to stdout."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY  |  output_type = {output_type}")
    print("=" * 60)

    mean_row = df[df['patient_id'] == 'MEAN'].iloc[0]

    for group_name, cols in _METRIC_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        print(f"\n  {group_name}")
        print("  " + "-" * 40)
        for col in available:
            val = mean_row.get(col, float('nan'))
            if not np.isnan(val):
                print(f"    {col:<40s}  {val:.4f}")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate aortic annulus plane detection predictions."
    )
    parser.add_argument('--pred_dir',    required=True,
                        help='Directory with prediction JSON files.')
    parser.add_argument('--gt_dir',      required=True,
                        help='Directory with ground-truth JSON files.')
    parser.add_argument('--patient_ids', nargs='+', default=None,
                        help='List of patient IDs. If omitted, discovers from pred_dir.')
    parser.add_argument('--output_type', choices=['landmarks', 'centre_normal'],
                        default='landmarks',
                        help='Prediction output type (default: landmarks).')
    parser.add_argument('--gt_suffix',   default='_labels.json',
                        help='GT filename suffix (default: _labels.json).')
    parser.add_argument('--pred_suffix', default='_pred.json',
                        help='Pred filename suffix (default: _pred.json).')
    parser.add_argument('--save_csv',    default=None,
                        help='Optional path to save per-patient CSV results.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # discover patient IDs from pred_dir if not supplied
    if args.patient_ids is None:
        pred_files = list(Path(args.pred_dir).glob(f'*{args.pred_suffix}'))
        patient_ids = [f.name.replace(args.pred_suffix, '') for f in pred_files]
        if not patient_ids:
            raise RuntimeError(f"No prediction files found in {args.pred_dir}")
        print(f"Discovered {len(patient_ids)} patients.")
    else:
        patient_ids = args.patient_ids

    df = evaluate_dataset(
        patient_ids,
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_type=args.output_type,
        gt_suffix=args.gt_suffix,
        pred_suffix=args.pred_suffix,
    )

    print_summary(df, args.output_type)

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, float_format='%.4f')
        print(f"Results saved → {out_path}")