"""
conformal_gating.py
====================
Upgrades the cross-model disagreement gating rule from a correlational,
threshold-swept heuristic (precision/recall on an arbitrary cutoff) to a
formally calibrated split-conformal / Mondrian-conformal procedure with a
distribution-free finite-sample coverage guarantee.

BACKGROUND (per project results)
---------------------------------
Leak-free, n=37: cross-model disagreement (|M1 angle - M2 angle| or landmark
distance, however it's currently computed in cv_disagreement_full.csv) is
significantly correlated with actual angle error (Spearman rho=0.358,
p=0.0297). The existing gating rule sweeps a fixed threshold (e.g. 15 deg,
20 deg) and reports precision/recall at each cutoff. That's a fine
correlational result but makes no formal guarantee about a NEW patient.

WHAT CONFORMAL PREDICTION ADDS
--------------------------------
Split/Mondrian conformal prediction turns "these two are correlated" into:
"if a new patient's disagreement puts them in the LOW-disagreement group,
then with probability >= 1-alpha (over the randomness of a fresh
exchangeable patient), their true angle error is <= q_hat" — where q_hat is
computed from your existing leave-one-fold-out calibration data via a
single order-statistic formula. No modelling assumptions, no distribution
assumptions — only exchangeability of the calibration and test patients,
which leave-one-fold-out CV already gives you.

METHOD
------
1. Load the existing leak-free calibration data (patient_id, disagreement,
   angle_error) from cv_disagreement_full.csv — no new inference needed,
   this is purely a post-processing / re-analysis step on numbers you
   already have.
2. Partition calibration patients into groups by disagreement threshold(s)
   (Mondrian conformal: calibrate separately within each group so the
   guarantee holds group-conditionally, not just marginally).
3. Within each group of size n, the conformal quantile at level 1-alpha is
   the k-th order statistic of that group's angle errors, where
       k = ceil((n + 1) * (1 - alpha))   (clipped to n)
   This finite-sample correction (not simply the empirical (1-alpha)
   quantile) is what gives the exact coverage guarantee.
4. Output, for each candidate disagreement threshold and each requested
   alpha: the group sizes and the calibrated error bound per group.
5. Optionally run leave-one-out validation of the guarantee itself
   (empirical coverage check) as a sanity check that the guarantee holds
   on this exact dataset (it must, by construction, but this catches
   e.g. accidental non-exchangeability from sorting/leakage bugs).

USAGE
-----
    python scripts/evaluation/conformal_gating.py \
        --calibration_csv results/cv_disagreement_full.csv \
        --disagreement_col disagreement \
        --error_col angle_error_deg \
        --thresholds 15 20 \
        --alphas 0.1 0.2 \
        --out_csv results/conformal_gating_results.csv

If your actual column names in cv_disagreement_full.csv differ, just pass
--disagreement_col / --error_col accordingly — no code changes needed.
"""

import argparse
import math

import numpy as np
import pandas as pd


def conformal_quantile(errors: np.ndarray, alpha: float) -> float:
    """Finite-sample-correct (1-alpha) conformal quantile of a calibration
    set. errors: 1D array of nonconformity scores (here, angle error in
    degrees) for an exchangeable calibration group.

    Returns +inf if the group is too small to make any guarantee at this
    alpha (n < ceil(1/alpha) - 1), matching standard conformal convention
    rather than silently reporting a meaningless number.
    """
    n = len(errors)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")  # cannot guarantee this alpha with this few points
    sorted_errors = np.sort(errors)
    return float(sorted_errors[k - 1])


def mondrian_gate_table(df: pd.DataFrame, disagreement_col: str, error_col: str,
                         thresholds, alphas) -> pd.DataFrame:
    """For each threshold, split into low/high disagreement groups and
    compute the calibrated error bound at each alpha, per group."""
    rows = []
    for thresh in thresholds:
        low = df[df[disagreement_col] <= thresh][error_col].to_numpy()
        high = df[df[disagreement_col] > thresh][error_col].to_numpy()
        for group_name, errs in [("low_disagreement", low), ("high_disagreement", high)]:
            for alpha in alphas:
                q = conformal_quantile(errs, alpha)
                rows.append({
                    "disagreement_threshold": thresh,
                    "group": group_name,
                    "n": len(errs),
                    "alpha": alpha,
                    "coverage_target": 1 - alpha,
                    "calibrated_error_bound_deg": q,
                    "empirical_mean_error_deg": float(np.mean(errs)) if len(errs) else float("nan"),
                })
    return pd.DataFrame(rows)


def leave_one_out_coverage_check(df: pd.DataFrame, disagreement_col: str,
                                  error_col: str, thresh: float, alpha: float) -> float:
    """Sanity check: with n patients, leave each one out, recompute the
    conformal bound from the remaining n-1 (same group as the held-out
    point), and check whether the held-out point's actual error falls
    within the bound. Empirical coverage should be >= 1-alpha (it is
    guaranteed in expectation by exchangeability, but bugs like
    accidental sorting/leakage would show up here as a violation)."""
    covered = 0
    total = 0
    for i in range(len(df)):
        held_out = df.iloc[i]
        rest = df.drop(df.index[i])
        group = rest[rest[disagreement_col] <= thresh] if held_out[disagreement_col] <= thresh \
            else rest[rest[disagreement_col] > thresh]
        q = conformal_quantile(group[error_col].to_numpy(), alpha)
        if q == float("inf"):
            continue
        total += 1
        if held_out[error_col] <= q:
            covered += 1
    return covered / total if total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration_csv", required=True)
    ap.add_argument("--disagreement_col", default="disagreement")
    ap.add_argument("--error_col", default="angle_error_deg")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[15.0, 20.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--out_csv", default="results/conformal_gating_results.csv")
    ap.add_argument("--skip_loo_check", action="store_true",
                     help="Skip the leave-one-out coverage sanity check "
                          "(it's O(n^2) but n=37 so this is instant).")
    args = ap.parse_args()

    df = pd.read_csv(args.calibration_csv)
    for col in (args.disagreement_col, args.error_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {args.calibration_csv}. "
                f"Available columns: {list(df.columns)}"
            )

    table = mondrian_gate_table(df, args.disagreement_col, args.error_col,
                                  args.thresholds, args.alphas)
    table.to_csv(args.out_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {args.out_csv}")

    if not args.skip_loo_check:
        print("\nLeave-one-out empirical coverage check "
              "(should be >= 1-alpha for each row; small n means some "
              "noise around the target is expected):")
        for thresh in args.thresholds:
            for alpha in args.alphas:
                cov = leave_one_out_coverage_check(
                    df, args.disagreement_col, args.error_col, thresh, alpha
                )
                print(f"  threshold={thresh:>5.1f}  alpha={alpha:.2f}  "
                      f"target_coverage={1-alpha:.2f}  empirical_coverage={cov:.3f}")


if __name__ == "__main__":
    main()