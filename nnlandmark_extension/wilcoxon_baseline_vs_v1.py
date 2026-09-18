#!/usr/bin/env python3
"""
Wilcoxon signed-rank test: from-scratch baseline vs V1 PartialUnfreeze,
paired on the same fold-0 validation cases (same 4 patients, same task,
same ground truth) -- the comparison the supervisor actually asked for.

Paired at the per-landmark level (12 pairs: 4 cases x 3 landmarks each),
matching how MRE itself is computed (pooled per-landmark), giving more
test power than n=4 case-level pairing while still being defensible
(same case's 3 landmarks aren't fully independent, but this is a
standard, honestly-caveated compromise given how small this dataset is).
"""
from scipy.stats import wilcoxon
import numpy as np

# Baseline (from-scratch), confirmed via training log dates (28-29 July,
# no pretrained_weights loaded) -- fold-0 val, same 4 cases as V1.
baseline = {
    "CONTCT_R_14_FBA": [59.96502, 59.75603, 62.85904],
    "CONTCT_R_15_FBA": [38.89126, 25.56976, 30.72165],
    "CONTCT_R_16_FBA": [47.02448, 47.11016, 48.95023],
    "CONTCT_R_24_FBA_no_extended_seg": [79.62344, 81.92326, 85.33899],
}

# V1 PartialUnfreeze, fold-0 val (confirmed fresh run, 5-6 August).
partialunfreeze = {
    "CONTCT_R_14_FBA": [1.00563, 2.61872, 2.59576],
    "CONTCT_R_15_FBA": [17.91121, 16.07691, 1.81507],
    "CONTCT_R_16_FBA": [0.84961, 3.71475, 2.81955],
    "CONTCT_R_24_FBA_no_extended_seg": [2.25, 4.2162, 3.66592],
}

cases = ["CONTCT_R_14_FBA", "CONTCT_R_15_FBA", "CONTCT_R_16_FBA", "CONTCT_R_24_FBA_no_extended_seg"]
baseline_vals = np.array([v for c in cases for v in baseline[c]])
partial_vals = np.array([v for c in cases for v in partialunfreeze[c]])

print(f"Baseline:         mean={baseline_vals.mean():.2f}mm, n={len(baseline_vals)}")
print(f"V1 PartialUnfreeze: mean={partial_vals.mean():.2f}mm, n={len(partial_vals)}")

stat, p = wilcoxon(baseline_vals, partial_vals)
print(f"\nWilcoxon signed-rank test: statistic={stat:.3f}, p={p:.6f}")
if p < 0.05:
    print("Result: SIGNIFICANT -- V1 PartialUnfreeze's improvement over the from-scratch "
          "baseline is statistically significant.")
else:
    print("Result: not significant at alpha=0.05.")

print("\nCAVEAT: n=12 pairs from only 4 validation cases (3 landmarks each, not fully "
      "independent) -- report this as an exploratory/supporting statistic given the "
      "tiny sample, consistent with the sample-size caveats used throughout this project.")