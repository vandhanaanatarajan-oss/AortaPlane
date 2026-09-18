#!/usr/bin/env python3
import json
import numpy as np

RESULTS_ROOT = "nnLM_data/results/Dataset741_AnnulusCT"
TRAINERS = ["FrozenEncoder", "DifferentialLR", "PartialUnfreeze"]

for trainer in TRAINERS:
    all_case_mre = {}
    for fold in range(5):
        path = f"{RESULTS_ROOT}/nnLandmark_{trainer}__nnUNetPlansV1Match__3d_fullres/fold_{fold}/validation/summary_mm.json"
        with open(path) as f:
            d = json.load(f)
        for case_id, per_landmark in d["detailed_results"].items():
            all_case_mre[case_id] = float(np.mean(list(per_landmark.values())))

    values = list(all_case_mre.values())
    values_excl_r15 = [v for cid, v in all_case_mre.items() if cid != "CONTCT_R_15_FBA"]

    print(f"{trainer}:")
    print(f"  n={len(values)} cases, pooled CV MRE = {np.mean(values):.2f}mm")
    print(f"  excl. R_15 (n={len(values_excl_r15)}): {np.mean(values_excl_r15):.2f}mm")
    print()
