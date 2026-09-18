import numpy as np
import pandas as pd

RESULTS_DIR = "results"
df = pd.read_csv(f"{RESULTS_DIR}/per_fold_predictions.csv")

rows = []
for (pid, model), g in df.groupby(["patient_id", "model"]):
    normals = g[["pred_normal_D","pred_normal_H","pred_normal_W"]].values
    centres = g[["pred_centre_D","pred_centre_H","pred_centre_W"]].values

    # align sign of normals to fold 0 before measuring spread
    ref = normals[0]
    normals_aligned = np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])

    mean_normal = normals_aligned.mean(0)
    mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
    mean_centre = centres.mean(0)

    # per-fold angular deviation from the ensemble mean normal (degrees)
    angle_devs = [
        np.degrees(np.arccos(np.clip(np.abs(np.dot(n, mean_normal)), 0, 1)))
        for n in normals_aligned
    ]
    # per-fold centre deviation from ensemble mean centre (voxels)
    centre_devs = [np.linalg.norm(c - mean_centre) for c in centres]

    rows.append(dict(
        patient_id=pid, model=model,
        normal_uncertainty_deg=round(float(np.std(angle_devs)), 3),
        normal_spread_max_deg=round(float(np.max(angle_devs)), 3),
        centre_uncertainty_vx=round(float(np.std(centre_devs)), 3),
        centre_spread_max_vx=round(float(np.max(centre_devs)), 3),
    ))

out = pd.DataFrame(rows).sort_values(["patient_id","model"])
out.to_csv(f"{RESULTS_DIR}/uncertainty_summary.csv", index=False)
print(out.to_string(index=False))
print(f"\nSaved: {RESULTS_DIR}/uncertainty_summary.csv")
