import numpy as np
import pandas as pd

df = pd.read_csv("results/per_fold_predictions.csv")
pid = "CONTCT_R_08_FBA"

m2 = df[(df["patient_id"] == pid) & (df["model"] == "M2")]
if len(m2) == 0:
    raise SystemExit(f"No M2 rows found for {pid}")

gt_n = m2.iloc[0][["gt_normal_D", "gt_normal_H", "gt_normal_W"]].to_numpy(dtype=float)
gt_c = m2.iloc[0][["gt_centre_D", "gt_centre_H", "gt_centre_W"]].to_numpy(dtype=float)

normals = m2[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].to_numpy(dtype=float)
centres = m2[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].to_numpy(dtype=float)

# Sign-align each fold's normal to GT before averaging (plane normals are
# ambiguous up to sign — same reasoning as the loss function, and the same
# fix needed for the D1/R13 arrows).
aligned = np.array([
    n if np.dot(n, gt_n) >= 0 else -n
    for n in normals
])

mean_normal = aligned.mean(axis=0)
mean_normal /= np.linalg.norm(mean_normal) + 1e-8
mean_centre = centres.mean(axis=0)

print(f"{pid} — M2 ensembled across {len(m2)} folds")
print(f"  pred_normal: {mean_normal}")
print(f"  pred_centre: {mean_centre}")
print(f"  gt_normal:   {gt_n}")
print(f"  gt_centre:   {gt_c}")

out = pd.DataFrame([{
    "patient_id": pid, "model": "M2",
    "pred_normal_D": mean_normal[0], "pred_normal_H": mean_normal[1], "pred_normal_W": mean_normal[2],
    "pred_centre_D": mean_centre[0], "pred_centre_H": mean_centre[1], "pred_centre_W": mean_centre[2],
    "gt_normal_D": gt_n[0], "gt_normal_H": gt_n[1], "gt_normal_W": gt_n[2],
    "gt_centre_D": gt_c[0], "gt_centre_H": gt_c[1], "gt_centre_W": gt_c[2],
}])
out.to_csv("results/fig1_r08_m2_ensembled.csv", index=False)
print("\nSaved: results/fig1_r08_m2_ensembled.csv")