import numpy as np
import pandas as pd

SEG_PATHS = {
    "D1": "/data/DERI-ecgai/__users/Vandhanaa/processed_output/D1/D1_seg_resized_64.npy",
    "CONTCT_R_13_FBA": "/data/DERI-ecgai/__users/Vandhanaa/processed_output/CONTCT_R_13_FBA/CONTCT_R_13_FBA_seg_resized_64.npy",
    "CONTCT_R_08_FBA": "/data/DERI-ecgai/__users/Vandhanaa/processed_output/CONTCT_R_08_FBA/CONTCT_R_08_FBA_seg_resized_64.npy",
}

cv = pd.read_csv("results/fig1_cv_patient_vectors.csv")  # D1, R13 — final_coordconv + M2
r08_m2 = pd.read_csv("results/fig1_r08_m2_ensembled.csv")  # R08 — M2 only
coordconv_test = pd.read_csv("results_coordconv/ensemble_results_coordconv_vs_baseline.csv")

rows = []

# D1 + R13: already have both models
for pid in ["D1", "CONTCT_R_13_FBA"]:
    sub = cv[cv["patient_id"] == pid]
    gt_row = sub.iloc[0]
    gt_normal = gt_row[["gt_normal_D", "gt_normal_H", "gt_normal_W"]].to_numpy(dtype=float)
    gt_centre = gt_row[["gt_centre_D", "gt_centre_H", "gt_centre_W"]].to_numpy(dtype=float)

    for model_tag, out_name in [("final_coordconv", "final"), ("M2", "m2")]:
        r = sub[sub["model"] == model_tag].iloc[0]
        normal = r[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].to_numpy(dtype=float)
        centre = r[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].to_numpy(dtype=float)
        if np.dot(normal, gt_normal) < 0:
            normal = -normal
        rows.append(dict(patient_id=pid, source="final" if out_name == "final" else "m2",
                          normal_D=normal[0], normal_H=normal[1], normal_W=normal[2],
                          centre_D=centre[0], centre_H=centre[1], centre_W=centre[2]))
    rows.append(dict(patient_id=pid, source="gt",
                      normal_D=gt_normal[0], normal_H=gt_normal[1], normal_W=gt_normal[2],
                      centre_D=gt_centre[0], centre_H=gt_centre[1], centre_W=gt_centre[2]))

# R_08: final model + GT from coordconv_test, M2 from the separate ensemble
r = coordconv_test[coordconv_test["patient_id"] == "CONTCT_R_08_FBA"].iloc[0]
gt_normal = r[["gt_normal_D", "gt_normal_H", "gt_normal_W"]].to_numpy(dtype=float)
gt_centre = r[["gt_centre_D", "gt_centre_H", "gt_centre_W"]].to_numpy(dtype=float)
final_normal = r[["coordconv_normal_D", "coordconv_normal_H", "coordconv_normal_W"]].to_numpy(dtype=float)
final_centre = r[["coordconv_centre_D", "coordconv_centre_H", "coordconv_centre_W"]].to_numpy(dtype=float)
if np.dot(final_normal, gt_normal) < 0:
    final_normal = -final_normal

m2r = r08_m2.iloc[0]
m2_normal = m2r[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].to_numpy(dtype=float)
m2_centre = m2r[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].to_numpy(dtype=float)
if np.dot(m2_normal, gt_normal) < 0:
    m2_normal = -m2_normal

rows.append(dict(patient_id="CONTCT_R_08_FBA", source="gt",
                  normal_D=gt_normal[0], normal_H=gt_normal[1], normal_W=gt_normal[2],
                  centre_D=gt_centre[0], centre_H=gt_centre[1], centre_W=gt_centre[2]))
rows.append(dict(patient_id="CONTCT_R_08_FBA", source="final",
                  normal_D=final_normal[0], normal_H=final_normal[1], normal_W=final_normal[2],
                  centre_D=final_centre[0], centre_H=final_centre[1], centre_W=final_centre[2]))
rows.append(dict(patient_id="CONTCT_R_08_FBA", source="m2",
                  normal_D=m2_normal[0], normal_H=m2_normal[1], normal_W=m2_normal[2],
                  centre_D=m2_centre[0], centre_H=m2_centre[1], centre_W=m2_centre[2]))

df = pd.DataFrame(rows)
df["seg_path"] = df["patient_id"].map(SEG_PATHS)
df.to_csv("results/fig1_merged.csv", index=False)
print(df.to_string())
print("\nSaved: results/fig1_merged.csv")