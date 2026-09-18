"""
consolidate_predictions.py -- merges per-sample CV + test predictions into
one file: results/all_sample_predictions.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
RESULTS = ROOT / "results"


def infer_cohort(patient_id: str) -> str:
    if patient_id.startswith("CONTCT_R"):
        return "CONTCT_R"
    if patient_id.startswith("D"):
        return "D"
    if patient_id.startswith("K"):
        return "K"
    if patient_id.startswith("R"):
        return "R"
    return "UNKNOWN"


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    pred_c = df[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].to_numpy(dtype=float)
    gt_c = df[["gt_centre_D", "gt_centre_H", "gt_centre_W"]].to_numpy(dtype=float)
    df["centre_error_mm"] = np.linalg.norm(pred_c - gt_c, axis=1)

    pred_n = df[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].to_numpy(dtype=float)
    gt_n = df[["gt_normal_D", "gt_normal_H", "gt_normal_W"]].to_numpy(dtype=float)
    pred_n = pred_n / np.linalg.norm(pred_n, axis=1, keepdims=True)
    gt_n = gt_n / np.linalg.norm(gt_n, axis=1, keepdims=True)
    dot = np.abs(np.sum(pred_n * gt_n, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    df["angular_error_deg"] = np.degrees(np.arccos(dot))
    return df


def main():
    frames = []

    cv = pd.read_csv(RESULTS / "cv_results.csv")
    cv["split"] = "cv"
    frames.append(cv)

    test_ens = pd.read_csv(RESULTS / "ensembled_test_predictions.csv")
    test_ens["split"] = "test"
    test_ens["fold"] = "ensembled"
    frames.append(test_ens)

    test_pf = pd.read_csv(RESULTS / "per_fold_predictions.csv")
    test_pf["split"] = "test"
    frames.append(test_pf)

    df = pd.concat(frames, ignore_index=True, sort=False)

    manifest_path = ROOT / "dataset_manifest.csv"
    cohort_map = {}
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        # look for a cohort-like column: prefer literal "cohort", else "source"
        cohort_col = next((c for c in manifest.columns if "cohort" in c.lower()), None)
        if cohort_col is None:
            cohort_col = next((c for c in manifest.columns if c.lower() == "source"), None)
        id_col = next((c for c in manifest.columns if "patient" in c.lower() and "id" in c.lower()), None)
        if cohort_col and id_col:
            cohort_map = dict(zip(manifest[id_col], manifest[cohort_col]))
            print(f"Using manifest column '{cohort_col}' as cohort.")
        else:
            print(f"WARNING: no cohort/source column detected (columns: {manifest.columns.tolist()}) "
                  f"-- falling back to name inference.")
    else:
        print("WARNING: dataset_manifest.csv not found -- falling back to name inference.")

    # build as object dtype from the start to avoid float64 NaN-column trap
    df["cohort"] = pd.array([cohort_map.get(pid) for pid in df["patient_id"]], dtype="object")
    missing = df["cohort"].isna()
    if missing.any():
        df.loc[missing, "cohort"] = df.loc[missing, "patient_id"].apply(infer_cohort)
        n_unknown = (df["cohort"] == "UNKNOWN").sum()
        if n_unknown:
            print(f"WARNING: {n_unknown} rows UNKNOWN cohort -- check manually: "
                  f"{sorted(df.loc[df['cohort']=='UNKNOWN','patient_id'].unique())}")

    df = add_errors(df)

    cols = ["patient_id", "cohort", "split", "fold", "model",
            "pred_centre_D", "pred_centre_H", "pred_centre_W",
            "gt_centre_D", "gt_centre_H", "gt_centre_W", "centre_error_mm",
            "pred_normal_D", "pred_normal_H", "pred_normal_W",
            "gt_normal_D", "gt_normal_H", "gt_normal_W", "angular_error_deg"]
    df = df[cols].sort_values(["split", "model", "fold", "patient_id"]).reset_index(drop=True)

    out_path = RESULTS / "all_sample_predictions.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.groupby(["split", "model"]).size())
    print("\nCV mean errors per model (compare vs Table 2 top block):")
    print(df[df["split"] == "cv"].groupby("model")[["centre_error_mm", "angular_error_deg"]].agg(["mean", "std"]))
    print("\nTest (ensembled) mean errors per model (compare vs Table 2 bottom block, native rows):")
    print(df[(df["split"] == "test") & (df["fold"] == "ensembled")]
          .groupby("model")[["centre_error_mm", "angular_error_deg"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
