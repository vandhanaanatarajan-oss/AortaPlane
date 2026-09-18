"""
parse_new_landmarks.py

Parses Slicer .mrk.json fiducial files (RC/NC/LC) from the new
/gpfs/DERI-ecgai/001_CTA_Segmention/Clipping/landmarks/annulus folder
into the same RC/NC/LC + derived centre/normal representation your
existing preprocess.py pipeline expects.

IMPORTANT: these positions are in world (LPS) mm space already
(coordinateSystem: "LPS", coordinateUnits: "mm" in the JSON header) --
unlike the historical bug #1 in your original 42-patient CONTCT_R data
(which stored image-voxel coordinates mislabelled as mm), these new files
say explicitly they're already physical mm coordinates. Don't assume this
means no conversion is needed though -- you still need the matching raw
CT volume to (a) confirm this patient's orientation/spacing and (b) run
it through your existing resample -> crop -> resize pipeline to get
voxel-space centre/normal targets for dataset_manifest.csv. This script
only gets you to "world-space ground truth landmarks" -- the raw CT is
still required for the rest.

Usage:
    python parse_new_landmarks.py --landmarks_dir /path/to/landmarks/annulus \
                                   --out_csv new_landmarks_world_mm.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def extract_patient_id(filename: str) -> str:
    """
    'CONTCT_R_08_Pre_CT_20191129_FBA...json' -> 'CONTCT_R_08'
    Adjust this if your actual filenames need a different split point --
    check a few real filenames first, since the screenshot only showed
    truncated names.
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    # CONTCT_R_08_Pre_CT_... -> take first 3 underscore-separated tokens
    return "_".join(parts[:3])


def parse_one_file(filepath: str) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)

    control_points = data["markups"][0]["controlPoints"]
    coords = {}
    for cp in control_points:
        label = cp["label"].strip().upper()
        if label in ("RC", "NC", "LC"):
            coords[label] = np.array(cp["position"], dtype=float)

    missing = {"RC", "NC", "LC"} - coords.keys()
    if missing:
        raise ValueError(f"{filepath}: missing landmark(s) {missing}")

    centre = (coords["RC"] + coords["NC"] + coords["LC"]) / 3.0
    normal = np.cross(coords["NC"] - coords["RC"], coords["LC"] - coords["RC"])
    normal = normal / np.linalg.norm(normal)

    return {
        "patient_id": extract_patient_id(filepath),
        "rc_x_mm": coords["RC"][0], "rc_y_mm": coords["RC"][1], "rc_z_mm": coords["RC"][2],
        "nc_x_mm": coords["NC"][0], "nc_y_mm": coords["NC"][1], "nc_z_mm": coords["NC"][2],
        "lc_x_mm": coords["LC"][0], "lc_y_mm": coords["LC"][1], "lc_z_mm": coords["LC"][2],
        "centre_x_mm": centre[0], "centre_y_mm": centre[1], "centre_z_mm": centre[2],
        "normal_x": normal[0], "normal_y": normal[1], "normal_z": normal[2],
        "source_file": os.path.basename(filepath),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_dir", required=True,
                         help="Path to the 'annulus' folder, e.g. "
                              "/gpfs/DERI-ecgai/001_CTA_Segmention/Clipping/landmarks/annulus")
    parser.add_argument("--out_csv", default="new_landmarks_world_mm.csv")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.landmarks_dir, "*.json")))
    print(f"Found {len(json_files)} landmark files in {args.landmarks_dir}")

    rows = []
    errors = []
    for fp in json_files:
        try:
            rows.append(parse_one_file(fp))
        except Exception as e:
            errors.append((fp, str(e)))

    if errors:
        print(f"\n{len(errors)} file(s) failed to parse:")
        for fp, err in errors:
            print(f"  {fp}: {err}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} patients to {args.out_csv}")
    print("\nPatient IDs found (check these against your existing 42-patient "
          "manifest for overlaps -- see note above about CONTCT_R Bug C):")
    print(df["patient_id"].tolist())


if __name__ == "__main__":
    main()