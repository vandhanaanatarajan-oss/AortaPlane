#!/usr/bin/env python3
"""
Iterative semi-supervised learning for AnnulusPlaneNet / nnLandmark, per
Dr Chen's definition (1 Aug 2026 meeting):

    train on labelled data -> generate pseudo-labels for unlabelled data
    -> retrain on labelled + pseudo-labelled -> repeat over several iterations

This REPLACES the earlier single-pass pseudo-labeling attempt, which only
did one train->relabel pass and was flagged as not matching the definition.

Design decisions (documented so you can defend them to Dr Chen / in the
write-up):
  - Base model: V1 encoder-transplant, 5-fold ensemble (best validated
    result so far, 6.24mm fold-0 val / 2.65mm test).
  - Confidence = 5-fold ensemble agreement. nnLandmark has no native
    per-prediction confidence output, so agreement across the 5 fold
    checkpoints is used as a proxy: tight agreement -> accept as
    pseudo-label, wide spread -> discard as unreliable.
  - Each iteration: (1) predict on remaining unlabelled pool with all 5
    folds, (2) accept confident cases, (3) merge into training data,
    (4) retrain all 5 folds from V1 weights on labelled+accumulated
    pseudo-labelled data, (5) evaluate on fold-0 val + true test set,
    (6) log results so you can see whether each iteration helps, plateaus,
    or hurts (report honestly either way).
  - Stops when no new cases are accepted, max_iterations is hit, or
    test-set error worsens for 2 consecutive iterations (early-stop
    guard against pseudo-label drift/confirmation bias).

USAGE (on Apocrita, inside nnlm_venv, from nnlandmark_comparison/):
    python semi_supervised_iterative.py --config config.yaml

This script SHELLS OUT to your existing nnLM_train / nnLM_predict /
nnLM_evaluate CLI commands rather than reimplementing training, so it
stays consistent with everything else you've already validated. It does
NOT submit SLURM jobs itself -- run it inside an interactive/salloc
session with GPU access, or wrap the `sbatch` call shown at the bottom
in your existing job-submission scripts (same pattern as your other
nnLM_train submissions).

Fill in the CONFIG block below (or pass a YAML file with --config) before
running. Every path/flag with a comment marked "CHECK" should be verified
against your actual folder layout before the first real run -- do a dry
run (--dry-run) first.
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CONFIG -- edit these to match your actual paths, or override via YAML
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Root of your nnLandmark comparison workspace
    workspace: Path = Path("/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison")

    # Labelled dataset to start from (your 18-patient V1-matched dataset)
    labelled_dataset_id: str = "Dataset741_AnnulusCT"          # CHECK
    plans_name: str = "nnUNetPlansV1Match"                     # CHECK: from your V1 encoder-transplant run

    # Pretrained weights to (re-)initialise each iteration's training from.
    # Using V1's weights fresh each iteration (not the previous iteration's
    # trained weights) keeps each iteration an independent, fair test of
    # "does more pseudo-labelled data help", rather than compounding drift.
    # Base dir containing fold_0..fold_4 subfolders, each with checkpoint_final.pth
    # (confirmed on Apocrita, 5 Aug 2026)
    pretrained_weights_base: Path = Path(
        "/gpfs/DERI-ecgai/001_CTA_Segmention/models/Aorta_v1.0/weights/"
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    # Unlabelled CTA pool (the 143 valid volumes used for contrastive pretraining)
    unlabelled_manifest: Path = Path("pretraining/manifest.json")  # confirmed on Apocrita, 5 Aug 2026

    # True held-out test set (never touched by pseudo-labeling)
    test_case_ids: list = field(default_factory=lambda: ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg"])

    # Fold-0 validation cases (for per-iteration tracking, same as your other runs)
    fold0_val_ids: list = field(default_factory=lambda: ["R_14", "R_15", "R_16", "R_24"])

    # Iteration control
    max_iterations: int = 4
    agreement_threshold_mm: float = 3.0   # max std-dev across 5-fold predictions to accept a pseudo-label. CHECK: tune after seeing iteration-1 spread distribution
    min_new_cases_to_continue: int = 3    # stop early if fewer than this many new cases pass the bar
    early_stop_patience: int = 2          # stop if test error worsens this many iterations in a row

    # Output
    output_root: Path = Path("semi_supervised_runs")
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, cfg: Config, cwd: Path = None):
    """Run a shell command, or just print it if dry_run."""
    printable = " ".join(str(c) for c in cmd)
    print(f"[RUN] {printable}")
    if cfg.dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed (exit {result.returncode}): {printable}")
    print(result.stdout)
    return result.stdout


def load_unlabelled_pool(cfg: Config) -> list:
    """
    Reads pretraining/manifest.json (real schema confirmed on Apocrita:
    a list of {"source": ..., "path": <raw .nrrd path>}).

    Returns list of dicts: [{"case_id": ..., "raw_path": ...}, ...]
    case_id is derived from the filename stem (e.g. "K5.nrrd" -> "K5"),
    matching the folder-name-as-case-ID pattern visible in the manifest
    paths (.../SEGA/K5/K5.nrrd).
    """
    with open(cfg.unlabelled_manifest) as f:
        manifest = json.load(f)

    pool = []
    for item in manifest:
        raw_path = Path(item["path"])
        case_id = raw_path.stem  # "K5.nrrd" -> "K5"
        # SEGA/CIS_UNet/Barts_CT_01 all reuse short IDs like K5/D15 across
        # sources, so disambiguate to avoid collisions when merging pools.
        case_id = f"{item['source']}_{case_id}"
        pool.append({"case_id": case_id, "raw_path": str(raw_path)})
    return pool


def stage_prediction_input(cases: list, cfg: Config, stage_dir: Path) -> Path:
    """
    nnLM_predict (like nnU-Net) expects an input FOLDER of correctly-named
    images, not a list of arbitrary raw paths. This builds that staging
    folder for the given batch of unlabelled cases, converting each
    raw .nrrd to .nii.gz with the "<case_id>_0000.nii.gz" naming nnU-Net
    expects for a single-channel input.

    CHECK: verify "<case_id>_0000.nii.gz" is really the naming nnLM_predict
    wants -- confirm against the exact -i folder layout you used for the
    R_08/R_27 true-test-set predictions (that call is already known-good).
    If nnLM_predict accepts raw .nrrd directly, this conversion step can
    be dropped and cases can be symlinked in as-is instead.
    """
    import SimpleITK as sitk  # already a dependency of your nnU-Net/nnLandmark env

    stage_dir.mkdir(parents=True, exist_ok=True)
    for item in cases:
        dst = stage_dir / f"{item['case_id']}_0000.nii.gz"
        if dst.exists():
            continue
        img = sitk.ReadImage(item["raw_path"])
        sitk.WriteImage(img, str(dst))
    return stage_dir


def predict_ensemble(cases: list, fold_checkpoints: list, cfg: Config, out_dir: Path):
    """
    Runs nnLM_predict with each of the 5 fold checkpoints on the given
    unlabelled cases, saving per-fold predicted landmark coordinates.

    Returns: dict {case_id: {fold_idx: np.array([x,y,z per landmark])}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}

    input_dir = stage_prediction_input(cases, cfg, out_dir / "staged_input")

    for fold_idx, ckpt in enumerate(fold_checkpoints):
        fold_out = out_dir / f"fold_{fold_idx}"
        cmd = [
            "nnLM_predict",
            "-i", str(input_dir),
            "-o", str(fold_out),
            "-d", cfg.labelled_dataset_id,
            "-p", cfg.plans_name,
            "-f", str(fold_idx),
            "-chk", str(ckpt),
        ]
        run(cmd, cfg)

        if cfg.dry_run:
            continue

        for item in cases:
            case_id = item["case_id"]
            pred_file = fold_out / f"{case_id}_landmarks.json"   # CHECK: actual nnLM_predict output naming
            if not pred_file.exists():
                continue
            with open(pred_file) as f:
                coords = np.array(json.load(f)["landmarks_mm"])   # CHECK: actual output schema
            predictions.setdefault(case_id, {})[fold_idx] = coords

    return predictions


def filter_by_agreement(predictions: dict, cfg: Config) -> dict:
    """
    Accept a case's pseudo-label only if the 5-fold predictions agree
    tightly (per-landmark std-dev across folds below threshold).
    Pseudo-label used = mean across folds for accepted cases.

    Returns: {case_id: {"landmarks_mm": np.array, "agreement_std_mm": float}}
    """
    accepted = {}
    for case_id, fold_preds in predictions.items():
        if len(fold_preds) < 5:
            continue  # incomplete ensemble, skip
        stacked = np.stack(list(fold_preds.values()))  # (5, n_landmarks, 3)
        std_per_landmark = stacked.std(axis=0)          # (n_landmarks, 3)
        mean_std_mm = std_per_landmark.mean()

        if mean_std_mm <= cfg.agreement_threshold_mm:
            accepted[case_id] = {
                "landmarks_mm": stacked.mean(axis=0).tolist(),
                "agreement_std_mm": float(mean_std_mm),
            }
    return accepted


def merge_into_dataset(accepted: dict, unlabelled_pool_by_id: dict, iteration: int, cfg: Config) -> str:
    """
    Writes accepted pseudo-labels into a new dataset (e.g.
    Dataset743_AnnulusSemiSup_iterN), combining the original 18 labelled
    cases with all pseudo-labelled cases accepted so far (cumulative
    across iterations, not just this iteration's new ones).

    Adapted from your convert_annulus.py -- same imagesTr/labelsTr/
    dataset.json structure, same 3x3x3-cube landmark-burning scheme
    (half=1), so the new dataset is structurally identical to Dataset741.

    CHECK (coordinate handling): convert_annulus.py's source landmarks
    (processed_landmarks.json) are ALREADY in voxel space -- the RAS/LPS
    world-to-voxel conversion happened upstream, earlier in your
    pipeline, not inside convert_annulus.py itself. Pseudo-label
    predictions from nnLM_predict, by contrast, are assumed to come back
    in PHYSICAL mm space (matching the "landmarks_mm" key predict_ensemble
    already expects). This function converts mm -> voxel via SimpleITK's
    own TransformPhysicalPointToIndex, which is correct IF nnLM_predict's
    output is in the same physical (LPS) space as the image object it
    predicted on. Given your project's history with exactly this kind of
    RAS/LPS mismatch, sanity-check the first iteration's converted voxel
    coordinates by eye (e.g. plot 2-3 pseudo-labelled cases the same way
    you visually verified the original preprocessing) before trusting
    iteration 2 onward.

    CHECK (preprocessing): this writes RAW data only (imagesTr/labelsTr/
    dataset.json), matching what convert_annulus.py does for Dataset741.
    Dataset741 was then separately preprocessed (nnLM_plan_experiment /
    nnLM_preprocess, run manually) before nnLM_train could use it -- this
    function shells out to the same preprocessing command below, but
    verify the exact flags against whatever you actually ran for
    Dataset741/742 (visible in your shell history / the .sh scripts in
    this folder), since I'm inferring the command name and flags rather
    than reading them directly.

    Returns the new dataset_id.
    """
    new_dataset_id = f"Dataset74{2 + iteration}_AnnulusSemiSup_iter{iteration}"  # CHECK: pick real free dataset IDs
    print(f"[MERGE] Writing {len(accepted)} pseudo-labelled cases into {new_dataset_id}")
    if cfg.dry_run:
        return new_dataset_id

    import shutil as _shutil
    import SimpleITK as sitk

    nnlm_data_raw = cfg.workspace / "nnLM_data" / "raw"          # CHECK: matches convert_annulus.py's OUT_RAW pattern
    src_raw = nnlm_data_raw / cfg.labelled_dataset_id            # Dataset741_AnnulusCT's existing raw folder
    out_raw = nnlm_data_raw / new_dataset_id
    images_tr = out_raw / "imagesTr"
    labels_tr = out_raw / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    name_to_label = {"annulus_1": 1, "annulus_2": 2, "annulus_3": 3}
    half = 1  # 3x3x3 cube, same as convert_annulus.py

    # 1. Copy the original 18 labelled cases in as-is (images + label maps
    #    already burned correctly by convert_annulus.py -- don't regenerate,
    #    just reuse so nothing about the trusted ground truth changes).
    all_landmarks_voxel = {}
    spacing_map = {}
    with open(src_raw / "all_landmarks_voxel.json") as f:
        all_landmarks_voxel.update(json.load(f))
    with open(src_raw / "spacing.json") as f:
        spacing_map.update(json.load(f))

    n_copied = 0
    for img_file in (src_raw / "imagesTr").glob("*_0000.nii.gz"):
        _shutil.copy2(img_file, images_tr / img_file.name)
        n_copied += 1
    for lbl_file in (src_raw / "labelsTr").glob("*.nii.gz"):
        _shutil.copy2(lbl_file, labels_tr / lbl_file.name)

    # 2. Write each accepted pseudo-labelled case in the same format.
    n_pseudo = 0
    for case_id, info in accepted.items():
        raw_path = unlabelled_pool_by_id[case_id]["raw_path"]
        img = sitk.ReadImage(raw_path)
        size = img.GetSize()

        out_img_path = images_tr / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(img, str(out_img_path))

        label_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
        case_landmarks_voxel = {}
        landmarks_mm = info["landmarks_mm"]  # [[x,y,z], [x,y,z], [x,y,z]] in physical mm
        for i, point_mm in enumerate(landmarks_mm, start=1):
            xi, yi, zi = img.TransformPhysicalPointToIndex(tuple(point_mm))
            case_landmarks_voxel[f"annulus_{i}"] = [xi, yi, zi]
            x0, x1 = max(0, xi - half), min(size[0], xi + half + 1)
            y0, y1 = max(0, yi - half), min(size[1], yi + half + 1)
            z0, z1 = max(0, zi - half), min(size[2], zi + half + 1)
            label_arr[z0:z1, y0:y1, x0:x1] = i

        label_img = sitk.GetImageFromArray(label_arr)
        label_img.CopyInformation(img)
        sitk.WriteImage(label_img, str(labels_tr / f"{case_id}.nii.gz"))

        all_landmarks_voxel[case_id] = case_landmarks_voxel
        spacing_map[case_id] = {"image_spacing": list(img.GetSpacing()), "annotation_spacing": None}
        n_pseudo += 1

    total_cases = n_copied + n_pseudo
    with open(out_raw / "all_landmarks_voxel.json", "w") as f:
        json.dump(all_landmarks_voxel, f, indent=2)
    with open(out_raw / "spacing.json", "w") as f:
        json.dump(spacing_map, f, indent=2)
    with open(out_raw / "name_to_label.json", "w") as f:
        json.dump(name_to_label, f, indent=2)
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, **name_to_label},
        "numTraining": total_cases,
        "file_ending": ".nii.gz",
        "name": new_dataset_id,
    }
    with open(out_raw / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[MERGE] {new_dataset_id}: {n_copied} original + {n_pseudo} pseudo-labelled = {total_cases} total")

    # 3. Preprocess the new dataset so nnLM_train can use it.
    # CHECK: confirm this matches your actual preprocessing command --
    # inferred from your V1-matched-plans workflow (nnLM_plan_experiment
    # producing nnUNetPlansV1Match), not read directly from your scripts.
    preprocess_cmd = [
        "nnLM_plan_and_preprocess",
        "-d", new_dataset_id,
        "-overwrite_plans_name", cfg.plans_name,
        "-c", "3d_fullres",
    ]
    run(preprocess_cmd, cfg)

    return new_dataset_id


def retrain_all_folds(dataset_id: str, cfg: Config, out_tag: str):
    for fold_idx in range(5):
        # Each fold retrains from ITS OWN matching V1 fold checkpoint, not
        # a single shared one -- matches how your 3 currently-running V1
        # fine-tuning-strategy jobs each load their own fold's weights.
        fold_weights = cfg.pretrained_weights_base / f"fold_{fold_idx}" / "checkpoint_final.pth"
        cmd = [
            "nnLM_train",
            dataset_id,
            "-p", cfg.plans_name,
            str(fold_idx),
            "-pretrained_weights", str(fold_weights),
        ]
        run(cmd, cfg)


def evaluate(dataset_id: str, cfg: Config, tag: str) -> dict:
    """Evaluate on fold-0 val ids and true test ids, return summary dict."""
    results = {}
    for group_name, ids in [("fold0_val", cfg.fold0_val_ids), ("true_test", cfg.test_case_ids)]:
        cmd = ["nnLM_evaluate", "-d", dataset_id, "-p", cfg.plans_name, "--case-ids", *ids]
        out = run(cmd, cfg)
        results[group_name] = out  # CHECK: parse actual summary_mm.json instead of raw stdout
    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(cfg: Config):
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    log = []
    accepted_so_far = {}
    consecutive_worse = 0
    prev_test_error = None
    current_dataset_id = cfg.labelled_dataset_id

    unlabelled_pool = load_unlabelled_pool(cfg)
    print(f"Starting semi-supervised loop: {len(unlabelled_pool)} unlabelled candidates, "
          f"max {cfg.max_iterations} iterations")

    for iteration in range(1, cfg.max_iterations + 1):
        print(f"\n=== ITERATION {iteration} ===")

        # 1. Predict on remaining (not-yet-accepted) unlabelled cases with current 5-fold ensemble
        remaining = [c for c in unlabelled_pool if c["case_id"] not in accepted_so_far]
        fold_checkpoints = [
            Path(cfg.workspace) / "checkpoints" / f"semisup_iter{iteration-1}_fold{f}.pth"  # CHECK naming
            if iteration > 1 else cfg.pretrained_weights_base / f"fold_{f}" / "checkpoint_final.pth"
            for f in range(5)
        ]
        preds = predict_ensemble(remaining, fold_checkpoints, cfg, cfg.output_root / f"iter{iteration}_preds")

        # 2. Filter by ensemble agreement
        newly_accepted = filter_by_agreement(preds, cfg)
        print(f"Accepted {len(newly_accepted)} / {len(remaining)} candidates this iteration")

        if len(newly_accepted) < cfg.min_new_cases_to_continue:
            print("Fewer than min_new_cases_to_continue accepted -- stopping.")
            break

        accepted_so_far.update(newly_accepted)

        # 3. Merge into training set (cumulative)
        pool_by_id = {c["case_id"]: c for c in unlabelled_pool}
        current_dataset_id = merge_into_dataset(accepted_so_far, pool_by_id, iteration, cfg)

        # 4. Retrain all 5 folds from V1 weights on labelled + pseudo-labelled
        retrain_all_folds(current_dataset_id, cfg, out_tag=f"semisup_iter{iteration}")

        # 5. Evaluate honestly on fold-0 val + true test set
        results = evaluate(current_dataset_id, cfg, tag=f"iter{iteration}")
        log.append({
            "iteration": iteration,
            "n_pseudo_labelled_cumulative": len(accepted_so_far),
            "results": results,
        })

        # 6. Early-stop guard: if true test error gets worse for `early_stop_patience`
        # iterations in a row, stop and report the best iteration, not the last one.
        # CHECK: extract the actual scalar MRE from `results["true_test"]` once
        # evaluate() parses summary_mm.json properly.
        # current_test_error = results["true_test"]["mre_micro_mm"]
        # if prev_test_error is not None and current_test_error > prev_test_error:
        #     consecutive_worse += 1
        #     if consecutive_worse >= cfg.early_stop_patience:
        #         print("Test error worsened for consecutive iterations -- stopping, "
        #               "report iteration", iteration - consecutive_worse, "as best.")
        #         break
        # else:
        #     consecutive_worse = 0
        # prev_test_error = current_test_error

    with open(cfg.output_root / "semi_supervised_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Log written to {cfg.output_root / 'semi_supervised_log.json'}")
    print("Report ALL iterations' results in the write-up, not just the best one -- "
          "that's the honest-reporting standard the rest of this project has followed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    cfg = Config()
    cfg.dry_run = args.dry_run
    main(cfg)