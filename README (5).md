# AortaPlane

### Deep Learning-Based Aortic Annulus Plane Detection from 3D Cardiac CT for TAVI Planning

##### If you find this project useful, please give it a star 🌟

**Vandhanaa**¹ — supervised by **Dr Ahmed Sayed**¹

¹ DERI/ECGAI Lab, Queen Mary University of London

![AortaPlane end-to-end pipeline: CT input, 3D ResNet-CBAM training, direct 6-DoF regression, and conformal QC](pics/aortaplane-ct-training-pipeline.gif)

---

## Introduction

Accurate localisation of the aortic annulus plane in 3D cardiac CT is a critical
step in planning Transcatheter Aortic Valve Implantation (TAVI), but is
traditionally a manual, time-consuming process. This project, **AortaPlane**,
compares two deep-learning approaches to automating annulus plane detection:

- **M1 — AortaPlane (AnnulusPlaneNet):** direct 6D regression (plane centre +
  normal) via a 3D ResNet18 backbone with CBAM attention.
- **M2 — Baseline (AnnulusLandmarkNet):** landmark heatmap regression over the
  right/non/left coronary cusps (RC/NC/LC), with the plane derived
  geometrically from the predicted landmarks.

Both models are evaluated with 5-fold cross-validation on a 42-patient,
four-cohort dataset (Dongyang / KiTS / Rider / CONTCT\_R) with 5 patients held
out as a locked test set, and are additionally benchmarked against a
fine-tuned nnU-Net-derived landmark baseline (**nnLandmark**) under three
fine-tuning strategies (frozen encoder, partial unfreeze, differential
learning rate).

**Key result:** M1 (direct regression) achieves a substantially lower centre
error than M2 (~9.36mm vs ~16.64mm MAE) while orientation accuracy between the
two is statistically indistinguishable. A cross-model disagreement analysis
further shows M1 and M2 fail in qualitatively different ways — M1 is prone to
confident, low-disagreement errors on outlier anatomy, while M2 is more
volatile but averages closer to correct — a finding relevant to uncertainty
estimation for clinical deployment.

This repository accompanies an MSc dissertation and a related submission to
**MICAD 2026**, *"Direct Regression versus Landmark-Based Aortic Annulus
Plane Detection from 3D Cardiac CT."*

---

## Directory structure

```
src/                        Core model, dataset, and loss definitions
    model.py                  Native AortaPlane (AnnulusPlaneNet) and
                               Baseline (AnnulusLandmarkNet) architectures
    dataset.py                 Dataset loading / preprocessing
    loss.py                    PlaneLoss, LandmarkPlaneLoss, error metrics
    coordconv3d.py              CoordConv3D extension (Section III-D)

train_m1_variants.py          Trains AortaPlane and its architecture variants
train_m2_variants.py          Trains Baseline and its architecture variants
m1_unet.py, m2_resnet3d_noskip.py,
m2_densenet3d.py, densenet3d_encoder.py
                               Architecture-variant model definitions (root)

arch_variants/                 Remaining architecture-variant model definitions
    m1_densenet3d.py
    m1_nnlandmark.py            AortaPlane + nnLandmark-encoder transplant
    m2_nnlandmark.py            Baseline + nnLandmark-encoder transplant

evaluation/
    eval_cv.py                    5-fold CV evaluation (native models)
    compute_full_grid_5fold_true_test.py
                                    Full 8-row architecture-grid evaluation
                                    on the true n=5 held-out test set
    compute_variant_flops.py       FLOPs/parameter profiling for all 8
                                    architecture-grid rows (see note below
                                    on nnLandmark-row parameter accounting)
    compute_nnlandmark_native_flops.py
                                    FLOPs profiling for nnLandmark's own
                                    standalone backbone (KNOWN ISSUE — see
                                    "Known issues" below; do not treat its
                                    output as final)
    [gating sweep / disagreement-correlation / MAE scripts]

nnlandmark_comparison/          Separate codebase adapting MIC-DKFZ's
                                 nnU-Net-derived nnLandmark framework for
                                 direct comparison against AortaPlane/Baseline
    convert_annulus.py             Converts project data into nnLandmark's
                                    expected dataset format
    semi_supervised_iterative.py   Iterative pseudo-labelling experiment
                                    (reported as a negative result)
    build_matched_splits.py        Builds nnLandmark's cross-validation splits
                                    to exactly match the main project's own
                                    fold_config.json assignments
    arch_variants/                  nnLandmark-transplant model definitions
                                    (mirrors root arch_variants/ above)
    training/nnUNetTrainer/variants/finetune_strategies.py
                                    The three fine-tuning-strategy trainer
                                    subclasses developed for this project
                                    (nnLandmark_FrozenEncoder,
                                    nnLandmark_PartialUnfreeze,
                                    nnLandmark_DifferentialLR), each
                                    registered with nnU-Net's own CLI via its
                                    `-tr <ClassName>` flag. This is a small,
                                    self-contained file extracted from a much
                                    larger third-party nnU-Net/nnLandmark
                                    installation — see "Third-party code not
                                    included" below.

fold_config.json                5-fold CV patient split definition (37
                                 train/val patients + 5 held-out test patients)
dataset_manifest.csv            Patient/cohort manifest

results/                        Small result artefacts (CSV/JSON summaries,
                                 aggregate tables). No per-patient identifiable
                                 imaging data is included here.
    all_sample_predictions.csv     Detailed per-sample predictions for every
                                    patient in both the 5-fold CV set (n=37)
                                    and the held-out test set (n=5), across
                                    all three native-encoder models (M1
                                    AortaPlane, M2 Baseline, CoordConv).
                                    Columns: patient_id, cohort, split
                                    (cv/test), fold (0-4 for CV; ensembled
                                    or 0-4 for test), model, predicted and
                                    ground-truth centre (D/H/W, voxel-mm),
                                    centre_error_mm, predicted and
                                    ground-truth normal (D/H/W, unit vector),
                                    and angular_error_deg. For the test split,
                                    fold="ensembled" gives the final reported
                                    prediction (mean across the 5 fold
                                    checkpoints, matching Table II's bottom
                                    block); fold=0-4 gives the individual
                                    pre-ensemble checkpoint predictions.
                                    Built by consolidate_predictions.py
                                    (project root) from cv_results.csv,
                                    ensembled_test_predictions.csv, and
                                    per_fold_predictions.csv; sanity-checked
                                    against Table II before packaging.

checkpoints/                    Trained model weights for the two native
                                 models only (variant/nnLandmark-transplant
                                 checkpoints are not included — several are
                                 far larger, e.g. ~88M-parameter nnLandmark
                                 rows, and would not fit a reasonable
                                 submission size)
    M1_fold0_best_weights_only.pth  Native AortaPlane, fold 0. Stripped of
                                    optimizer/scheduler state (only needed to
                                    resume training) — reduces this file from
                                    ~42MB to ~14MB.
    M2_fold0_best.pth               Native Baseline, fold 0. Included as-is —
                                    already saved without optimizer state
                                    (~23MB).

slurm_scripts/                  Launch scripts for the SLURM job scheduler on
                                 QMUL's Apocrita HPC, showing the exact
                                 commands/arguments used for each reported
                                 result. Only scripts confirmed to produce a
                                 result reported in the dissertation are
                                 included — see "SLURM scripts not included"
                                 below for what was excluded and why.

logs/
    native/                       stdout/stderr from the native AortaPlane and
                                   Baseline fold-0 training runs
    nnlandmark_finetuning/        stdout/stderr from the three fine-tuning-
                                   strategy runs (FrozenEncoder,
                                   PartialUnfreeze, DifferentialLR), including
                                   their per-fold continuation runs

environment/
    requirements_main.txt          Full pip freeze from the main project venv
    requirements_nnlandmark.txt    Full pip freeze from the nnLandmark
                                    comparison venv (includes torch, the
                                    dynamic_network_architectures package,
                                    and the editable nnLandmark install)
```

---

## Environment setup

Two separate environments were used and are needed to reproduce different
parts of the project. Package versions for both are pinned exactly in
`environment/requirements_main.txt` and `environment/requirements_nnlandmark.txt`.

**Main project environment** (`src/`, `train_m*_variants.py`, `arch_variants/`,
evaluation scripts):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r environment/requirements_main.txt
```

**nnLandmark comparison environment** (`nnlandmark_comparison/`):

```bash
module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
python3 -m venv nnlm_venv          # create inside nnlandmark_comparison/
source nnlm_venv/bin/activate
pip install -r environment/requirements_nnlandmark.txt
pip install -e nnlandmark_comparison/nnLandmark  # editable install of the
                                                    # nnU-Net-derived package
```

This part of the project was run under Python 3.12.1 with CUDA 12.6.2 on
QMUL's Apocrita HPC cluster, with the module-load sequence above required
before activating the venv (Apocrita's module system, not a general
requirement of the code itself). GPU access via SLURM is effectively
required for any of the 3D volume training to run in reasonable time.

---

## Data availability

This project used two categories of data:

- **Barts Health NHS Trust cohort (CONTCT\_R, n=20).** Retrospective,
  de-identified TAVI case data, used under DERI's data-sharing agreement with
  Barts Health NHS Trust. This data cannot be redistributed as part of a
  student submission; it is not included in this archive.
- **AVT dataset cohorts (D/K/R, n=22).** Publicly available multicentre CTA
  data (Radl et al., 2022); this project's own manual landmark annotations on
  a subset of these scans are also not redistributed here, but the underlying
  public imaging data can be obtained from the AVT dataset's own public
  release.

---

## Usage

An executable build is not provided, for two reasons:

1. **Hardware/scheduler dependency.** Training and full-volume inference
   require a CUDA GPU and were run via SLURM job arrays on QMUL's Apocrita HPC
   cluster (5-fold cross-validation across 8+ model configurations, each
   requiring several GPU-hours). This cannot be packaged as a standalone
   executable that would run on examiner hardware.
2. **Data dependency.** The code requires the source CT volumes, segmentation
   masks, and landmark annotation files as input (see "Data availability"
   above), which are not included in this submission.

**Steps to run the code**, given access to equivalent data and a
CUDA-capable machine:

1. Set up the environment(s) as above.
2. Obtain/place patient data under the path expected by `dataset.py` /
   `convert_annulus.py` (see comments in those files for the expected
   directory layout).
3. Run `train_m1_variants.py` / `train_m2_variants.py` with the desired
   `--fold` argument (0–4) to train the native models or a named variant, or
   use the SLURM scripts under `slurm_scripts/` directly (these show the
   exact arguments used for the reported results).
4. Alternatively, load the provided checkpoints directly for inference
   without retraining: `checkpoints/M1_fold0_best_weights_only.pth` /
   `checkpoints/M2_fold0_best.pth`, using
   `AnnulusPlaneNet(dropout_p=0.3, use_se=False, use_cbam=True)` /
   `AnnulusLandmarkNet(dropout_p=0.3, use_cbam=True)` from `src/model.py`
   respectively.
5. Run `evaluation/eval_cv.py` or
   `evaluation/compute_full_grid_5fold_true_test.py` to reproduce the
   evaluation tables reported in the dissertation.
6. For the nnLandmark comparison: run `convert_annulus.py` to build the
   nnLandmark-format dataset, then use nnLandmark's own CLI
   (`nnLM_extract_fingerprint`, `nnLM_plan_experiment`, `nnLM_train`) with the
   custom trainer classes provided under `nnlandmark_comparison/training/`,
   or the SLURM scripts under `slurm_scripts/` directly.

---

## SLURM scripts not included

`nnlandmark_comparison/` contains a number of additional launch scripts
beyond those in `slurm_scripts/` (e.g. `train_dataset741*.sh` variants,
`retrain_v1_matched.sh`, `submit_semi_supervised_live.sh`). These were
earlier, superseded, or exploratory runs (confirmed via file timestamps and
by checking which scripts actually launch the code paths behind reported
results) and are not included, to keep the submission focused on the scripts
that actually produced the numbers in the dissertation.

## Third-party code not included

This project builds on MIC-DKFZ's nnU-Net framework (via a locally installed
`nnLandmark` package derived from it) for the nnLandmark comparison arm of
the project. Only the code specific to this project is included in this
archive — namely `finetune_strategies.py` (the three fine-tuning-strategy
trainer subclasses) and the dataset-conversion/split-building scripts listed
above. The underlying nnU-Net/nnLandmark library itself (stock trainer
classes, base architecture code, plans files, etc.) is not included, since it
is unmodified third-party code installable via its own public package rather
than project deliverable code.

---

## Known issues (disclosed for transparency)

- **`compute_nnlandmark_native_flops.py`**: the freshly-instantiated model in
  this script has ~30.8M parameters, but the actual trained checkpoint has
  ~88.2M — the architecture configuration in this script does not match what
  was actually trained. Its FLOPs output should not be relied upon without
  further debugging.
- **Parameter/FLOPs double-counting for the two nnLandmark-transplant
  variants** (`M1_nnLandmark`, `M2_nnLandmark`): these models' encoder blocks
  (from the `dynamic_network_architectures` package) expose the same
  underlying conv/norm layers via two attribute paths (`.conv`/`.norm` and
  `.all_modules`) for introspection purposes. Naive parameter/FLOPs counting
  tools that don't deduplicate by object identity (including `thop`, as used
  in earlier profiling) will double-count these layers. The true,
  deduplicated parameter counts are 14,046,054 (AortaPlane+nnLandmark) and
  31,196,399 (Baseline+nnLandmark); a fully corrected FLOPs figure was still
  being worked out as of this submission.

---

## Citation

A related conference submission is in preparation:

> Vandhanaa, et al. *"Direct Regression versus Landmark-Based Aortic Annulus
> Plane Detection from 3D Cardiac CT."* MICAD 2026 (under review).

A full citation with venue details will be added once the paper is finalised.

## Acknowledgement

Thanks to Dr Ahmed Sayed for supervision throughout this project, and to Dr
Fuyu Cheng for clinical input on the MICAD 2026 submission. This project also
builds on MIC-DKFZ's [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) /
nnLandmark framework for the comparison baseline.
