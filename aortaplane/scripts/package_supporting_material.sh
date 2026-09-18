#!/bin/bash
# package_supporting_material.sh
#
# Builds the supporting-material zip for dissertation submission.
# Run from the project root: /data/DERI-ecgai/__users/Vandhanaa
#
# Includes: source code, native model checkpoints (M1 stripped of
# optimizer/scheduler state, M2 as-is), confirmed SLURM launch scripts,
# representative training logs, and pinned requirements for both venvs.
# Excludes: patient data, venvs themselves, __pycache__, superseded/
# developmental scripts (see README for the full exclusion rationale).
#
# PREREQUISITES (run once before this script, if not already done):
#   pip freeze > requirements_main.txt                    (in main venv)
#   module unload miniforge
#   module load python/3.12.1-gcc-12.2.0
#   module load cuda/12.6.2-gcc-12.2.0
#   source nnlandmark_comparison/nnlm_venv/bin/activate
#   pip freeze > requirements_nnlandmark.txt
#   deactivate; module unload cuda/...; module unload python/...
#   source venv/bin/activate   (back to main venv before running this script)

set -e

STAGE_DIR="AortaPlane_SupportingMaterial"
ZIP_NAME="AortaPlane_SupportingMaterial.zip"

rm -rf "$STAGE_DIR" "$ZIP_NAME"
mkdir -p "$STAGE_DIR"

# --- core source ---
mkdir -p "$STAGE_DIR/src"
cp src/model.py src/dataset.py src/loss.py src/coordconv3d.py "$STAGE_DIR/src/"

# --- training scripts + root-level variant models ---
cp train_m1_variants.py train_m2_variants.py "$STAGE_DIR/"
cp m1_unet.py m2_resnet3d_noskip.py m2_densenet3d.py densenet3d_encoder.py "$STAGE_DIR/"

# --- arch_variants (nnLandmark-transplant models) ---
mkdir -p "$STAGE_DIR/arch_variants"
cp nnlandmark_comparison/arch_variants/m1_densenet3d.py \
   nnlandmark_comparison/arch_variants/m1_nnlandmark.py \
   nnlandmark_comparison/arch_variants/m2_nnlandmark.py \
   "$STAGE_DIR/arch_variants/"

# --- evaluation / analysis scripts ---
mkdir -p "$STAGE_DIR/evaluation"
cp scripts/evaluation/eval_cv.py compute_full_grid_5fold_true_test.py compute_variant_flops.py \
   "$STAGE_DIR/evaluation/" 2>/dev/null || echo "NOTE: some evaluation scripts not found at expected path -- check filenames/location and copy manually"
cp nnlandmark_comparison/compute_nnlandmark_native_flops.py \
   nnlandmark_comparison/compute_nnlandmark_native_params.py \
   nnlandmark_comparison/compute_nnlandmark_mae.py \
   "$STAGE_DIR/evaluation/" 2>/dev/null || echo "NOTE: nnLandmark evaluation scripts not found at expected path -- check manually"

# --- configs / manifests (small, no patient-identifiable content) ---
cp fold_config.json dataset_manifest.csv "$STAGE_DIR/" 2>/dev/null || echo "NOTE: fold_config.json / dataset_manifest.csv not found at root -- check location"

# --- nnlandmark_comparison codebase (excluding data + venv) ---
mkdir -p "$STAGE_DIR/nnlandmark_comparison"
cp nnlandmark_comparison/convert_annulus.py \
   nnlandmark_comparison/semi_supervised_iterative.py \
   nnlandmark_comparison/build_matched_splits.py \
   "$STAGE_DIR/nnlandmark_comparison/" 2>/dev/null || echo "NOTE: some nnlandmark_comparison scripts not found -- check manually"

# fine-tuning strategy classes (FrozenEncoder, PartialUnfreeze, DifferentialLR --
# the actual novel trainer code for this project; NOT a broad trainer glob,
# which would also sweep in ~25 unmodified stock nnU-Net trainer files)
mkdir -p "$STAGE_DIR/nnlandmark_comparison/training/nnUNetTrainer/variants"
cp nnlandmark_comparison/nnLandmark/nnlandmark/training/nnUNetTrainer/variants/finetune_strategies.py \
   "$STAGE_DIR/nnlandmark_comparison/training/nnUNetTrainer/variants/" 2>/dev/null || echo "NOTE: finetune_strategies.py not found at expected path -- check manually"

# --- results ---
mkdir -p "$STAGE_DIR/results"

# Per-sample predictions (CV + test, all native models: M1/M2/CoordConv),
# built by consolidate_predictions.py. Sanity-checked against Table II of
# the dissertation before packaging (see README). Supersedes the older
# stage1_test_results.csv (no model column, no pred/GT vectors).
python3 consolidate_predictions.py || echo "NOTE: consolidate_predictions.py failed -- run manually first"
cp results/all_sample_predictions.csv "$STAGE_DIR/results/" 2>/dev/null \
   || echo "NOTE: results/all_sample_predictions.csv not found"

# Aggregate/summary CSVs (small, no per-patient identifiable content) --
# REVIEW before including
find . -maxdepth 2 -iname "*results*.csv" -o -iname "*summary*.json" 2>/dev/null | \
    grep -v -E "checkpoints|venv|nnLM_data|stage1_test_results|cv_results\.csv$|ensembled_test_predictions|per_fold_predictions" | \
    xargs -I{} cp {} "$STAGE_DIR/results/" 2>/dev/null || true

# --- README ---
cp README.md "$STAGE_DIR/"  # place the README (from this conversation) here first

# --- checkpoints (native models only; M1's optimizer/scheduler state
#     stripped -- only needed for resuming training, not for inference/
#     reproducing reported metrics -- keeps this well under budget) ---
mkdir -p "$STAGE_DIR/checkpoints"
python3 -c "
import torch
sd = torch.load('checkpoints/M1_fold0/M1_fold0_best.pth', map_location='cpu')
stripped = {k: v for k, v in sd.items() if k not in ('optimizer_state_dict', 'scheduler_state_dict')}
torch.save(stripped, '$STAGE_DIR/checkpoints/M1_fold0_best_weights_only.pth')
print('M1_fold0 stripped checkpoint saved')
" || echo "NOTE: M1 checkpoint stripping failed -- check torch is importable in this shell"
cp checkpoints/M2_fold0/M2_fold0_best.pth "$STAGE_DIR/checkpoints/M2_fold0_best.pth" \
   2>/dev/null || echo "NOTE: M2_fold0 checkpoint not found -- check path"

# --- SLURM launch scripts (confirmed real, not superseded/developmental --
#     see README for what was excluded and why) ---
mkdir -p "$STAGE_DIR/slurm_scripts"
cp train_arch_variants.sh train_arch_variants_folds1to4.sh train_arch_nnlandmark_folds1to4.sh \
   deploy_arch_variants.sh deploy_nnlandmark_variants.sh deploy_eval_script.sh \
   "$STAGE_DIR/slurm_scripts/" 2>/dev/null || echo "NOTE: some root SLURM scripts not found -- check manually"
cp nnlandmark_comparison/train_v1_nnLandmark_FrozenEncoder.sh \
   nnlandmark_comparison/train_v1_nnLandmark_PartialUnfreeze.sh \
   nnlandmark_comparison/train_v1_nnLandmark_DifferentialLR.sh \
   nnlandmark_comparison/submit_semi_supervised_37p.sh \
   "$STAGE_DIR/slurm_scripts/" 2>/dev/null || echo "NOTE: some nnlandmark_comparison SLURM scripts not found -- check manually"

# --- representative training logs (small, real provenance evidence) ---
mkdir -p "$STAGE_DIR/logs/native"
cp logs/M1_fold0_13053498.out logs/M1_fold0_13053498.err \
   logs/M2_fold0_13062157.out logs/M2_fold0_13062157.err \
   "$STAGE_DIR/logs/native/" 2>/dev/null || echo "NOTE: native fold0 logs not found -- check paths"
mkdir -p "$STAGE_DIR/logs/nnlandmark_finetuning"
# explicit whitelist only -- nnlandmark_comparison/logs/ actually contains
# ~120 files from many superseded/exploratory runs (pretrain_*, contrastive_ft_*,
# encoder_transplant_*, train_741_*, v1_37p_*, v1_matched_retrain_*,
# semi_supervised_*, semisup_iter1_*, etc.) with no corresponding script in
# slurm_scripts/ -- only copy logs whose job-name prefix matches a confirmed
# real launcher (FrozenEncoder/PartialUnfreeze/DifferentialLR strategies,
# and semisup_37p matching submit_semi_supervised_37p.sh's #SBATCH -J name)
cp nnlandmark_comparison/logs/nnLandmark_FrozenEncoder_22173308.out \
   nnlandmark_comparison/logs/nnLandmark_FrozenEncoder_22173308.err \
   nnlandmark_comparison/logs/nnLandmark_FrozenEncoder_22173615.out \
   nnlandmark_comparison/logs/nnLandmark_FrozenEncoder_22173615.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_22173309.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_22173309.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_22173616.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_22173616.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold1_22218080.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold1_22218080.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold2_22218081.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold2_22218081.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold3_22218082.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold3_22218082.err \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold4_22218073.out \
   nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold4_22218073.err \
   nnlandmark_comparison/logs/nnLandmark_DifferentialLR_22173310.out \
   nnlandmark_comparison/logs/nnLandmark_DifferentialLR_22173310.err \
   nnlandmark_comparison/logs/nnLandmark_DifferentialLR_22173617.out \
   nnlandmark_comparison/logs/nnLandmark_DifferentialLR_22173617.err \
   nnlandmark_comparison/logs/semisup_37p_23210815.out \
   nnlandmark_comparison/logs/semisup_37p_23210815.err \
   nnlandmark_comparison/logs/semisup_37p_23216189.out \
   nnlandmark_comparison/logs/semisup_37p_23216189.err \
   "$STAGE_DIR/logs/nnlandmark_finetuning/" 2>/dev/null || echo "NOTE: some whitelisted nnlandmark logs not found -- check filenames/job IDs"

# --- environment pins (both venvs -- main project + nnLandmark comparison) ---
mkdir -p "$STAGE_DIR/environment"
cp requirements_main.txt "$STAGE_DIR/environment/" 2>/dev/null \
   || echo "NOTE: requirements_main.txt not found -- run 'pip freeze > requirements_main.txt' in the main venv first"
cp requirements_nnlandmark.txt "$STAGE_DIR/environment/" 2>/dev/null \
   || echo "NOTE: requirements_nnlandmark.txt not found -- generate from nnlandmark_comparison/nnlm_venv (see README)"

# --- clean any stray cache files that slipped in ---
find "$STAGE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE_DIR" -name "*.pyc" -delete

zip -r "$ZIP_NAME" "$STAGE_DIR"

echo ""
echo "Built $ZIP_NAME"
echo "Total size:"
du -h "$ZIP_NAME"
echo "Contents:"
find "$STAGE_DIR" -type f | sort
echo ""
echo "REVIEW BEFORE SUBMITTING:"
echo "  - check every 'NOTE:' warning printed above and copy missing files manually"
echo "  - open results/ and confirm no file contains per-patient identifiable data"
echo "  - confirm zip size is under your 50MB submission limit"
echo "  - confirm README.md content matches what's actually in the zip"