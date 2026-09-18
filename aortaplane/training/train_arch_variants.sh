#!/bin/bash
#SBATCH --job-name=arch_variants
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arch_variant_%a_%j.out
#SBATCH --error=logs/arch_variant_%a_%j.err
#SBATCH --array=0-5

source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa

# Array index -> (script, arch, run_name), fold 0 only for now (matching
# M1_fold0.sh/M2_fold0.sh -- the baseline single-fold comparison point).
# CHECK: once these look right, submit folds 1-4 the same way (matching
# M1/M2's own full 5-fold CV) for a complete comparison, same principle
# used for PartialUnfreeze earlier in this project.
case $SLURM_ARRAY_TASK_ID in
  0)
    python train_m1_variants.py \
      --arch unet --fold 0 --run_name M1_unet_fold0
    ;;
  1)
    python train_m1_variants.py \
      --arch densenet3d --fold 0 --run_name M1_densenet3d_fold0
    ;;
  2)
    python train_m2_variants.py \
      --arch resnet3d_noskip --fold 0 --run_name M2_resnet3d_noskip_fold0
    ;;
  3)
    python train_m2_variants.py \
      --arch densenet3d --fold 0 --run_name M2_densenet3d_fold0
    ;;
  4)
    python train_m1_variants.py \
      --arch nnlandmark --fold 0 --run_name M1_nnlandmark_fold0
    ;;
  5)
    python train_m2_variants.py \
      --arch nnlandmark --fold 0 --run_name M2_nnlandmark_fold0
    ;;
esac
