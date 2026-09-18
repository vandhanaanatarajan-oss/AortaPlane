#!/bin/bash
#SBATCH -J v1_PartialUnfreeze_remaining_folds
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold%a_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/nnLandmark_PartialUnfreeze_fold%a_%j.err
#SBATCH --array=1-4

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate
export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

# SLURM_ARRAY_TASK_ID runs this same script 4 times in parallel, once each
# for fold 1, 2, 3, 4 (fold 0 already trained). Each fold initializes from
# its OWN matching Aorta_v1.0 fold checkpoint -- NOT all from fold_0's,
# which is what the original fold_0 job script did (correct for training
# fold_0 specifically, but would be a fold-mismatch bug if copied as-is
# for folds 1-4, same issue caught and fixed in semi_supervised_iterative.py).
nnLM_train 741 3d_fullres ${SLURM_ARRAY_TASK_ID} -p nnUNetPlansV1Match -tr nnLandmark_PartialUnfreeze \
    -pretrained_weights /gpfs/DERI-ecgai/001_CTA_Segmention/models/Aorta_v1.0/weights/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_final.pth