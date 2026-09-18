#!/bin/bash
#SBATCH -J semisup_iter1_retrain
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/semisup_iter1_fold%a_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/semisup_iter1_fold%a_%j.err
#SBATCH --array=0-4

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate
export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

# Each fold reinitializes from its OWN matching Aorta_v1.0 fold checkpoint
# (same fold-matching principle used for PartialUnfreeze's folds 1-4),
# fine-tuning fresh on the pseudo-labelled Dataset743 rather than
# continuing from the already-trained PartialUnfreeze weights -- this
# keeps this iteration an independent, fair test of whether pseudo-label
# augmentation helps, per the semi-supervised script's original design.
nnLM_train 743 3d_fullres ${SLURM_ARRAY_TASK_ID} -p nnUNetPlansV1Match -tr nnLandmark_PartialUnfreeze \
    -pretrained_weights /gpfs/DERI-ecgai/001_CTA_Segmention/models/Aorta_v1.0/weights/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_final.pth