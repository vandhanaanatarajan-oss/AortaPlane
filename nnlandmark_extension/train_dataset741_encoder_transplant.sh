#!/bin/bash
#SBATCH -J nnLM_encoder_transplant
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/encoder_transplant_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/encoder_transplant_%j.err

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0

cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate

export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

nnLM_train 741 3d_fullres 0 -p nnUNetPlansAortaMatch \
    -pretrained_weights /data/DERI-ecgai/001_CTA_Segmention/models/Aorta_v0/weights/Dataset050_SEGA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
