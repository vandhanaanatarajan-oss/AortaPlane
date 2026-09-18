#!/bin/bash
#SBATCH -J v1_nnLandmark_DifferentialLR
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/nnLandmark_DifferentialLR_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/nnLandmark_DifferentialLR_%j.err

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate

export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

nnLM_train 741 3d_fullres 0 -p nnUNetPlansV1Match -tr nnLandmark_DifferentialLR \
    -pretrained_weights /gpfs/DERI-ecgai/001_CTA_Segmention/models/Aorta_v1.0/weights/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
