#!/bin/bash
#SBATCH -J nnLM_pseudo_predict
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/logs/predict_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/logs/predict_%j.err

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate

export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

nnLM_predict -i /data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/input \
             -o /data/DERI-ecgai/__users/Vandhanaa/pseudo_labeling/output \
             -d 741 \
             -c 3d_fullres \
             -p nnUNetPlansAortaMatch \
             -tr nnLandmark \
             -f 0 \
             -npp 1 \
             -nps 1 \
             --save_probabilities
