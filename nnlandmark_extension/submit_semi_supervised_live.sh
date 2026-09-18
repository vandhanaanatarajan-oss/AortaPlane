#!/bin/bash
#SBATCH -J semi_supervised_live
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/semi_supervised_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/semi_supervised_%j.err

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate
export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

# --time=96:00:00 (4 days) is a generous upper bound given each iteration
# involves predicting up to 143 unlabelled cases across 5 folds, then a
# full 1000-epoch retrain across 5 folds -- CHECK actual per-iteration
# wall-clock time after iteration 1 completes and adjust expectations /
# max_iterations in the script if this is running longer than your
# remaining timeline allows.
python semi_supervised_iterative.py