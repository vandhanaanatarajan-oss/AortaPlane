#!/bin/bash
#SBATCH -J contrastive_pretrain_v2
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/pretrain_v2_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/logs/pretrain_v2_%j.err

module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0

cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
source nnlm_venv/bin/activate

export PYTHONUNBUFFERED=1

python pretraining/pretrain_contrastive_v2.py
