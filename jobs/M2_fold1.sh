#!/bin/bash
#SBATCH --job-name=M2_fold1
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/M2_fold1_%j.out
#SBATCH --error=logs/M2_fold1_%j.err

source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa

python train_m2.py     --fold 1     --fold_config fold_config.json     --root_path /data/DERI-ecgai/__users/Vandhanaa/processed_output     --target_size 64     --batch_size 2     --epochs 300     --lr 1e-3     --dropout 0.3     --patience 60     --no_wandb
