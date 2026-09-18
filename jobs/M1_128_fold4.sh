#!/bin/bash
#SBATCH --job-name=M1_128_fold4
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/M1_128_fold4_%j.out
#SBATCH --error=logs/M1_128_fold4_%j.err

source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa

python train.py \
    --fold 4 \
    --fold_config fold_config.json \
    --root_path /data/DERI-ecgai/__users/Vandhanaa/data_processed_128 \
    --manifest_path /data/DERI-ecgai/__users/Vandhanaa/dataset_manifest_128.csv \
    --use_cbam \
    --target_size 128 \
    --batch_size 1 \
    --epochs 300 \
    --lr 1e-3 \
    --dropout 0.3 \
    --patience 60 \
    --run_name M1_128_fold4 \
    --save_dir checkpoints/M1_128_fold4 \
    --no_wandb
