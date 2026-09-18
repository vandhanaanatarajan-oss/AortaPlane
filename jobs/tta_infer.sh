#!/bin/bash
#SBATCH --job-name=tta_infer
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tta_infer_%j.out
#SBATCH --error=logs/tta_infer_%j.err
source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa
python tta_infer.py \
    --manifest_path dataset_manifest.csv \
    --root_path processed_output \
    --checkpoint_dir checkpoints \
    --checkpoint_subdir_pattern "M1_fold{fold}" \
    --checkpoint_filename_pattern "M1_fold{fold}_best.pth" \
    --fold_config fold_config.json \
    --n_folds 5 \
    --target_size 64 \
    --use_cbam \
    --out_csv results/ensemble_results_tta.csv
