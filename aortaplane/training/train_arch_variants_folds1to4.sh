#!/bin/bash
#SBATCH -J arch_var_f
#SBATCH -A pilot_apini
#SBATCH -p apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /data/DERI-ecgai/__users/Vandhanaa/logs/arch_var_%x_%j.out
#SBATCH -e /data/DERI-ecgai/__users/Vandhanaa/logs/arch_var_%x_%j.err
source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
cd /data/DERI-ecgai/__users/Vandhanaa
python train_${SCRIPT}_variants.py --arch ${ARCH} --fold ${FOLD} --run_name ${RUN_NAME}
