cat > /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/retrain_v1_matched.sh << 'RETRAINEOF'
#!/bin/bash
#SBATCH --job-name=v1_matched_retrain
#SBATCH --account=pilot_apini
#SBATCH --partition=apini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v1_matched_retrain_%a_%j.out
#SBATCH --error=logs/v1_matched_retrain_%a_%j.err
#SBATCH --array=0-4

cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison
module unload miniforge
module load python/3.12.1-gcc-12.2.0
module load cuda/12.6.2-gcc-12.2.0
source nnlm_venv/bin/activate
export nnLM_raw=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/raw
export nnLM_preprocessed=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/preprocessed
export nnLM_results=/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/nnLM_data/results
export PYTHONUNBUFFERED=1

# CHECK: confirm this V1 weights path matches exactly what the original
# PartialUnfreeze training used (see AnnulusPlaneNet notes) before
# trusting this run's results as genuinely comparable to the original.
V1_WEIGHTS="/gpfs/DERI-ecgai/001_CTA_Segmention/models/Aorta_v1.0/weights/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_final.pth"

nnLM_train 741 3d_fullres $SLURM_ARRAY_TASK_ID \
  -p nnUNetPlansV1Match \
  -tr nnLandmark_PartialUnfreeze \
  -pretrained_weights "$V1_WEIGHTS"
RETRAINEOF