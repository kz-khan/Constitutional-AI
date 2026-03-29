#!/bin/bash
#SBATCH --job-name=cai-qwen-2.5-abliterated
#SBATCH --partition=A40short
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cache .hf

echo "HOST:  $(hostname)"
echo "DATE:  $(date)"
echo "JOBID: $SLURM_JOB_ID"

nvidia-smi || true

module purge
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

# Caches in project space
export HF_HOME="$SLURM_SUBMIT_DIR/.hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export VLLM_DOWNLOAD_DIR="$SLURM_SUBMIT_DIR/cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$VLLM_DOWNLOAD_DIR"

# Model config
export MODEL_NAME="huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3 "
export VLLM_DTYPE="half"
export MAX_MODEL_LEN="8192"
export GPU_MEM_UTIL="0.90"
export TP="1"
export ENFORCE_EAGER="1"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Starting main.py: $(date)"
python -u src/main.py
echo "DONE: $(date)"
