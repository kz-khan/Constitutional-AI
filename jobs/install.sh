#!/bin/bash
#SBATCH --job-name=install-env
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/install-%j.out
#SBATCH --error=logs/install-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Create venv if missing
if [ ! -d "venv" ]; then
  python -m venv venv
fi
source venv/bin/activate

pip install -U pip setuptools wheel

# Install torch cu118 (this will still bring nvidia libs, but fewer extra deps overall)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core NLP deps
pip install numpy transformers datasets accelerate safetensors

# vLLM: force wheel-only, use version that previously ran for you
# pip install --only-binary=:all: "vllm==0.11.2"

# Remove pip download cache (optional)
# pip cache purge
