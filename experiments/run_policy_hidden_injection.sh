#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name=pka_hidden_inject
#SBATCH --output=/home/bingxing2/home/scx9kyg/pKa/runs/hidden_injection_%j.log
#SBATCH --error=/home/bingxing2/home/scx9kyg/pKa/runs/hidden_injection_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -e
module load compilers/gcc/11.3.0 compilers/cuda/12.4 cudnn/8.9.5.29_cuda12.x
source /home/bingxing2/apps/anaconda3/etc/profile.d/conda.sh
conda activate latentchem_dev
GOMP_LIB=$(ls $CONDA_PREFIX/lib/python3.10/site-packages/scikit_learn.libs/libgomp-*.so.1.0.0 2>/dev/null | head -1)
if [ -n "$GOMP_LIB" ]; then export LD_PRELOAD="$GOMP_LIB"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/bingxing2/home/scx9kyg/pKa
EXP=$ROOT/experiments
LIMIT=${1:-2}
TAG=${2:-smoke}
if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" -eq 1 ]; then shift; fi
cd "$EXP"
python3 policy_hidden_injection.py \
  --policy_models "$EXP/policy_decode_models" \
  --reference_policy_results "$EXP/policy_head_baseline_results.json" \
  --h_test "$ROOT/runs/fig/h_test.npz" \
  --test_pairs "$EXP/frozen_ridge_test_pairs.csv" \
  --ckpt "$ROOT/runs/v3_C_gen2_rej1/stage3_sub3" \
  --property_ckpt "$ROOT/runs/v3_C_latent_head_fix/stage3_sub3" \
  --oracle_ckpt "$ROOT/runs/v3_C_latent_head/stage3_sub3" \
  --out "$EXP/policy_hidden_injection_${TAG}.json" \
  --limit "$LIMIT" --steps 0.1 0.5 1.0 2.0 --max_new_tokens 96 \
  "$@"
