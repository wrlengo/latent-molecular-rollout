#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name=pka_goal
#SBATCH --output=/home/bingxing2/home/scx9kyg/pKa/runs/goal_rollout_%j.log
#SBATCH --error=/home/bingxing2/home/scx9kyg/pKa/runs/goal_rollout_%j.err
#SBATCH --time=02:00:00

module load compilers/gcc/11.3.0 compilers/cuda/12.4 cudnn/8.9.5.29_cuda12.x
source /home/bingxing2/apps/anaconda3/etc/profile.d/conda.sh
conda activate latentchem_dev
GOMP_LIB=$(ls $CONDA_PREFIX/lib/python3.10/site-packages/scikit_learn.libs/libgomp-*.so.1.0.0 2>/dev/null | head -1)
if [ -n "$GOMP_LIB" ]; then export LD_PRELOAD="$GOMP_LIB"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/bingxing2/home/scx9kyg/pKa/experiments
python3 goal_rollout.py \
  --h_train /home/bingxing2/home/scx9kyg/pKa/runs/fig/h_train.npz \
  --h_test /home/bingxing2/home/scx9kyg/pKa/runs/fig/h_test.npz \
  --pairs /home/bingxing2/home/scx9kyg/pKa/experiments/pairs_train.jsonl \
  --test_pairs /home/bingxing2/home/scx9kyg/pKa/experiments/frozen_ridge_test_pairs.csv \
  --ridge /home/bingxing2/home/scx9kyg/pKa/experiments/ridge_selected_weights.npz \
  --out /home/bingxing2/home/scx9kyg/pKa/experiments/goal_rollout_results.json \
  --seeds 3 --epochs 120 --patience 20 --batch 128 --hidden 1024 --lr 1e-3 --K 5
echo "EXIT=$?"
