#!/bin/bash
#SBATCH --gpus=1
#SBATCH --job-name=pka_amp_scale
#SBATCH --output=/home/bingxing2/home/scx9kyg/pKa/runs/pka_amp_scale_%j.log
#SBATCH --error=/home/bingxing2/home/scx9kyg/pKa/runs/pka_amp_scale_%j.err
#SBATCH --time=00:40:00

module load compilers/gcc/11.3.0 compilers/cuda/12.4 cudnn/8.9.5.29_cuda12.x
source /home/bingxing2/apps/anaconda3/etc/profile.d/conda.sh
conda activate latentchem_dev
GOMP_LIB=$(ls $CONDA_PREFIX/lib/python3.10/site-packages/scikit_learn.libs/libgomp-*.so.1.0.0 2>/dev/null | head -1)
if [ -n "$GOMP_LIB" ]; then export LD_PRELOAD="$GOMP_LIB"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8

cd /home/bingxing2/home/scx9kyg/pKa/experiments
python3 fit_amplitude_scale.py \
  --h_train /home/bingxing2/home/scx9kyg/pKa/runs/fig/h_train.npz \
  --pairs /home/bingxing2/home/scx9kyg/pKa/experiments/pairs_train.jsonl \
  --policy_models /home/bingxing2/home/scx9kyg/pKa/experiments/policy_decode_models \
  --pka_head_ckpt /home/bingxing2/home/scx9kyg/pKa/runs/v3_C_latent_head_fix/stage3_sub3/mm_projector.pt \
  --out /home/bingxing2/home/scx9kyg/pKa/experiments/amplitude_scale_train.json
echo "EXIT=$?"
