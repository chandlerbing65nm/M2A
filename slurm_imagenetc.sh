#!/bin/bash

#SBATCH --job-name=chandlertasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465002264
#SBATCH --output=logs/imagenetc/output_%j.txt

# Use node-local scratch for MIOpen DB (avoid Lustre/NFS locking issues)
MIOPEN_LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/miopen-${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="$MIOPEN_LOCAL"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_LOCAL"
mkdir -p "$MIOPEN_LOCAL"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=1

# Activate conda in non-interactive shells and activate the env
source /scratch/project_465002264/miniconda3/etc/profile.d/conda.sh
conda activate m2a

# cd /users/doloriel/work/Repo/M2A/data_imagenet
# python -m imagenetc \
#       --cfg cfgs/vitb16/source.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       CORRUPTION.NUM_EX 5000

# cd /users/doloriel/work/Repo/M2A/data_imagenet
# python -m imagenetc_tent \
#       --cfg cfgs/vitb16/tent.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       CORRUPTION.NUM_EX 5000

# cd /users/doloriel/work/Repo/M2A/data_imagenet
# python -m imagenetc_cotta \
#       --cfg cfgs/vitb16/cotta.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       CORRUPTION.NUM_EX 5000

# cd /users/doloriel/work/Repo/M2A/data_imagenet
# python -m imagenetc_vida \
#       --cfg cfgs/vitb16/vida.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       --checkpoint /users/doloriel/work/Repo/M2A/ckpt/imagent_vit_vida.pt \
#       --lr 5e-07 \
#       CORRUPTION.NUM_EX 5000

# cd /users/doloriel/work/Repo/M2A/data_imagenet
# python -m imagenetc_rem \
#       --cfg cfgs/vitb16/rem.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       CORRUPTION.NUM_EX 5000

cd /users/doloriel/work/Repo/M2A/data_imagenet
python -m imagenetc_m2a \
      --cfg cfgs/vitb16/m2a.yaml \
      --data_dir /scratch/project_465002264/datasets/imagenetc \
      --seed 1 \
      --random_masking spatial \
      --spatial_type patch \
      --spectral_type all \
      --num_squares 1 \
      --mask_type binary \
      --disable_erl \
      CORRUPTION.NUM_EX 5000