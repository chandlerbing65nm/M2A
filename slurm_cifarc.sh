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
#SBATCH --output=logs/cifar10c/output_%j.txt

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

# cd data_cifar
# python -m cifar100c_vit_m2a \
#      --cfg cfgs/cifar100/m2a.yaml \
#      --data_dir /scratch/project_465002264/datasets/cifar100c \
#      --lr 1e-3 \
#      --seed 1 \
#      --lamb 1.0 \
#      --margin 0.0 \
#      --random_masking spectral \
#      --spatial_type patch \
#      --spectral_type high \
#      --num_squares 1 \
#      --mask_type binary \
#      --m 0.1 \
#      --n 3 \
#      --mcl_distance ce \
#      --steps 1 \
#      --disable_erl \
#      CORRUPTION.NUM_EX 100000

cd data_cifar
python -m cifar10c_vit_m2a \
     --cfg cfgs/cifar10/m2a.yaml \
     --data_dir /scratch/project_465002264/datasets/cifar10c \
     --lr 1e-3 \
     --seed 1 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking spatial \
     --spatial_type patch \
     --spectral_type all \
     --num_squares 1 \
     --mask_type binary \
     --m 0.1 \
     --n 3 \
     --mcl_distance ce \
     --steps 1 \
     --disable_erl \
     CORRUPTION.NUM_EX 100000

# cd /users/doloriel/work/Repo/M2A/data_cifar
# python -m cifar10c_vit_rem \
#      --cfg cfgs/cifar10/rem.yaml \
#      --data_dir /scratch/project_465002264/datasets/cifar10c \
#      CORRUPTION.NUM_EX 10000