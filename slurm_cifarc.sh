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
#SBATCH --output=logs/cifar100c/output_%j.txt

# Use node-local scratch for MIOpen DB (avoid Lustre/NFS locking issues)
MIOPEN_LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/miopen-${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="$MIOPEN_LOCAL"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_LOCAL"
mkdir -p "$MIOPEN_LOCAL"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=1

# Activate conda in non-interactive shells and activate the env
source /scratch/project_465002264/miniconda3/etc/profile.d/conda.sh
conda activate rem

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# --plot_loss --plot_ema_alpha 0.9 --plot_loss_path plots/M2A/Loss/losses_marn_amr-1k.png \

cd /users/doloriel/work/Repo/M2A/data_cifar
python -m cifar100c_vit_m2a \
     --cfg cfgs/cifar100/m2a.yaml \
     --data_dir /scratch/project_465002264/datasets/cifar100c \
     --lr 0.0001 \
     --seed 1 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking spatial \
     --num_squares 1 \
     --mask_type binary \
     --m 0.1 --n 3 \
     --mcl_distance ce \
     --steps 1 \
     --disable_erl \
     CORRUPTION.NUM_EX 10000

# cd /users/doloriel/work/Repo/M2A/data_cifar
# python -m cifar10c_vit_m2a \
#      --cfg cfgs/cifar10/m2a.yaml \
#      --data_dir /scratch/project_465002264/datasets/cifar10c \
#      --lr 0.001 \
#      --seed 1 \
#      --lamb 1.0 \
#      --margin 0.0 \
#      --random_masking spatial \
#      --num_squares 1 \
#      --mask_type binary \
#      --m 0.1 --n 3 \
#      --mcl_distance ce \
#      --steps 1 \
#      --disable_erl \
#      CORRUPTION.NUM_EX 10000