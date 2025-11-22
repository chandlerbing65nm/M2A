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
conda activate rem

cd /users/doloriel/work/Repo/SPARC/data_imagenet
python -m imagenetc_sparc \
      --cfg cfgs/vit/sparc.yaml \
      --data_dir /scratch/project_465002264/datasets/imagenetc \
      --seed 1 \
      --random_masking spectral \
      --num_squares 1 \
      --mask_type binary \
      CORRUPTION.NUM_EX 5000

# cd /users/doloriel/work/Repo/SPARC/data_imagenet
# python -m imagenetc \
#       --cfg cfgs/vit/rem.yaml \
#       --data_dir /scratch/project_465002264/datasets/imagenetc \
#       CORRUPTION.NUM_EX 5000