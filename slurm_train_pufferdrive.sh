#!/bin/bash
# SLURM batch job pufferdrive training

#SBATCH --job-name=pufferdrive_1
#SBATCH --account=berzelius-2026-36
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --constraint=thin
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=casko@kth.se

#SBATCH --output /proj/rpl-soro/users/x_carsk/PufferDrive/logs/outlog-%J.log
#SBATCH --error  /proj/rpl-soro/users/x_carsk/PufferDrive/logs/errlog-%J.log

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate pufferdrive_py311_cu121_torch210

