#!/usr/bin/env bash
# Submit with: sbatch -A <project-id> scripts/train_gigaflow.slurm
#
# Eight local DDP ranks each collect 32,000 transitions per PPO update. The
# configuration therefore produces GIGAFLOW's disclosed 256,000-sample global
# batch and targets one trillion aggregate transitions. Submit through
# scripts/submit_gigaflow.sh so the required W&B credentials are checked before
# this job enters the queue.
#
# Berzelius currently accepts at most three days at submission. Once this job
# has started, deliberately extend it only after checking the token cost:
#   nsc-boost-timelimit -c -t 10-00:00:00 <job-id>
# A ten-day request may require --accept-the-risks. This is intentionally not
# automated because it spends project boost tokens.
#
# Adjust the module names and conda environment to match your cluster setup.
# The defaults below follow berzelius-setup.md.
#SBATCH --job-name=pufferdrive-gigaflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# The NSC Mamba wrapper reads CONDA_DEFAULT_ENV before it exists. Enable
# nounset only after activating the environment.
set -eo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0

mamba activate puffer-nov-cu121.venv
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the PufferDrive repository root}"
mkdir -p experiments logs

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Export it before submitting." >&2
  exit 1
fi

# Fail early if the configured training maps are unavailable.
test -f resources/drive/binaries/training/map_000.bin

export PYTHONUNBUFFERED=1
# Eight ranks share the CPU allocation, so give each rank eight CPU threads.
export OMP_NUM_THREADS=8
export NCCL_ASYNC_ERROR_HANDLING=1

# Keep W&B's local files and PufferLib checkpoints together under the Slurm ID.
wandb_results_dir="${PWD}/logs/wandb-results-${SLURM_JOB_ID}"
mkdir -p "${wandb_results_dir}"
export WANDB_DIR="${wandb_results_dir}"
wandb_group="${WANDB_GROUP:-gigaflow-inspired}-job-${SLURM_JOB_ID}"
wandb_name="${WANDB_RUN_NAME:-gigaflow-inspired}-job-${SLURM_JOB_ID}"

srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m pufferlib.pufferl train puffer_drive_gigaflow \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-pufferdrive}" \
  --wandb-group "${wandb_group}" \
  --wandb-name "${wandb_name}" \
  --train.data-dir "${wandb_results_dir}"
