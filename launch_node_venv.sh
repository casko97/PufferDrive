#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pufferdrive_py311_cu121_torch210"
MINIFORGE_MODULE="Miniforge3/24.7.1-2-hpc1-bdist"
BUILDENV_MODULE="buildenv-gcccuda/12.1.1-gcc12.3.0"
export ENV_NAME MINIFORGE_MODULE BUILDENV_MODULE

srun --pty bash -lc '
set -euo pipefail

log() { printf "[%s] %s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }

log "Loading modules..."
module load "$MINIFORGE_MODULE"
module load "$BUILDENV_MODULE"

if ! command -v mamba >/dev/null 2>&1; then
  log "mamba not found on PATH after module load."
  exit 1
fi

# Avoid set -u failures in the mamba wrapper when CONDA_DEFAULT_ENV is unset.
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"

eval "$(mamba shell hook --shell bash)"

if mamba env list | awk "NR>2 {print \$1}" | grep -qx "$ENV_NAME"; then
  log "Found env: $ENV_NAME."
else
  log "Env not found: $ENV_NAME. Creating."
  mamba create -y --name "$ENV_NAME" python=3.11
  log "Installing pytorch..."
  CONDA_OVERRIDE_CUDA=12.1 mamba install -y --name "$ENV_NAME" "pytorch==2.10.0=cuda*"
fi

log "Activating env: $ENV_NAME."
mamba activate "$ENV_NAME"
log "Active CONDA_PREFIX: ${CONDA_PREFIX:-<none>}"

exec bash -i
'
