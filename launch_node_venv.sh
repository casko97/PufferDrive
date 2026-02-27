#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="py311_cu121_torch210"
MINIFORGE_MODULE="Miniforge3/24.7.1-2-hpc1-bdist"
BUILDENV_MODULE="buildenv-gcccuda/12.1.1-gcc12.3.0"

srun --pty bash -lc "
module load ${MINIFORGE_MODULE}
module load ${BUILDENV_MODULE}

if mamba env list | awk '{print \$1}' | grep -Fxq \"${ENV_NAME}\"; then
  mamba activate ${ENV_NAME}
else
  mamba create -y --name ${ENV_NAME} python=3.11
  mamba activate ${ENV_NAME}
  CONDA_OVERRIDE_CUDA=12.1 mamba install -y 'pytorch==2.10.0=cuda*'
fi

exec bash -l
"
