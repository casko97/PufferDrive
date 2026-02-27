#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-${CONDA_DEFAULT_ENV:-pufferdrive}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
USE_CUDA="${USE_CUDA:-1}"
USE_EXISTING="${USE_EXISTING:-1}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if command -v stdbuf >/dev/null 2>&1; then
    STDBUF=(stdbuf -oL -eL)
else
    STDBUF=()
fi

run_conda() {
    "${STDBUF[@]}" "${CONDA_EXEC}" "$@"
}

run_in_env() {
    "${STDBUF[@]}" "${CONDA_EXEC}" run -n "${ENV_NAME}" env PYTHONUNBUFFERED=1 "$@"
}

probe_torch() {
    log "Detecting torch/cuda in '${ENV_NAME}'..."
    run_in_env python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"[torch] not available ({exc.__class__.__name__}: {exc})")
    sys.exit(0)

print(f"[torch] version: {torch.__version__}")
print(f"[torch] cuda version: {torch.version.cuda}")
print(f"[torch] cuda available: {torch.cuda.is_available()}")
print(f"[torch] cuda device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    try:
        print(f"[torch] cuda device 0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[torch] cuda device name error: {exc}")
try:
    print(f"[torch] cudnn version: {torch.backends.cudnn.version()}")
except Exception as exc:
    print(f"[torch] cudnn version error: {exc}")
PY
}

if command -v mamba >/dev/null 2>&1; then
    CONDA_EXEC="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_EXEC="conda"
else
    echo "Error: mamba or conda not found in PATH." >&2
    exit 1
fi
log "Using ${CONDA_EXEC}"

if [[ -z "${ENV_NAME}" ]]; then
    echo "Error: ENV_NAME is empty. Pass an env name or activate a conda env first." >&2
    exit 1
fi

log "Target env: ${ENV_NAME}"
log "Options: USE_EXISTING=${USE_EXISTING} USE_CUDA=${USE_CUDA} CUDA_VERSION=${CUDA_VERSION} PYTHON_VERSION=${PYTHON_VERSION}"

if ! run_conda env list | awk 'NR>2 {print $1}' | grep -qx "${ENV_NAME}"; then
    log "Conda env '${ENV_NAME}' not found. Please create it first."
    exit 1
fi

if [[ "${USE_EXISTING}" != "1" ]]; then
    log "USE_EXISTING=${USE_EXISTING} ignored; this script requires a pre-existing env."
fi

log "Using existing env '${ENV_NAME}'..."
probe_torch

log "Installing repo (editable, no build isolation)..."
run_in_env python -m pip install -e . --no-build-isolation

log "Building extensions in-place..."
run_in_env python setup.py build_ext --inplace --force

log "Done. Activate with: conda activate ${ENV_NAME}"
