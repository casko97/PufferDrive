#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-${CONDA_DEFAULT_ENV:-pufferdrive}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
USE_CUDA="${USE_CUDA:-1}"
USE_EXISTING="${USE_EXISTING:-0}"

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

if [[ "${USE_EXISTING}" == "1" ]]; then
    log "Using existing env '${ENV_NAME}'..."
    if ! run_conda env list | awk 'NR>2 {print $1}' | grep -qx "${ENV_NAME}"; then
        echo "Error: conda env '${ENV_NAME}' not found." >&2
        exit 1
    fi
    probe_torch
else
    log "Creating env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    run_conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

    log "Installing build tools..."
    run_conda install -y -n "${ENV_NAME}" -c conda-forge cmake ninja pkg-config

    if [[ "${USE_CUDA}" == "1" ]]; then
        log "Installing PyTorch with CUDA ${CUDA_VERSION}..."
        run_conda install -y -n "${ENV_NAME}" -c pytorch -c nvidia \
            pytorch torchvision torchaudio "pytorch-cuda=${CUDA_VERSION}"
    else
        log "Installing CPU-only PyTorch..."
        run_conda install -y -n "${ENV_NAME}" -c pytorch pytorch torchvision torchaudio cpuonly
    fi
    probe_torch
fi

log "Installing repo (editable, no build isolation)..."
run_in_env python -m pip install -e . --no-build-isolation

log "Building extensions in-place..."
run_in_env python setup.py build_ext --inplace --force

log "Done. Activate with: conda activate ${ENV_NAME}"
