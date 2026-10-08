#!/usr/bin/env bash
# Activate the MultiMFM micromamba environment and set repo-local caches.
#   source scripts/env.sh
# Override the env name with MULTIMFM_ENV=<name>.

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
ENV_NAME="${MULTIMFM_ENV:-multimfm}"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${ENV_NAME}"
else
  # Clusters with only conda/mamba (e.g. FASRC Miniforge).
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
fi

# CUDA runtime bundled with the pip torch wheel, plus the env's own libstdc++.
NV="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia"
if [ -d "${NV}" ]; then
  for d in "${NV}"/*/lib; do LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; done
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Offline Hugging Face (bundled QM9) + repo-local scratch caches.
export HF_HOME="${REPO_ROOT}/data/qm9"
export HF_DATASETS_CACHE="${REPO_ROOT}/data/qm9/datasets"
export HF_HUB_CACHE="${REPO_ROOT}/data/qm9/hub"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${REPO_ROOT}/cache/matplotlib"
export XDG_CACHE_HOME="${REPO_ROOT}/cache/xdg"
export TORCH_HOME="${REPO_ROOT}/cache/torch"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "MultiMFM env ready: ${ENV_NAME} ($(python -c 'import sys;print("python",sys.version.split()[0])'))"
