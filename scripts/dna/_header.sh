# Shared preamble for the DNA (dMFM) Slurm wrappers.
#   source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
# Sets REPO (repo root), activates the multimfm env and cds to the repo root.
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${REPO}"
mkdir -p logs
# shellcheck disable=SC1091
source scripts/env.sh
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
echo "repo=${REPO} host=$(hostname) job=${SLURM_JOB_ID:-none}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi
