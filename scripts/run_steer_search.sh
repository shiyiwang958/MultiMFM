#!/usr/bin/env bash
# Reproduce steer_search (SS) for one QM9 property with the best-known config.
#
#   scripts/run_steer_search.sh <property> [num_samples] [mu] [reward_scale] [select_min_t]
#
# property in {alpha, cv, gap, homo, lumo, mu}. The per-property mu / reward_scale
# / select_min_t below are the configs behind the reported n=1000 SS numbers; pass
# positional overrides to sweep them.
set -euo pipefail
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/env.sh"

PROPERTY="${1:-alpha}"
NUM="${2:-100}"

# Best-known per-property steering config (mu, reward_scale, select_min_t).
case "${PROPERTY}" in
  alpha) MU=11.0;  RS=0.3;     SELMINT=0.3 ;;
  cv)    MU=5.0;   RS=1.5;     SELMINT=0.5 ;;
  gap)   MU=1.0;   RS=12000.0; SELMINT=0.3 ;;
  homo)  MU=1.5;   RS=41500.0; SELMINT=0.3 ;;
  lumo)  MU=1.5;   RS=12000.0; SELMINT=0.3 ;;
  mu)    MU=3.0;   RS=8.638;   SELMINT=0.3 ;;
  *) echo "unknown property '${PROPERTY}' (expected alpha|cv|gap|homo|lumo|mu)"; exit 1 ;;
esac
MU="${3:-$MU}"; RS="${4:-$RS}"; SELMINT="${5:-$SELMINT}"

echo "steer_search: prop=${PROPERTY} n=${NUM} mu=${MU} reward_scale=${RS} select_min_t=${SELMINT}"
python -m multimfm.steer_search \
  --property-name "${PROPERTY}" \
  --num-samples "${NUM}" \
  --mu "${MU}" --reward-scale "${RS}" --select-min-t "${SELMINT}" \
  --mfm-diagonal --guide-every 1 --value-samples 32 --value-glass-steps 4 \
  --sample-steps 128 --pb-workers 16 \
  --output-dir "${REPO_ROOT}/outputs/steer_search/${PROPERTY}"
