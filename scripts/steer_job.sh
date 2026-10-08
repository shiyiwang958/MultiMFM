#!/usr/bin/env bash
#SBATCH --job-name=mfm-steer
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/steer_logs/slurm-%x-%j.out
#
# One QM9 property-steering run (base-flow trajectory, MFM look-ahead value
# gradient from the guide regressor, held-out oracle evaluation).
#   sbatch --job-name=NAME scripts/steer_job.sh NAME MFM_CKPT {diag|jump} MU RS [N] [SEED] [TSEED] [PROP]
# Results land in outputs/steer_${PROP}/${NAME}/summary.json. Env overrides:
#   VS VSTEPS VBS GMAXT GMINT GEVERY STEPS CRMS ARMS SELMINT
set -euo pipefail
NAME=$1; MFM=$2; MODE=$3; MU=$4; RS=$5
N=${6:-200}; SEED=${7:-8}; TSEED=${8:-2026}; PROP=${9:-alpha}
VS=${VS:-32}; VSTEPS=${VSTEPS:-4}   # look-ahead samples per molecule / MFM steps per look-ahead
VBS=${VBS:-200}                      # molecules per look-ahead chunk (memory only)
GMAXT=${GMAXT:-0.95}; GMINT=${GMINT:-0.05}; GEVERY=${GEVERY:-1}  # guidance window / stride
STEPS=${STEPS:-128}; CRMS=${CRMS:-0}; ARMS=${ARMS:-0}  # ODE steps / guidance RMS caps
SELMINT=${SELMINT:-0.3}              # earliest t whose look-ahead may be steer-search selected
DET=${DET:-0}                        # 1 = disable TF32 + deterministic kernels (reproducible)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/env.sh
DIAG=""; [ "${MODE}" = diag ] && DIAG="--mfm-diagonal"
DETFLAG=""
if [ "${DET}" = 1 ]; then
  DETFLAG="--deterministic"
  # must be set before cuBLAS initializes, hence here rather than in python
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
echo "steer ${NAME}: mfm=${MFM} mode=${MODE} mu=${MU} rs=${RS} n=${N} seed=${SEED} tseed=${TSEED} prop=${PROP} vs=${VS} vsteps=${VSTEPS} gmaxt=${GMAXT} gevery=${GEVERY} steps=${STEPS} crms=${CRMS} arms=${ARMS}"
# shellcheck disable=SC2086
python -m multimfm.steer_search --property-name "${PROP}" --num-samples "${N}" \
  --seed "${SEED}" --target-seed "${TSEED}" \
  --mfm-checkpoint "${MFM}" ${DIAG} ${DETFLAG} --mu "${MU}" --reward-scale "${RS}" --select-min-t "${SELMINT}" \
  --guide-every "${GEVERY}" --guide-min-t "${GMINT}" --guide-max-t "${GMAXT}" \
  --value-samples "${VS}" --value-glass-steps "${VSTEPS}" --value-batch-size "${VBS}" \
  --sample-steps "${STEPS}" --pb-workers 16 \
  --guidance-max-coord-rms "${CRMS}" --guidance-max-atom-rms "${ARMS}" \
  --output-dir "outputs/steer_${PROP}/${NAME}"
