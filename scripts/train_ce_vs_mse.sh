#!/usr/bin/env bash
#SBATCH --job-name=mfm-ce-vs-mse
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/ce_vs_mse/slurm-%x-%j.out
#
# Two-stage QM9 multiMFM training for one strand of the CE-vs-MSE comparison:
#   stage 1: diagonal loss only (Prop. 1), init from the base flow
#   stage 2: diagonal + ESD, init from the stage-1 EMA checkpoint
# The strands differ only in the atom term (--atom-loss-type); coordinates use the
# repo's velocity MSE in both.
#
#   sbatch --job-name=mfm-mse scripts/train_ce_vs_mse.sh mse
#   sbatch --job-name=mfm-ce  scripts/train_ce_vs_mse.sh ce
set -euo pipefail

STRAND="${1:?usage: train_ce_vs_mse.sh mse|ce}"
case "${STRAND}" in mse|ce) ;; *) echo "unknown strand: ${STRAND}" >&2; exit 2;; esac
STEPS1="${STEPS1:-20000}"
STEPS2="${STEPS2:-20000}"
TAG="${TAG:-ce_vs_mse}"
RUN_STAGE1="${RUN_STAGE1:-1}"   # 0 = reuse the latest stage-1 checkpoint in ${OUT}
# Extra stage-2 args shared by both strands (ESD loss on coordinates -- the CE
# strand's atom term is always the logit-space CE -- and a stage-2 lr override).
# On QM9 plain-MSE and adaptive ESD both diverged at ~3.5-5.5k steps; the
# per-sample delta clip + lower lr below ran 20k steps stably.
ESD_ARGS="${ESD_ARGS:---esd-loss-type adaptive --esd-adaptive-p 0.5 --esd-adaptive-c 0.01 --esd-delta-clip 2.0 --lr 5e-5}"
S2_SUFFIX="${S2_SUFFIX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"     # appended to both stages (e.g. --time-encoding-max-len 4)
STAGE1_ARGS="${STAGE1_ARGS:-}"   # appended to stage 1 only
S2_INIT="${S2_INIT:-}"           # stage-2 init checkpoint (default: latest stage-1 final)
WANDB_GROUP="${WANDB_GROUP:-ce_vs_mse}"
OUT="outputs/${TAG}/${STRAND}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/env.sh
mkdir -p "${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

COMMON=(
  --device cuda
  --atom-loss-type "${STRAND}"
  --limit-data 50000 --n-eval 256
  --batch-size 64 --lr 2e-4 --warmup 500 --ema 0.999
  --t-max 0.95
  --coord-loss-weight 1.0 --atom-loss-weight 1.0
  --eval-every 2000 --eval-at-start
  --eval-t-conds 0.0,0.3,0.6,0.9 --glass-steps 64 --eval-seed 1234
  --log-every 100 --seed 0
  --wandb-project multimfm --wandb-entity fwang958-harvard-university
  --wandb-group "${WANDB_GROUP}"
)
# shellcheck disable=SC2206
COMMON+=(${EXTRA_ARGS})

if [ "${RUN_STAGE1}" = "1" ]; then
  python -m multimfm.train_mfm_student "${COMMON[@]}" \
    --stage 1 --steps "${STEPS1}" --init-from-teacher \
    --eval-configs diag:4,diag:32 \
    --wandb-name "${STRAND}-stage1${S2_SUFFIX}" \
    --out-dir "${OUT}" ${STAGE1_ARGS}
fi

if [ -n "${S2_INIT}" ]; then
  # may be a glob (resolved at run time, e.g. for jobs chained with --dependency)
  # shellcheck disable=SC2086
  STAGE1_CKPT="$(ls -d ${S2_INIT} | sort | tail -n 1)"
else
  STAGE1_DIR="$(ls -d "${OUT}"/stage1_* | sort | tail -n 1)"
  STAGE1_CKPT="${STAGE1_DIR}/student_step_${STEPS1}.pt"
fi
echo "stage-2 init: ${STAGE1_CKPT}"

python -m multimfm.train_mfm_student "${COMMON[@]}" \
  --stage 2 --steps "${STEPS2}" --init-ckpt "${STAGE1_CKPT}" \
  --esd-weight 1.0 --diag-weight 1.0 ${ESD_ARGS} \
  --eval-configs diag:32,jump:1,jump:2,jump:4 \
  --wandb-name "${STRAND}-stage2${S2_SUFFIX}" \
  --out-dir "${OUT}${S2_SUFFIX:+/s2${S2_SUFFIX}}"
