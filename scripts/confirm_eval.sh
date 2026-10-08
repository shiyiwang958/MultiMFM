#!/usr/bin/env bash
#SBATCH --job-name=mfm-confirm
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/ce_vs_mse/slurm-%x-%j.out
#
# Eval-only confirmation of a student checkpoint on an unseen QM9 set: 1024
# molecules of the TFG "validation" split (never used by teacher or student
# training, which use train_flow), fresh seed, with its own GLASS reference.
#   sbatch scripts/confirm_eval.sh CKPT "--time-encoding-max-len 4 --hidden-dim 128 ..."
set -euo pipefail
CKPT="${1:?checkpoint}"
ARCH="${2:-}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/env.sh
echo "confirm: ${CKPT}"
# shellcheck disable=SC2086
# ARCH goes last so it can also override defaults such as --eval-configs.
python -m multimfm.train_mfm_student --device cuda --stage 2 --steps 0 \
  --init-ckpt "${CKPT}" \
  --partition validation --limit-data 1800 --n-eval 1024 --eval-seed 999 \
  --eval-at-start --eval-t-conds 0.0,0.3,0.6,0.9 --glass-steps 64 \
  --eval-configs jump:4,jump:2,jump:1,diag:32 \
  --out-dir outputs/ce_vs_mse/confirm ${ARCH}
