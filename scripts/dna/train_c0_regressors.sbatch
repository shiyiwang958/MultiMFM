#!/bin/bash
#SBATCH --job-name=dna_c0_regressors
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# C0 guide (parent set a) and independent oracle (parent set b), Table 11.
# Original: slurm/train_yeast_parent_c0_regressors.sbatch, job 35241778
# (kempner_h100). Same arguments; the shipped checkpoints are
# checkpoints/dna/c0/{guide,oracle}/best_state.pt with their metadata.json.
#
#   sbatch scripts/dna/train_c0_regressors.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

DATA_PT="${DATA_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50.pt}"
SPLIT_PT="${SPLIT_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt}"
RUN_TAG="${RUN_TAG:-2026-07-26}"
GUIDE_OUT="${GUIDE_OUT:-outputs/dna/yeast_parent_c0_guide_${RUN_TAG}}"
ORACLE_OUT="${ORACLE_OUT:-outputs/dna/yeast_parent_c0_oracle_${RUN_TAG}}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-cuda:0}"

common_args=(
  --data_pt "${DATA_PT}"
  --split_pt "${SPLIT_PT}"
  --model_type park_cnn
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch_size "${BATCH_SIZE}"
  --lr 0.001
  --weight_decay 0.0001
  --reverse_complement_augmentation
  --reverse_complement_average
  --device "${DEVICE}"
)

${PY} -u -m dmfm.experiments.train_c0_regressor "${common_args[@]}" \
  --role guide --train_split a_train --validation_split a_val --test_split test \
  --seed 0 --out_dir "${GUIDE_OUT}"

${PY} -u -m dmfm.experiments.train_c0_regressor "${common_args[@]}" \
  --role oracle --train_split b_train --validation_split b_val --test_split test \
  --seed 1 --out_dir "${ORACLE_OUT}"
