#!/bin/bash
#SBATCH --job-name=dna_train_base
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Base DFM DiT generator for one length (Table 9, top half).
# Original: slurm/train_yeast_parent_L{50,100,200,400}_dfm.sbatch, jobs
# 35140730/32/29/34 (partition seas_gpu, account albergo_lab). Arguments below
# are identical. "best.pt" is the best a_val checkpoint (steps 50k/55k/45k/45k),
# not the step-200k model.
#
#   LENGTH=50 sbatch scripts/dna/train_base_dfm.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L="${LENGTH:?Set LENGTH to one of 50, 100, 200, 400}"
RUN_NAME="${RUN_NAME:-yeast_parent_a_L${L}_dfm_dit_h192_b4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dna/${RUN_NAME}}"
STEPS="${STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VAL_EVERY="${VAL_EVERY:-5000}"
# Original per-length batch sizes (train_yeast_parent_L*_dfm.sbatch, recorded in args.json).
case "${L}" in
  50) DEFAULT_BS=512 ;; 100) DEFAULT_BS=256 ;; 200) DEFAULT_BS=128 ;; 400) DEFAULT_BS=64 ;;
  *) echo "Unsupported LENGTH=${L}" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BS}}"

${PY} -u -m dmfm.experiments.train_base_dfm \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --yeast_data_pt "data/dna/yeast_parent_disjoint/yeast_parent_L${L}.pt" \
  --yeast_split_pt "data/dna/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt" \
  --yeast_split a_train \
  --yeast_val_split a_val \
  --max_train_seqs 0 \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --hidden_dim 192 \
  --transformer_heads 6 \
  --transformer_ff_mult 4 \
  --dit_blocks 4 \
  --dropout 0.0 \
  --gaussian_beta_schedule linear \
  --save_every "${SAVE_EVERY}" \
  --val_every "${VAL_EVERY}" \
  --val_batches 0 \
  --eval_samples 0 \
  --num_workers 4
