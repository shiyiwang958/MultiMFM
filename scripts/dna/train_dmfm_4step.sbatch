#!/bin/bash
#SBATCH --job-name=dna_train_dmfm4
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# 4-step dMFM student (Table 18): continue the diagonal dMFM with ESD on short
# gaps, delta ~ U(0, 0.25]. Original: slurm/train_yeast_parent_dmfm_4step_continue.sbatch,
# jobs 35458567-70 (kempner_h100). Arguments identical, and --init_ckpt points at
# the shipped diagonal dMFM of the same length.
#
# The gap bug matters here: the earlier jobs 35409636-39 used the DNA-MFM@187fe7b
# gap sampling, where min_gap=0 silently became 0.75 (mean gap 0.50 instead of
# 0.125). The shipped students come from the corrected July code, which is the
# default in dmfm; pass --mfm_legacy_gap_or_default to get the buggy behaviour.
#
#   LENGTH=50 sbatch scripts/dna/train_dmfm_4step.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L="${LENGTH:?Set LENGTH to one of 50, 100, 200, 400}"
case "${L}" in
  50|100) BATCH_SIZE="${BATCH_SIZE:-64}" ;;
  200)    BATCH_SIZE="${BATCH_SIZE:-32}" ;;
  400)    BATCH_SIZE="${BATCH_SIZE:-16}" ;;
  *) echo "Unsupported LENGTH=${L}" >&2; exit 2 ;;
esac

DATA_PT="${DATA_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L${L}.pt}"
SPLIT_PT="${SPLIT_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/dna/base/L${L}/best.pt}"
TEACHER_ARGS="${TEACHER_ARGS:-checkpoints/dna/base/L${L}/args.json}"
# Shipped diagonal dMFM of this length (the same file the original --init_ckpt used).
INIT_CKPT="${INIT_CKPT:-$(${PY} -c "from dmfm import paths; print(paths.dmfm_ckpt(${L}))")}"
MAX_STEPS="${MAX_STEPS:-150000}"
LR="${LR:-2e-5}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
RUN_NAME="${RUN_NAME:-yeast_parent_a_L${L}_dmfm_4step_esd_gap_upto025_corrected_20260727}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dna/${RUN_NAME}}"

for path in "${DATA_PT}" "${SPLIT_PT}" "${TEACHER_CKPT}" "${TEACHER_ARGS}" "${INIT_CKPT}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

echo "L=${L}; batch=${BATCH_SIZE}; steps=${MAX_STEPS}; lr=${LR}; gap=(0,0.25]; init=${INIT_CKPT}"
${PY} -u -m dmfm.experiments.train_dmfm \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_ckpt "${INIT_CKPT}" \
  --no-wandb \
  --dataset_type yeast \
  --yeast_data_pt "${DATA_PT}" \
  --yeast_split_pt "${SPLIT_PT}" \
  --yeast_train_split a_train \
  --yeast_val_split a_val \
  --mode gaussian \
  --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill \
  --gaussian_adaptive_loss \
  --gaussian_adaptive_loss_c 0.01 \
  --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt "${TEACHER_CKPT}" \
  --mfm_teacher_ckpt_hparams "${TEACHER_ARGS}" \
  --mfm_t_cond_warmup_steps 0 \
  --mfm_t_cond_power 1.0 \
  --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency \
  --mfm_consistency_type esd_teacher \
  --mfm_consistency_warmup_steps 0 \
  --mfm_consistency_gap_mode long \
  --mfm_consistency_min_gap 0.0 \
  --mfm_consistency_max_gap 0.25 \
  --mfm_diag_loss_weight 1.0 \
  --mfm_lsd_loss_weight 1.0 \
  --mfm_diag_adaptive_p 0.5 \
  --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.0625 0.125 0.25 \
  --mfm_fixed_probe_batch_size "${BATCH_SIZE}" \
  --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric val_mfm_probe_esd_gap_0.25 \
  --model dit \
  --hidden_dim 192 \
  --transformer_heads 6 \
  --transformer_ff_mult 4 \
  --dit_blocks 4 \
  --dropout 0.0 \
  --optimizer adam \
  --weight_decay 0.0 \
  --lr "${LR}" \
  --grad_clip 1.0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 4 \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --max_steps "${MAX_STEPS}" \
  --print_freq 200 \
  --log_every_n_steps 50 \
  --no_tqdm
