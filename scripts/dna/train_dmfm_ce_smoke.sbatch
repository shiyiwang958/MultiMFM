#!/bin/bash
#SBATCH --job-name=dna_dmfm_ce_smoke
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_grads
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# Short dMFM run used to verify the CE diagonal objective trains end to end.
#   ATOM_LOSS=ce   MAX_STEPS=400 sbatch scripts/dna/train_dmfm_ce_smoke.sbatch
#   ATOM_LOSS=velocity ...        for the baseline objective
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L="${LENGTH:-50}"; ATOM_LOSS="${ATOM_LOSS:-ce}"; MAX_STEPS="${MAX_STEPS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_NAME="${RUN_NAME:-smoke_L${L}_dmfm_${ATOM_LOSS}}"
DATA_PT="data/dna/yeast_parent_disjoint/yeast_parent_L${L}.pt"
SPLIT_PT="data/dna/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt"
TEACHER_CKPT="checkpoints/dna/base/L${L}/best.pt"
TEACHER_ARGS="checkpoints/dna/base/L${L}/args.json"
for path in "${DATA_PT}" "${SPLIT_PT}" "${TEACHER_CKPT}" "${TEACHER_ARGS}"; do
  [[ -f "${path}" ]] || { echo "Missing: ${path}" >&2; exit 2; }
done
echo "L=${L} atom_loss=${ATOM_LOSS} steps=${MAX_STEPS} batch=${BATCH_SIZE}"
${PY} -u -m dmfm.experiments.train_dmfm \
  --run_name "${RUN_NAME}" --output_dir "outputs/dna/${RUN_NAME}" --no-wandb \
  --dataset_type yeast --yeast_data_pt "${DATA_PT}" --yeast_split_pt "${SPLIT_PT}" \
  --yeast_train_split a_train --yeast_val_split a_val \
  --mode gaussian --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill --mfm_diag_atom_loss "${ATOM_LOSS}" \
  --gaussian_adaptive_loss --gaussian_adaptive_loss_c 0.01 --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt "${TEACHER_CKPT}" --mfm_teacher_ckpt_hparams "${TEACHER_ARGS}" \
  --mfm_t_cond_warmup_steps 0 --mfm_t_cond_power 1.0 --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency --mfm_consistency_type esd_teacher \
  --mfm_consistency_warmup_steps 200 --mfm_consistency_anneal_end_step 1 \
  --mfm_consistency_step_offset 0 \
  --mfm_diag_loss_weight 1.0 --mfm_lsd_loss_weight 1.0 \
  --mfm_diag_adaptive_p 0.5 --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.01 0.1 0.5 --mfm_fixed_probe_batch_size "${BATCH_SIZE}" \
  --mfm_fixed_probe_seed 12345 --mfm_checkpoint_metric val_loss \
  --model dit --hidden_dim 192 --transformer_heads 6 --transformer_ff_mult 4 \
  --dit_blocks 4 --dropout 0.0 --optimizer adam --weight_decay 0.0 --lr 1e-4 \
  --grad_clip 1.0 --batch_size "${BATCH_SIZE}" --num_workers 4 \
  --limit_val_batches 1 --val_check_interval 200 --max_steps "${MAX_STEPS}" \
  --print_freq 50 --log_every_n_steps 25 --no_tqdm
