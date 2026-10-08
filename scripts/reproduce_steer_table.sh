#!/usr/bin/env bash
# Reproduce the QM9 property-steering runs behind Table 2 (multiMFM / multiMFM-SS).
#
#   scripts/reproduce_steer_table.sh              # all 6 properties x 3 seeds = 18 jobs
#   scripts/reproduce_steer_table.sh alpha lumo   # only these properties
#
# Env overrides:
#   SEEDS="99:7777 7:101 23:555"   sample:target seed pairs (one run each)
#   DET=1                          deterministic kernels (default; see note below)
#   CONFIG=configs/steer/qm9_table2.tsv
#   PARTITION / ACCOUNT / CONSTRAINT  passed through to scripts/steer_submit.sh
#
# Each run writes outputs/steer_<prop>/T2_<prop>_s<seed>/summary.json. Afterwards:
#   python3 scripts/steer_table.py        # mean over seeds + LaTeX rows
#
# NOTE on DET=1: steering runs are NOT reproducible without it — identical configs
# and seeds otherwise differ by up to ~12% in reported MAE, because non-deterministic
# atomicAdd reductions in the autograd backward shift molecules across PoseBusters
# thresholds, changing both the valid population and its mean. DET=1 sets
# torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG and costs ~15% wall
# clock. Numbers in the paper were produced before this flag existed, so they carry
# that jitter; rerunning with DET=1 shifts individual digits but not the means.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/steer/qm9_table2.tsv}"
SEEDS="${SEEDS:-99:7777 7:101 23:555}"
DET="${DET:-1}"
CE_CKPT="checkpoints/mfm_student_ce_esd/student_step_100000.pt"
REPO_CKPT="checkpoints/mfm_student/student_step_1000.pt"

want=("$@")
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

while read -r prop ckpt mode mu rs crms arms smt steps vs vsteps n; do
  case "${prop:-}" in ""|\#*) continue;; esac
  if [ ${#want[@]} -gt 0 ]; then
    hit=0; for w in "${want[@]}"; do [ "$w" = "$prop" ] && hit=1; done
    [ $hit -eq 1 ] || continue
  fi
  case "$ckpt" in
    ce)   path="$CE_CKPT" ;;
    repo) path="$REPO_CKPT" ;;
    *)    echo "unknown ckpt '$ckpt' for $prop" >&2; exit 2 ;;
  esac
  [ -f "$path" ] || { echo "missing checkpoint: $path" >&2; exit 2; }
  for sp in $SEEDS; do
    seed="${sp%%:*}"; tseed="${sp##*:}"
    printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
      "T2_${prop}_s${seed}" "$path" "$mode" "$mu" "$rs" "$n" "$vs" "$vsteps" \
      200 0.95 1 "$steps" "$crms" "$arms" "$smt" > "$tmp/one.txt"
    # steer_submit.sh reads: NAME CKPT MODE MU RS N VS VSTEPS VBS GMAXT GEVERY STEPS CRMS ARMS SMT
    DET="$DET" PROP="$prop" SEED="$seed" TSEED="$tseed" scripts/steer_submit.sh "$tmp/one.txt"
  done
done < "$CONFIG"
