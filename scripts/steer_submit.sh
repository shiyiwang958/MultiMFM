#!/usr/bin/env bash
# Submit a batch of QM9 property-steering runs from a configs file.
#
#   PROP=cv [SEED=99 TSEED=7777] scripts/steer_submit.sh CONFIGS_FILE
#
# Configs file columns (whitespace separated, one run per line):
#   NAME CKPT MODE MU RS N VS VSTEPS [VBS [GMAXT [GEVERY [STEPS [CRMS [ARMS [SMT]]]]]]]
# SMT = select-min-t: earliest t whose look-ahead may be picked by steer-search.
# MODE is diag|jump. Results land in outputs/steer_${PROP}/${NAME}/summary.json.
#
# Jobs go to kempner_requeue under kempner_grads, constrained to cc8.0/cc9.0
# GPUs (newer RTX-6000-Pro nodes there have no kernels for this torch build).
# kempner_requeue is preemptible: re-submit any run whose summary.json is missing.
set -euo pipefail
cd /n/netscratch/albergo_lab/Everyone/ulrik_frank/MultiMFM_frank
CONF="${1:?configs file}"
PROP="${PROP:-alpha}"; SEED="${SEED:-8}"; TSEED="${TSEED:-2026}"
PARTITION="${PARTITION:-kempner_requeue}"; ACCOUNT="${ACCOUNT:-kempner_grads}"
CONSTRAINT="${CONSTRAINT:-cc8.0|cc9.0}"
mkdir -p "outputs/steer_${PROP}" outputs/steer_logs
while read -r NAME CK MODE MU RS N VS VSTEPS VBS GMAXT GEVERY STEPS CRMS ARMS SMT; do
  [ -z "${NAME:-}" ] && continue
  case "$NAME" in \#*) continue;; esac
  VS=${VS:-32} VSTEPS=${VSTEPS:-4} VBS=${VBS:-200} GMAXT=${GMAXT:-0.95} \
  GEVERY=${GEVERY:-1} STEPS=${STEPS:-128} CRMS=${CRMS:-0} ARMS=${ARMS:-0} \
  SELMINT=${SMT:-${SELMINT:-0.3}} DET=${DET:-0} \
  sbatch --export=ALL --job-name="st-${PROP}-${NAME}" --account="$ACCOUNT" \
    --partition="$PARTITION" --constraint="$CONSTRAINT" \
    --output="outputs/steer_logs/slurm-%x-%j.out" \
    scripts/steer_job.sh "$NAME" "$CK" "$MODE" "$MU" "$RS" "$N" "$SEED" "$TSEED" "$PROP" |
    awk -v n="$NAME" '{print $4, n}'
done < "$CONF"
