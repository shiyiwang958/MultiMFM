# QM9 property-steering configs (Table 2)

`qm9_table2.tsv` holds the final per-property steering configuration behind the
`multiMFM` and `multiMFM-SS` rows of Table 2, one line per property.

## Reproduce

```bash
source scripts/env.sh
scripts/reproduce_steer_table.sh                 # 6 properties x 3 seeds = 18 jobs
python3 scripts/steer_table.py                   # per-property detail + LaTeX rows
python3 scripts/steer_table.py --std             # same, with +/- SD over seeds
```

Single property, or custom seeds:

```bash
scripts/reproduce_steer_table.sh lumo
SEEDS="99:7777" scripts/reproduce_steer_table.sh alpha
```

Jobs go to `kempner_requeue` under `kempner_grads` (override with `PARTITION` /
`ACCOUNT`), constrained to `cc8.0|cc9.0` GPUs — some requeue nodes carry GPUs this
torch build has no kernels for. Each run is ~4-8 min; results land in
`outputs/steer_<prop>/T2_<prop>_s<seed>/summary.json`.

## What the two rows mean

Both come from the *same* run. The guided trajectory exposes a clean-sample
look-ahead at every outer step; `multiMFM` reports the final trajectory endpoint,
and `multiMFM-SS` reports the PoseBusters-valid look-ahead whose **guide**-predicted
property is closest to target (steer-search). Both are then scored with the
held-out **oracle** regressor, which never participates in steering.

Reported MAE is over PoseBusters-valid molecules only, matching the paper. Note the
two rows therefore average over *different* populations: steer-search rescues
molecules whose final endpoint failed validity (PB ~0.97-0.99 vs ~0.91), and those
are harder cases, so its MAE carries a handicap.

## Two parameters that matter more than they look

- **`SMT` (`--select-min-t`)** — earliest t whose look-ahead steer-search may pick.
  This is what makes steer-search beat the endpoint: restricting selection to late,
  well-converged look-aheads offsets the population handicap above. The configs
  here use 0.5-0.9; at 0.3 steer-search did not consistently beat the endpoint.
- **`DET=1`** (default in `reproduce_steer_table.sh`) — without it runs are not
  reproducible: identical configs and seeds differ by up to ~12% in reported MAE,
  enough to flip the sign of (steer-search MAE - endpoint MAE). Cause is
  non-deterministic atomicAdd reductions in the autograd backward; tiny float
  differences move molecules across PoseBusters thresholds, changing both the valid
  population and its mean. Costs ~15% wall clock.

Determinism gives repeatability, not robustness — hence the three seed pairs. Quote
means over seeds, not single runs.

## Checkpoints

| `ckpt` | Path | Notes |
|---|---|---|
| `repo` | `checkpoints/mfm_student/student_step_1000.pt` | shipped diagonal-only student; sampled with 4 diagonal Euler steps (`diag`), the mode it was trained for |
| `ce` | `checkpoints/mfm_student_ce_esd/student_step_100000.pt` | diagonal-CE + ESD student, same size as the teacher (hidden 64 / 4 layers / 4 heads, 363k params), trained with `--time-encoding-max-len 4`; sampled with 4 flow-map jumps (`jump`) |

Both are the same size; the per-property choice in `qm9_table2.tsv` is whichever won
at matched validity. The base flow (`checkpoints/base_flow/model_step_100000.pt`)
supplies the trajectory velocity in every case — the MFM student is used only to
draw the posterior look-aheads for the value gradient.
