#!/usr/bin/env python
"""Figure 4 (right) -- entry point for the matched-NFE steering benchmark.

History
-------
The submitted panel came from ``dirichlet-flow-matching/scripts/
benchmark_yeast_split_dmfm_steering.py`` (May 2026). That script ran only the
*dMFM* half of the figure; its four baseline curves were the hard-coded CSV
reproduced below as :data:`OLD_NON_DMFM_BASELINES`, whose producer never existed
in the DNA repository and whose provenance is unknown. Both halves used the
``split65k`` models, which the netscratch purge deleted, and the old Keras-derived
C0 guide ``dinko/C0free_torch.pt``, also deleted. The run directory
``workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45`` survives empty.

The authors decided on 2026-09-29 to rerun both panels on the parent-disjoint
models with *real* implementations of best-of-N, Feynman-Kac steering, beam search
and MCTS. That rerun is :mod:`dmfm.benchmarks`, and this module is its CLI:

    python -m dmfm.experiments.benchmark_nfe_steering --mode list --plan sweep
    python -m dmfm.experiments.benchmark_nfe_steering --plan sweep --index 3 --repeat 0
    python -m dmfm.experiments.benchmark_nfe_steering --mode calibrate --n-outputs 512

Run the whole thing with ``sbatch scripts/dna/fig4_panelB.sbatch`` and collect with
``bash scripts/dna/fig4_collect.sh``. Figure 4 (left) is
``python -m dmfm.benchmarks.value_error``.

The old dMFM-only sweep is not kept: its settings (target C0 0.30, 96 sampling
steps, MC in {1,2,4,8,16}, NFE_value 1, guidance on t in [0.01, 0.95],
guidance_frac 8, cap 10, clip 10, reward_sigma 0.15) are the defaults of the new
CLI and are recorded in ``results/dna/benchmarks/benchmark_script_defaults.json``.
"""

from __future__ import annotations

from dmfm.benchmarks.run_steering import main, parse_args  # noqa: F401

# The baselines of the submitted panel, verbatim from lines 28-48 of the original
# script. Kept here only as provenance -- they are NOT used by any plot any more;
# see results/dna/benchmarks/fig4_right_hardcoded_baselines.csv and
# docs/provenance/dna/README.md.
OLD_NON_DMFM_BASELINES = """method,config,target_mc,gen_nfe_per_output,reward_evals_per_output,mae_mean,mae_sem,frac05_mean,frac10_mean,successes_seen,n_repeats
beam,"L=1,K=48,beta=0.0",1,240.0,3.0,0.521709,0.000000,0.040000,0.050000,13.0,3
best_of_N,"N=2",1,192.0,2.0,0.378953,0.000000,0.060000,0.140000,,3
fk,"L=1,K=48,beta=5.0",1,240.0,3.0,0.310291,0.000000,0.100000,0.100000,10.0,3
mcts,"M=1,K=48,C=0.1,children=4,best_rollout",1,240.0,2.0,0.395437,0.000000,0.090000,0.150000,15.0,3
beam,"L=1,K=32,beta=0.0",2,288.0,4.0,0.500254,0.000000,0.050000,0.070000,19.0,3
best_of_N,"N=3",2,288.0,3.0,0.293011,0.000000,0.100000,0.200000,,3
fk,"L=1,K=32,beta=5.0",2,288.0,4.0,0.210982,0.000000,0.160000,0.270000,23.0,3
mcts,"M=1,K=32,C=0.1,children=4,best_rollout",2,288.0,3.0,0.321696,0.000000,0.130000,0.200000,21.0,3
beam,"L=2,K=48,beta=0.0",4,480.0,5.0,0.332530,0.000000,0.060000,0.130000,24.0,3
best_of_N,"N=5",4,480.0,5.0,0.207317,0.000000,0.140000,0.320000,,3
fk,"L=2,K=48,beta=5.0",4,480.0,5.0,0.223329,0.000000,0.120000,0.360000,26.0,3
mcts,"M=2,K=48,C=0.1,children=4,best_rollout",4,480.0,4.0,0.252176,0.000000,0.200000,0.260000,31.0,3
beam,"L=2,K=16,beta=0.0",8,864.0,13.0,0.153788,0.000000,0.180000,0.280000,102.0,3
best_of_N,"N=9",8,864.0,9.0,0.120121,0.000000,0.260000,0.530000,,3
fk,"L=2,K=16,beta=5.0",8,864.0,13.0,0.124627,0.000000,0.440000,0.560000,158.0,3
mcts,"M=2,K=16,C=0.1,children=4,best_rollout",8,864.0,12.0,0.131805,0.000000,0.240000,0.490000,116.0,3
beam,"L=3,K=12,beta=0.0",16,1584.0,25.0,0.037757,0.000000,0.660000,1.000000,413.0,3
best_of_N,"N=16",16,1536.0,16.0,0.078966,0.000000,0.440000,0.720000,,3
fk,"L=3,K=12,beta=5.0",16,1584.0,25.0,0.121939,0.000000,0.310000,0.570000,340.0,3
mcts,"M=2,K=8,C=0.1,children=4,best_rollout",16,1442.0,24.0,0.085869,0.000000,0.450000,0.700000,234.0,3
"""


if __name__ == "__main__":
    main()
