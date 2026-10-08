"""Inference-time steering benchmarks for the DNA flow (paper Figure 4).

Both panels of Figure 4 were produced in May 2026 from the ``split65k`` models,
which were purged; the left panel's producer was never in the DNA repository and
the right panel's four baselines were a hard-coded CSV inside
``scripts/benchmark_yeast_split_dmfm_steering.py`` whose origin is unknown
(see ``docs/provenance/dna/README.md``). This package is the rerun decided by
the authors on 2026-09-29: both panels on the *parent-disjoint* L=50 models,
with real implementations of the baselines.

Modules
-------
``core``
    Base-flow loading, the one-forward-pass Euler / Euler-Maruyama step shared by
    every method, the cyclizability reward, terminal scoring, and the NFE counter.
``search``
    Best-of-N, Feynman-Kac steering, beam search and MCTS for the DNA flow,
    following the descriptions in Didi et al. (2026).
``dmfm_steering``
    The dMFM (Meta-Flow-Map) value-gradient guided sampler, i.e. the method the
    baselines are compared against.
``value_error``
    Panel A: |V_hat_t(x) - V_t(x)| against the number of posterior samples N,
    for the dMFM (NFE=1), the DPS denoiser approximation and a one-step flow map.
``run_steering``
    Panel B: one (method, budget, repeat) shard of the target-error-vs-NFE sweep.
``collect``
    Aggregates the shards into ``results/dna/fig4/``.
"""

from dmfm.benchmarks.core import (  # noqa: F401
    BaseFlow,
    C0Guide,
    NFECounter,
    SamplerConfig,
    cyclizability_reward,
    load_base_flow,
    load_c0,
)

__all__ = [
    "BaseFlow",
    "C0Guide",
    "NFECounter",
    "SamplerConfig",
    "cyclizability_reward",
    "load_base_flow",
    "load_c0",
]
