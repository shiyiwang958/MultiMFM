"""Scale-free Monte Carlo gradient analysis with **dMFM** finite-N estimates.

Fig 5 and Tables 14-15 of the submission claim to measure the Monte Carlo error of
*dMFM* value-gradient estimates against an independent GLASS-2048 reference. The
published runs (``workdir/rebuttal_gradient_accuracy_{gc_pairmetrics,motif,conjunction}``
via ``dmfm.experiments.ablate_glass_gradient_mc``) instead used the GLASS posterior on
the base DFM for *both* sides: only the number of terminal-noise samples differed. This
package reruns the same protocol with the dMFM posterior sampler on the finite-N side,
so the figure measures what its caption claims.

Everything except the finite-N estimator matches the published run bit for bit: the same
three rewards, the same four lengths, the same seed and seed arithmetic, the same 32
conditioning states per length (base flow integrated from Gaussian noise to ``t=0.5``
with 32 NFE), the same 8 noise pools per state, the same ``N`` grid, the same
GLASS-2048 reference (8 Euler steps, ``end_time=1.0``) and the same metric code
(:mod:`dmfm.experiments.rescore_gradient_pairs_scale_free` and
:mod:`dmfm.experiments.summarize_gradient_accuracy_metrics`, both reused unchanged).

Entry points::

    python -m dmfm.scalefree.run_shard --length 50 --reward motif --student dmfm
    python -m dmfm.scalefree.collect            # aggregate shards -> results/
    python -m dmfm.scalefree.verify_reference   # recomputed vs published GLASS-2048

See ``docs/provenance/dna/scalefree.md``.
"""

from __future__ import annotations

STUDENT_TAGS = {"dmfm": "diag1step", "dmfm4": "dmfm4_1step"}
REWARDS = ("gc", "motif", "conjunction")
LENGTHS = (50, 100, 200, 400)
# The published sweep; Table 14 reports N = 8 and 128, Fig 5 plots 1, 4, 16, 64, 256.
MC_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TABLE14_MC = (8, 128)
FIG5_MC = (1, 4, 16, 64, 256)

__all__ = ["STUDENT_TAGS", "REWARDS", "LENGTHS", "MC_VALUES", "TABLE14_MC", "FIG5_MC"]
