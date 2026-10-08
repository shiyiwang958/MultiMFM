"""Regeneration of the DNA results whose data was lost in the netscratch purge.

Four paper items had surviving checkpoints but no surviving producer or output
(``docs/provenance/dna/README.md``, "Flagged items"):

======================  =============================================  ==============================
paper item              module                                          status
======================  =============================================  ==============================
Fig 6 + Table 16        :mod:`dmfm.regen.base_marginals`                 authors' June recipe, rerun
Fig 7                   ``dmfm.experiments.sample_c0_guidance``          original script, rerun
Fig 8                   :mod:`dmfm.regen.guidance_trace`                 new implementation
Table 6                 ``dmfm.experiments.score_yeast_c0_evo2``         recovered scorer, rerun
======================  =============================================  ==============================

For Fig 7 and Table 6 the producing code survives (as source, and as bytecode recovered from a
transcript); for Fig 6 the producer of the published PDFs is lost but the authors' own June
re-creation of the same four panels does survive
(``results/dna/original_scripts/make_yeast_split_sample_figures.py``), and its protocol is what
:mod:`dmfm.regen.base_marginals` follows -- the Table 16 correlations on top of it are new.
Only Fig 8 has no surviving producer at all.

In every case the *inputs* are gone: the published numbers came from May/June 2026 ``split65k``
models, data and guides that were purged, so everything here runs on the parent-disjoint
artifacts shipped in ``checkpoints/dna/`` and the numbers differ by construction.

:mod:`dmfm.regen.plots` redraws every figure from the small CSVs in
``results/dna/regen/`` on CPU (used by ``notebooks/09_dna_regenerated.ipynb``).
"""

from __future__ import annotations

__all__ = ["base_marginals", "guidance_trace", "plots"]
