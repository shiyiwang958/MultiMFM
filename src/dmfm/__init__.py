"""dMFM: discrete Meta Flow Maps for DNA (yeast loop-seq) steering.

Packaged port of the DNA-MFM library (GitHub ``tullebulle/DNA-MFM@187fe7b``,
recovered from the 2026-09-25 reconstruction) plus the July 2026 scripts that
produced the DNA experiments of the paper. See ``docs/provenance/dna/README.md``
for the map from paper items to modules, scripts and checkpoints.

Subpackages
-----------
- ``dmfm.models``       DiT denoiser (base DFM) and the dMFM student
- ``dmfm.lightning``    Lightning module used to train the dMFM
- ``dmfm.utils``        interpolant/flow helpers, argument parser, data, distiller
- ``dmfm.regressors``   C0 (intrinsic cyclizability) regressors
- ``dmfm.experiments``  one module per experiment, each runnable with
                        ``python -m dmfm.experiments.<name>``
- ``dmfm.paths``        repo-relative default locations of checkpoints and data

Importing ``dmfm`` itself is cheap (no torch import).
"""

__all__ = ["paths"]
