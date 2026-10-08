# Third-party components

MultiMFM builds on and vendors code from the following projects. All are used
under their original licenses; copyright remains with their respective authors.

## TABASCO (`src/tabasco/`)
The base flow-matching backbone, GLASS posterior sampler, and meta-flow-map
transformer are from TABASCO — a transformer-based flow-matching model for 3D
molecule generation.
- License: MIT (Copyright (c) 2025 Carlos Vonessen)
- Paper: `tabasco_paper.pdf` in the original repository.

## TFG-Flow (`third_party/tfg_flow/`)
The property **guide** and **oracle** regressors (`Predictor` = EGNN head) are
from TFG-Flow: *Training-Free Guidance in Multi-Modal Generative Flow*
(Lin, Li, Ye, Yang, Ermon, Liang, Ma — ICLR 2025).
- License: MIT
- Only the subset needed to instantiate and run the regressors is vendored
  (`diffusion/predictor.py`, `networks/egnn.py`, `dataloader/constants.py`).
- Trained regressor weights: `checkpoints/regressors/{guide,oracle}_clf_ckpt.zip`.

## MFM reference (`mfm/src/`)
The GLASS posterior-velocity distillation target used to train the meta flow map
is imported from the reference Meta Flow Map implementation (Potaptchik, Albergo
et al.). Vendored so that `tabasco.sample.glass.extract_glass_velocity` resolves
`mfm.losses.extract_posterior_velocity` at training time.

## QM9 (`data/qm9/`)
QM9 molecular dataset, loaded via the Hugging Face dataset `yairschiff/qm9`.
