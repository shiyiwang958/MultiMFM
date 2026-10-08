"""Minimal vendored pieces of the upstream MFM library used by the DNA code.

The DNA-MFM repo imported ``mfm`` from a vendored copy (``mfm/src/mfm``) via a
``sys.path`` hack and used only four objects from it:

- ``mfm.models.base_model.BaseModel``            -> :class:`dmfm._mfm.base_model.BaseModel`
- ``mfm.models.base_model.LossWeightingNetwork`` -> :class:`dmfm._mfm.base_model.LossWeightingNetwork`
- ``mfm.losses.losses.extract_posterior_velocity`` -> :func:`dmfm._mfm.losses.extract_posterior_velocity`
- ``mfm.losses.utils.compute_loss``              -> :func:`dmfm._mfm.losses.compute_loss`

They are copied verbatim (apart from imports) from the copy vendored in
``scratch_dfm_recon_20260925/mfm/src/mfm`` (the DNA-MFM@187fe7b vendored MFM).
That copy differs from the repo's top-level ``mfm/`` only in
``extract_posterior_velocity``: it passes ``(t_cond, xt_cond)`` to the teacher's
``v`` where the top-level copy passes zeros. For the DNA teacher
(:class:`dmfm.models.dna_models.DNAMFMTeacherAdapter`) both are identical because
its ``v`` ignores ``t_cond``/``x_cond``. Vendoring here keeps the DNA code
independent of the top-level ``mfm/`` (and of ``timm``, which ``mfm.models``
pulls in through ``mfm.models.dit``).
"""

from .base_model import BaseModel, LossWeightingNetwork
from .losses import compute_loss, extract_posterior_velocity

__all__ = ["BaseModel", "LossWeightingNetwork", "compute_loss", "extract_posterior_velocity"]
