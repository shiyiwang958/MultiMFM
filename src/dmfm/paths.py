"""Repo-relative default locations of the DNA checkpoints, data and outputs.

All defaults in ``dmfm`` resolve through this module, so nothing depends on
absolute cluster paths. The large artifacts are not in git; fetch them with

    python scripts/manifest.py verify --group dna --fetch-missing

Layout (mirrors ``checkpoints/MANIFEST.json``, group ``dna``)::

    checkpoints/dna/base/L{50,100,200,400}/{best.pt,args.json}         base DFM DiT
    checkpoints/dna/dmfm/L{..}/{<epoch=..>.ckpt,args.json}               dMFM (diag+ESD)
    checkpoints/dna/dmfm_4step/L{..}/{<epoch=..>.ckpt,args.json}         4-step students
    checkpoints/dna/c0/{guide,oracle}/{best_state.pt,metadata.json}      C0 regressors
    data/dna/yeast_parent_disjoint/yeast_parent_L{..}{,_split_seed0}.pt  data + splits

Original locations (``dirichlet-flow-matching/workdir/...``) are recorded in
``docs/provenance/dna/checkpoints.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/dmfm/paths.py -> repo root is two levels above src/.
REPO_ROOT = Path(os.environ.get("MULTIMFM_ROOT", Path(__file__).resolve().parents[2]))
CHECKPOINTS = REPO_ROOT / "checkpoints" / "dna"
DATA = REPO_ROOT / "data" / "dna"
RESULTS = REPO_ROOT / "results" / "dna"
OUTPUTS = REPO_ROOT / "outputs" / "dna"

LENGTHS = (50, 100, 200, 400)

# Checkpoint files selected for the paper (see docs/provenance/dna/checkpoints.md).
_DMFM_FILES = {
    50: "epoch=95-step=49000.ckpt",
    100: "epoch=80-step=39000.ckpt",
    200: "epoch=51-step=45000.ckpt",
    400: "epoch=33-step=45750-v1.ckpt",  # from yeast_parent_a_L400_dmfm_diagonly_finetune_20260726
}
_DMFM4_FILES = {
    50: "epoch=29-step=15000.ckpt",
    100: "epoch=306-step=149000.ckpt",
    200: "epoch=39-step=34000.ckpt",
    400: "epoch=96-step=129000.ckpt",
}


def _check_length(length: int) -> int:
    length = int(length)
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {LENGTHS}, got {length}")
    return length


def data_pt(length: int) -> Path:
    """Integer-encoded windows ``{'seqs': LongTensor[N, L], ...}`` for one length."""
    return DATA / "yeast_parent_disjoint" / f"yeast_parent_L{_check_length(length)}.pt"


def split_pt(length: int) -> Path:
    """Parent-disjoint split indices (``a_train_idx``, ``a_val_idx``, ``test_idx``, ...)."""
    return DATA / "yeast_parent_disjoint" / f"yeast_parent_L{_check_length(length)}_split_seed0.pt"


def split_json() -> Path:
    return DATA / "yeast_parent_disjoint" / "parent_split_seed0.json"


def base_ckpt(length: int) -> Path:
    return CHECKPOINTS / "base" / f"L{_check_length(length)}" / "best.pt"


def base_args(length: int) -> Path:
    return CHECKPOINTS / "base" / f"L{_check_length(length)}" / "args.json"


def dmfm_ckpt(length: int) -> Path:
    length = _check_length(length)
    return CHECKPOINTS / "dmfm" / f"L{length}" / _DMFM_FILES[length]


def dmfm_args(length: int) -> Path:
    return CHECKPOINTS / "dmfm" / f"L{_check_length(length)}" / "args.json"


def dmfm4_ckpt(length: int) -> Path:
    length = _check_length(length)
    return CHECKPOINTS / "dmfm_4step" / f"L{length}" / _DMFM4_FILES[length]


def dmfm4_args(length: int) -> Path:
    return CHECKPOINTS / "dmfm_4step" / f"L{_check_length(length)}" / "args.json"


#: Which dMFM artifact each posterior step count was distilled for. The ``dmfm`` students
#: were trained with ESD on *random* two-time gaps (mean 1/3) and are selected on
#: ``val_loss``; Tables 13 and 17 use them as **one-step** maps (gap 1.0). The
#: ``dmfm_4step`` students continue from those weights with ``gap_mode=long`` over
#: ``[0, 0.25]`` and are selected on ``val_mfm_probe_esd_gap_0.25``, i.e. specialised for
#: the jump a 4-step composition takes; Table 18 uses them at 4 steps. Composing the
#: ``dmfm`` student in 4 steps mixes the two (see ``docs/dmfm_glass_parity.md``).
_STUDENT_FOR_STEPS = {1: "dmfm"}


def dmfm_student_kind(n_steps: int) -> str:
    """``"dmfm"`` for a one-step posterior map, ``"dmfm4"`` for any few-step composition."""
    return _STUDENT_FOR_STEPS.get(int(n_steps), "dmfm4")


def dmfm_ckpt_for_steps(length: int, n_steps: int) -> Path:
    """dMFM checkpoint distilled for a posterior sampler that composes ``n_steps`` maps."""
    kind = dmfm_student_kind(n_steps)
    return dmfm_ckpt(length) if kind == "dmfm" else dmfm4_ckpt(length)


def dmfm_args_for_steps(length: int, n_steps: int) -> Path:
    kind = dmfm_student_kind(n_steps)
    return dmfm_args(length) if kind == "dmfm" else dmfm4_args(length)


def dmfm_l400_diagonal_ckpt() -> Path:
    """Resume point of the L400 diagonal-only finetune (not loaded by any result run)."""
    return CHECKPOINTS / "dmfm_lineage" / "L400_diagonal" / "epoch=33-step=45000.ckpt"


def c0_guide() -> Path:
    return CHECKPOINTS / "c0" / "guide" / "best_state.pt"


def c0_oracle() -> Path:
    return CHECKPOINTS / "c0" / "oracle" / "best_state.pt"


def rel(path: str | os.PathLike) -> str:
    """Path relative to the repo root when possible (for logs/metadata)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
