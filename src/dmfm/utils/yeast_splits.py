"""Named yeast split helpers (ported from DNA-MFM@187fe7b ``utils/yeast_splits.py``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from dmfm.utils.torch_io import torch_load


def load_yeast_split_indices(split_pt: str | Path, split: str) -> torch.Tensor:
    """Load one named yeast split from a torch-saved split file."""
    payload: dict[str, Any] = torch_load(split_pt, map_location="cpu")
    key = f"{split}_idx"
    if key not in payload:
        available = sorted(k[:-4] for k in payload if k.endswith("_idx"))
        raise KeyError(f"Split {split!r} not found in {split_pt}. Available: {available}")
    idx = payload[key]
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    idx = idx.long().cpu()
    if idx.ndim != 1:
        raise ValueError(f"Expected 1D indices for {key}, got shape {tuple(idx.shape)}")
    return idx


def subset_by_yeast_split(dataset, split_pt: str | Path | None, split: str):
    """Return dataset or Subset(dataset, split indices) if split_pt is set."""
    if not split_pt:
        return dataset
    if split == "full":
        return dataset
    return Subset(dataset, load_yeast_split_indices(split_pt, split).tolist())
