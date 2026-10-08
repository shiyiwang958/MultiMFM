"""Load a trained TABASCO flow-matching model from a ``model_step_*.pt`` checkpoint.

Extracted from the original ``pb_failure_diagnostics.py`` — the only thing the
steering/eval pipeline needs from it is this loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multimfm.train_base_flow import build_model


def load_lightweight_model(checkpoint_path: Path, device: torch.device):
    """Rebuild a ``FlowMatchingModel`` from a checkpoint and move it to ``device``.

    The checkpoint is the format saved by ``train_base_flow.py``: a dict with
    ``args`` (the training namespace), ``data_stats`` (atom names, normalizer,
    max atoms, ...), and ``model`` (the state dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = argparse.Namespace(**checkpoint["args"])
    data_stats = checkpoint["data_stats"]
    model = build_model(model_args, data_stats)
    model.load_state_dict(checkpoint["model"])
    model.set_data_stats(data_stats)
    model = model.to(device).eval()
    return model, data_stats, model_args
