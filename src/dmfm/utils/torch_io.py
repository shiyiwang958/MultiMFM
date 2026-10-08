"""Small I/O helpers shared by the DNA code."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def torch_load(path, map_location="cpu") -> Any:
    """``torch.load`` with ``weights_only=False``.

    The DNA checkpoints (Lightning ``.ckpt`` with pickled ``argparse.Namespace``
    hyper-parameters, standalone ``best.pt`` dicts with ``args``/``model_cfg``)
    are trusted local files written by this code base. torch>=2.6 defaults to
    ``weights_only=True``, which rejects them; the paper runs used torch 2.1
    whose default was the full unpickler, so this restores that behaviour.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def as_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())
