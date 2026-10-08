"""Load DNA base-DFM teachers, dMFM students and the parent-disjoint sequences.

These loaders were defined in ``dirichlet-flow-matching/scripts/probe_dna_mfm_glass.py``
and imported by most of the DNA experiment scripts (``from probe_dna_mfm_glass
import load_args_json, load_student, ...``). They are unchanged apart from the
package imports and ``torch.load(weights_only=False)``; the extra module-level
helper :func:`length_defaults` returns the repo-relative default paths for one
sequence length.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel, DNAMFMStudent, DNAMFMTeacherAdapter
from dmfm.utils.esm import upgrade_state_dict
from dmfm.utils.torch_io import torch_load
from dmfm.utils.yeast_splits import load_yeast_split_indices


def length_defaults(length: int) -> dict:
    """Repo-relative default artifact paths for one sequence length."""
    return {
        "teacher_ckpt": paths.base_ckpt(length),
        "teacher_args": paths.base_args(length),
        "dmfm_ckpt": paths.dmfm_ckpt(length),
        "dmfm_args": paths.dmfm_args(length),
        "dmfm4_ckpt": paths.dmfm4_ckpt(length),
        "dmfm4_args": paths.dmfm4_args(length),
        "data_pt": paths.data_pt(length),
        "split_pt": paths.split_pt(length),
    }


def _as_namespace(d: dict) -> SimpleNamespace:
    d = deepcopy(d)
    if d.get("dit_cond_dim") is None:
        d["dit_cond_dim"] = d.get("hidden_dim", 192)
    if d.get("cond_dim") is None:
        d["cond_dim"] = d["dit_cond_dim"]
    defaults = {
        "model": "dit",
        "mode": "gaussian",
        "clean_data": False,
        "self_condition_ratio": 0.0,
        "time_delta_param": True,
        "double_temb": True,
        "scale_by_sigma": True,
        "preserve_denoiser": False,
        "use_flash_attn": False,
        "softcap": 50.0,
        "dropout": 0.0,
        "gaussian_beta_schedule": "linear",
        "gaussian_beta_table_path": None,
        "gaussian_adaptive_loss_c": 0.01,
        "mfm_preserve_t_cond_0": True,
        "mfm_encoder_depth": None,
    }
    for k, v in defaults.items():
        if d.get(k) is None:
            d[k] = v
    return SimpleNamespace(**d)


def load_args_json(path: str | Path) -> SimpleNamespace:
    with open(path) as f:
        payload = json.load(f)
    return _as_namespace(payload["args"] if isinstance(payload, dict) and "args" in payload else payload)


def _state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch_load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "student_state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _load_flexible_state(model: torch.nn.Module, path: str | Path, *, prefixes=("model.", "student.")):
    raw = _state_dict_from_checkpoint(path)
    raw = {k: v for k, v in raw.items() if "cls_model" not in k and "distill_model" not in k}
    candidates = [raw]
    for prefix in prefixes:
        candidates.append(upgrade_state_dict(raw, prefixes=[prefix]))

    best = None
    best_score = -1
    model_keys = set(model.state_dict())
    for sd in candidates:
        score = len(model_keys.intersection(sd.keys()))
        if score > best_score:
            best = sd
            best_score = score
    return model.load_state_dict(best, strict=False)


def load_teacher(ckpt_path: str | Path, args_json: str | Path, *, alphabet_size: int, device):
    args = load_args_json(args_json)
    teacher = DiTSequenceModel(args, alphabet_size=alphabet_size)
    incompat = _load_flexible_state(teacher, ckpt_path, prefixes=("model.",))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    wrapper = DNAMFMTeacherAdapter(teacher).eval().to(device)
    for p in wrapper.parameters():
        p.requires_grad = False
    return args, teacher, wrapper, incompat


def load_student(
    args,
    *,
    alphabet_size: int,
    device,
    student_ckpt: str | Path | None = None,
    teacher: DiTSequenceModel | None = None,
):
    student = DNAMFMStudent(args, alphabet_size=alphabet_size).to(device)
    if student_ckpt is None:
        if teacher is not None:
            student.warm_start_from_denoiser_teacher(teacher)
        incompat = None
    else:
        incompat = _load_flexible_state(student, student_ckpt)
    student.eval()
    return student, incompat


def load_data_seqs(
    path: str | Path,
    *,
    max_n: int | None = None,
    split_pt: str | Path | None = None,
    split: str = "full",
) -> torch.Tensor:
    obj = torch_load(path, map_location="cpu")
    if isinstance(obj, dict):
        seqs = obj.get("seqs", None)
        if seqs is None:
            raise KeyError(f"{path} is a dict but has no 'seqs' key")
    else:
        seqs = obj
    seqs = seqs.long()
    if split_pt and split != "full":
        seqs = seqs[load_yeast_split_indices(split_pt, split)]
    if max_n is not None:
        seqs = seqs[: int(max_n)]
    return seqs
