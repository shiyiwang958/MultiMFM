"""GLASS posterior velocity utilities for TABASCO DFM models."""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from typing import Callable

import torch
from tensordict import TensorDict
from torch import Tensor

from tabasco.flow.interpolate import GaussianSimplexAtomInterpolant
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


def _broadcast_time(t: Tensor, target: Tensor) -> Tensor:
    return t.view(-1, *((1,) * (target.ndim - 1)))


def _import_mfm_extract_posterior_velocity() -> Callable:
    """Import MFM's GLASS posterior-velocity helper from the vendored folder."""
    repo_root = Path(__file__).resolve().parents[3]
    mfm_src = repo_root / "mfm" / "src"
    if not mfm_src.exists():
        from mfm.losses import extract_posterior_velocity

        return extract_posterior_velocity

    # Avoid importing mfm.losses.__init__, which also imports ImageNet-specific
    # finetuning dependencies. We only need losses.py and its utils.py import.
    for module_name in ("mfm.losses.losses", "mfm.losses.utils"):
        sys.modules.pop(module_name, None)

    mfm_pkg = sys.modules.get("mfm")
    if mfm_pkg is None:
        mfm_pkg = types.ModuleType("mfm")
        mfm_pkg.__path__ = [str(mfm_src / "mfm")]
        sys.modules["mfm"] = mfm_pkg

    losses_pkg = types.ModuleType("mfm.losses")
    losses_pkg.__path__ = [str(mfm_src / "mfm" / "losses")]
    sys.modules["mfm.losses"] = losses_pkg

    utils_path = mfm_src / "mfm" / "losses" / "utils.py"
    utils_spec = importlib.util.spec_from_file_location("mfm.losses.utils", utils_path)
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError(f"Could not load MFM losses utils from {utils_path}")
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["mfm.losses.utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    losses_path = mfm_src / "mfm" / "losses" / "losses.py"
    losses_spec = importlib.util.spec_from_file_location(
        "mfm.losses.losses", losses_path
    )
    if losses_spec is None or losses_spec.loader is None:
        raise ImportError(f"Could not load MFM losses module from {losses_path}")
    losses_module = importlib.util.module_from_spec(losses_spec)
    sys.modules["mfm.losses.losses"] = losses_module
    losses_spec.loader.exec_module(losses_module)
    return losses_module.extract_posterior_velocity


class MultimodalStateCodec:
    """Flatten/unflatten TABASCO coordinate and atom-flow states."""

    def __init__(self, padding_mask: Tensor):
        self.padding_mask = padding_mask.bool()
        self.batch_size, self.num_atoms = padding_mask.shape
        self.flat_mask: Tensor | None = None

    def _build_flat_mask(self, coords: Tensor, atomics: Tensor) -> Tensor:
        coord_mask = (~self.padding_mask).unsqueeze(-1).expand_as(coords)
        atom_mask = (~self.padding_mask).unsqueeze(-1).expand_as(atomics)
        return torch.cat(
            [
                coord_mask.reshape(self.batch_size, -1),
                atom_mask.reshape(self.batch_size, -1),
            ],
            dim=-1,
        ).to(dtype=coords.dtype, device=coords.device)

    def flatten_state(self, state: TensorDict) -> Tensor:
        coords = mask_and_zero_com(state["coords"], self.padding_mask)
        atomics = apply_mask(state["atomics"].float(), self.padding_mask)
        self.flat_mask = self._build_flat_mask(coords, atomics)
        flat = torch.cat(
            [
                coords.reshape(coords.shape[0], -1),
                atomics.reshape(atomics.shape[0], -1),
            ],
            dim=-1,
        )
        return flat * self.flat_mask

    def flatten_endpoint_prediction(self, pred: TensorDict) -> Tensor:
        coords = mask_and_zero_com(pred["coords"], self.padding_mask)
        atomics = torch.softmax(pred["atomics"], dim=-1)
        atomics = apply_mask(atomics, self.padding_mask)
        flat = torch.cat(
            [
                coords.reshape(coords.shape[0], -1),
                atomics.reshape(atomics.shape[0], -1),
            ],
            dim=-1,
        )
        if self.flat_mask is None:
            self.flat_mask = self._build_flat_mask(coords, atomics)
        return flat * self.flat_mask

    def unflatten_state(self, flat: Tensor, atom_dim: int) -> TensorDict:
        coord_dim = self.num_atoms * 3
        coords = flat[:, :coord_dim].reshape(self.batch_size, self.num_atoms, 3)
        atomics = flat[:, coord_dim:].reshape(self.batch_size, self.num_atoms, atom_dim)
        return TensorDict(
            {
                "coords": mask_and_zero_com(coords, self.padding_mask),
                "atomics": apply_mask(atomics, self.padding_mask),
                "padding_mask": self.padding_mask,
            },
            batch_size=self.batch_size,
        ).to(flat.device)


class TabascoDiagonalVelocityAdapter:
    """MFM-compatible diagonal velocity wrapper around a TABASCO DFM model."""

    def __init__(self, model, codec: MultimodalStateCodec, eps: float = 1e-5):
        self.model = model
        self.codec = codec
        self.eps = eps
        self.atom_dim = model.data_stats["atom_dim"]
        self.last_atom_logits: Tensor | None = None
        self.num_calls = 0

    def v(self, s, t, x, t_cond=None, x_cond=None, class_labels=None, **kwargs):
        del t, t_cond, x_cond, class_labels, kwargs
        state = self.codec.unflatten_state(x, self.atom_dim)
        pred = self.model._call_net(state, s)
        self.last_atom_logits = pred["atomics"]
        self.num_calls += 1
        endpoint = self.codec.flatten_endpoint_prediction(pred)
        denom = _broadcast_time(1.0 - s, x).clamp_min(self.eps)
        velocity = (endpoint - x) / denom
        if self.codec.flat_mask is not None:
            velocity = velocity * self.codec.flat_mask
        return velocity


def require_linear_dfm_atoms(model) -> None:
    atomics = model.atomics_interpolant
    if not isinstance(atomics, GaussianSimplexAtomInterpolant):
        raise TypeError(
            "GLASS posterior velocity is currently implemented for "
            "GaussianSimplexAtomInterpolant DFM atom states only."
        )
    if atomics.beta_schedule != "linear" or float(atomics.beta_power) != 1.0:
        raise ValueError(
            "GLASS posterior velocity assumes linear atom beta(t)=t. "
            f"Got beta_schedule={atomics.beta_schedule!r}, "
            f"beta_power={atomics.beta_power!r}."
        )


@torch.no_grad()
def extract_glass_velocity(
    model,
    state_s: TensorDict,
    cond_state: TensorDict,
    s: Tensor,
    t_cond: Tensor,
    *,
    eps: float = 1e-6,
    mfm_extract_posterior_velocity: Callable | None = None,
    return_atom_logits: bool = False,
) -> TensorDict | tuple[TensorDict, Tensor]:
    """Return GLASS posterior velocity for a TABASCO multimodal DFM state.

    Args:
        model: A ``FlowMatchingModel`` whose atom interpolant is linear DFM.
        state_s: Current auxiliary posterior-flow state ``I_s``.
        cond_state: Conditioning state ``Y_t``.
        s: Auxiliary flow time of shape ``(B,)``.
        t_cond: Conditioning time of shape ``(B,)``.
        eps: Numerical floor passed to MFM's posterior velocity helper.
        mfm_extract_posterior_velocity: Optional injected MFM function, useful for
            tests. By default this imports ``mfm.losses.extract_posterior_velocity``.
        return_atom_logits: also return the teacher's raw atom logits at the GLASS
            point ``(x*, t*)``; their softmax is the diagonal meta-denoiser target
            ``psi_{t*}(S)`` of Prop. 1 (the atom velocity equals
            ``(softmax(logits) - I_s) / (1 - s)``).

    Returns:
        TensorDict with ``coords`` and ``atomics`` velocity blocks, plus the
        ``(B, N, K)`` teacher atom logits if ``return_atom_logits``.
    """
    require_linear_dfm_atoms(model)
    if not torch.equal(state_s["padding_mask"], cond_state["padding_mask"]):
        raise ValueError("state_s and cond_state must share the same padding mask.")

    if mfm_extract_posterior_velocity is None:
        mfm_extract_posterior_velocity = _import_mfm_extract_posterior_velocity()

    codec = MultimodalStateCodec(state_s["padding_mask"])
    flat_state = codec.flatten_state(state_s)
    flat_cond = codec.flatten_state(cond_state)
    adapter = TabascoDiagonalVelocityAdapter(model, codec, eps=eps)

    flat_velocity = mfm_extract_posterior_velocity(
        s,
        flat_state,
        flat_cond,
        t_cond,
        labels=None,
        cfg_scales=None,
        teacher_model=adapter,
        eps=eps,
        checkpoint_type="sit",
    )
    if codec.flat_mask is not None:
        flat_velocity = flat_velocity * codec.flat_mask
    velocity = codec.unflatten_state(flat_velocity, model.data_stats["atom_dim"])
    if return_atom_logits:
        if adapter.num_calls != 1:
            raise RuntimeError(
                f"expected one teacher call for GLASS logits, got {adapter.num_calls}"
            )
        return velocity, adapter.last_atom_logits
    return velocity


@torch.no_grad()
def diagonal_velocity(model, state: TensorDict, t: Tensor, *, eps: float = 1e-5) -> TensorDict:
    """Return TABASCO's diagonal DFM velocity as a TensorDict."""
    require_linear_dfm_atoms(model)
    pred = model._call_net(state, t)
    t_coords = _broadcast_time(t, state["coords"]).clamp(max=1.0 - eps)
    t_atoms = _broadcast_time(t, state["atomics"]).clamp(max=1.0 - eps)

    coord_endpoint = mask_and_zero_com(pred["coords"], state["padding_mask"])
    atom_endpoint = apply_mask(torch.softmax(pred["atomics"], dim=-1), state["padding_mask"])
    coords_v = (coord_endpoint - state["coords"]) / (1.0 - t_coords)
    atomics_v = (atom_endpoint - state["atomics"]) / (1.0 - t_atoms)
    return TensorDict(
        {
            "coords": mask_and_zero_com(coords_v, state["padding_mask"]),
            "atomics": apply_mask(atomics_v, state["padding_mask"]),
            "padding_mask": state["padding_mask"],
        },
        batch_size=state["padding_mask"].shape[0],
    )
