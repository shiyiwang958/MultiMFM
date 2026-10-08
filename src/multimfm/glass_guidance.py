"""Inference-time GLASS value-gradient guidance experiment.

This runs two samples from the same initial noise:

1. Unguided TABASCO production sampling.
2. Guided sampling with

       u_t^*(x) = u_t(x) + mu * grad_x V_t(x),
       V_t(x) = log E[exp(lambda * r(X_1)) | X_t = x].

The value function is estimated by differentiating through a small joint
posterior sampler. By default that sampler is GLASS; with ``--value-sampler mfm``
it is the current TABASCO MFM student. This is intentionally a small experiment
script rather than part of the default production sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from multimfm.glass_posterior import bcast, endpoint_prediction, glass_posterior_velocity
from multimfm.qm9_data import REGRESSOR_LOSSES, load_tfg_regressor
from multimfm.train_base_flow import (
    build_model,
    choose_device,
    compute_pb_summary,
    install_posebusters_compat,
    set_seed,
)
from tabasco.chem.convert import MoleculeConverter
from tabasco.models.components.flow_map_transformer import TabascoFlowMap
from tabasco.sample.flow_map import _prior_like
from tabasco.sample.glass import MultimodalStateCodec, require_linear_dfm_atoms
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com

TOY_PROPERTIES = ("first_atom_x", "pairdist_mean", "radius_gyration")
TFG_PROPERTIES = tuple(REGRESSOR_LOSSES["guide"].keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--value-samples", type=int, default=4)
    parser.add_argument(
        "--value-batch-size",
        type=int,
        default=0,
        help=(
            "If positive, split value-gradient evaluation into chunks of this "
            "many conditioning samples to reduce peak memory."
        ),
    )
    parser.add_argument("--value-glass-steps", type=int, default=6)
    parser.add_argument(
        "--value-sampler",
        choices=["glass", "mfm"],
        default="glass",
        help="Posterior sampler differentiated through to estimate V_t.",
    )
    parser.add_argument(
        "--mfm-checkpoint",
        type=Path,
        default=None,
        help="TABASCO MFM student checkpoint used when --value-sampler=mfm.",
    )
    parser.add_argument(
        "--mfm-key",
        default="student",
        help="State-dict key in the MFM checkpoint.",
    )
    parser.add_argument(
        "--mfm-diagonal",
        action="store_true",
        help="Use v(s,s,.) Euler steps instead of off-diagonal MFM jumps.",
    )
    parser.add_argument("--guide-min-t", type=float, default=0.05)
    parser.add_argument("--guide-max-t", type=float, default=0.95)
    parser.add_argument("--guide-every", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--reward",
        choices=[
            "first_atom_x",
            "first_atom_target_x",
            "compactness",
            "radius_gyration",
            "target_property",
        ],
        default="first_atom_x",
    )
    parser.add_argument(
        "--property-name",
        choices=list(TOY_PROPERTIES + TFG_PROPERTIES),
        default="first_atom_x",
        help="Differentiable property used when --reward=target_property.",
    )
    parser.add_argument(
        "--regressor-kind",
        choices=["guide", "oracle"],
        default="guide",
        help="TFG regressor used inside the guidance reward for QM9 properties.",
    )
    parser.add_argument(
        "--eval-oracle",
        action="store_true",
        help="Also evaluate final TFG property values with the opposite oracle/guide checkpoint.",
    )
    parser.add_argument(
        "--property-target",
        type=float,
        default=2.0,
        help="Target property value used when --reward=target_property.",
    )
    parser.add_argument("--target-x", type=float, default=2.0)
    parser.add_argument(
        "--guide-atomics",
        action="store_true",
        help="Also add the atomics gradient component. Default guides coordinates only.",
    )
    parser.add_argument(
        "--guidance-max-coord-rms",
        type=float,
        default=0.0,
        help=(
            "If positive, clip each sample's coordinate guidance gradient to this "
            "masked RMS before applying it."
        ),
    )
    parser.add_argument(
        "--guidance-max-atom-rms",
        type=float,
        default=0.0,
        help=(
            "If positive, clip each sample's atomics guidance gradient to this "
            "masked RMS before applying it."
        ),
    )
    parser.add_argument(
        "--soft-atom-temperature",
        type=float,
        default=0.25,
        help="Temperature for differentiable soft atom embeddings during TFG guidance.",
    )
    parser.add_argument("--compute-pb", action="store_true")
    parser.add_argument(
        "--posebusters-config",
        type=Path,
        default=Path("src/tabasco/utils/posebusters_no_strain.yaml"),
    )
    parser.add_argument("--no-sanitize", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/glass_guided_sampling_experiment"),
    )
    return parser.parse_args()


def is_tfg_property(property_name: str) -> bool:
    return property_name in TFG_PROPERTIES


def seed_all(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_torch_rng(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_torch_rng(state: dict[str, torch.Tensor], device: torch.device) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def clone_state(state: TensorDict) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"].clone(),
            "atomics": state["atomics"].clone(),
            "padding_mask": state["padding_mask"].clone(),
        },
        batch_size=state["padding_mask"].shape[0],
    ).to(state.device)


def repeat_state(state: TensorDict, repeats: int) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"].repeat_interleave(repeats, dim=0),
            "atomics": state["atomics"].repeat_interleave(repeats, dim=0),
            "padding_mask": state["padding_mask"].repeat_interleave(repeats, dim=0),
        },
        batch_size=state["padding_mask"].shape[0] * repeats,
    )


def slice_state(state: TensorDict, index) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"][index],
            "atomics": state["atomics"][index],
            "padding_mask": state["padding_mask"][index],
        },
        batch_size=state["padding_mask"][index].shape[0],
    ).to(state.device)


def slice_batch_value(value, index):
    if torch.is_tensor(value) and value.ndim > 0:
        return value[index]
    return value


def slice_repeated_prior(
    prior_noise: TensorDict | None,
    start: int,
    stop: int,
    repeats: int,
) -> TensorDict | None:
    if prior_noise is None:
        return None
    return slice_state(prior_noise, slice(start * repeats, stop * repeats))


def pairwise_mean_distance(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = []
    for i in range(coords.shape[0]):
        real = ~mask[i]
        x = coords[i, real, :]
        if x.shape[0] < 2:
            values.append(coords.new_tensor(0.0))
            continue
        row, col = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=coords.device)
        values.append(torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0, row, col].mean())
    return torch.stack(values)


def radius_gyration(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = []
    for i in range(coords.shape[0]):
        real = ~mask[i]
        x = coords[i, real, :]
        if x.shape[0] == 0:
            values.append(coords.new_tensor(0.0))
            continue
        center = x.mean(dim=0, keepdim=True)
        values.append(torch.sqrt((x - center).square().sum(dim=-1).mean().clamp_min(0.0)))
    return torch.stack(values)


def tfg_property_prediction(state: TensorDict, regressor) -> torch.Tensor:
    if regressor is None:
        raise ValueError("A TFG regressor is required for QM9 property guidance.")
    coords = state["coords"].float()
    mask = (~state["padding_mask"]).long()
    atomics = state["atomics"].float()
    if atomics.requires_grad:
        temperature = float(getattr(regressor, "soft_atom_temperature", 0.25))
        probs = torch.softmax(atomics / max(temperature, 1e-4), dim=-1)
        h = probs @ regressor.atom_emb.weight
        h = h * mask.unsqueeze(-1)
        h, _ = regressor.gnn.forward(h, coords, mask.bool())
        h = h * mask.unsqueeze(-1)
        h = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        prediction = regressor.predictor(h)
        return prediction.squeeze(-1)
    atom_types = atomics.argmax(dim=-1).long()
    prediction = regressor(coords, atom_types, mask)
    return prediction.squeeze(-1)


def predicted_property(
    state: TensorDict,
    *,
    property_name: str,
    property_regressor=None,
) -> torch.Tensor:
    if property_name == "first_atom_x":
        return state["coords"][:, 0, 0]
    if property_name == "pairdist_mean":
        return pairwise_mean_distance(state["coords"], state["padding_mask"])
    if property_name == "radius_gyration":
        return radius_gyration(state["coords"], state["padding_mask"])
    if is_tfg_property(property_name):
        return tfg_property_prediction(state, property_regressor)
    raise ValueError(f"Unknown property {property_name!r}.")


def target_like(property_target, prop: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(property_target, dtype=prop.dtype, device=prop.device)
    if target.ndim == 0:
        return target
    target = target.reshape(-1)
    if target.shape[0] != prop.shape[0]:
        raise ValueError(
            f"Target shape {tuple(target.shape)} is incompatible with property "
            f"shape {tuple(prop.shape)}."
        )
    return target


def terminal_reward(
    state: TensorDict,
    *,
    reward_name: str,
    target_x: float,
    property_name: str,
    property_target: float,
    property_regressor=None,
) -> torch.Tensor:
    coords = state["coords"]
    mask = state["padding_mask"]
    if reward_name == "first_atom_x":
        return coords[:, 0, 0]
    if reward_name == "first_atom_target_x":
        return -(coords[:, 0, 0] - target_x).square()
    if reward_name == "compactness":
        return -pairwise_mean_distance(coords, mask)
    if reward_name == "radius_gyration":
        return -radius_gyration(coords, mask)
    if reward_name == "target_property":
        prop = predicted_property(
            state,
            property_name=property_name,
            property_regressor=property_regressor,
        )
        return -(prop - target_like(property_target, prop)).square()
    raise ValueError(f"Unknown reward {reward_name!r}.")


def grouped_rewards(
    state: TensorDict,
    *,
    batch_size: int,
    posterior_samples: int,
    reward_name: str,
    target_x: float,
    property_name: str,
    property_target: float,
    property_regressor=None,
) -> torch.Tensor:
    if torch.is_tensor(property_target) and property_target.ndim > 0:
        property_target = property_target.reshape(-1).repeat_interleave(
            posterior_samples
        )
    rewards = terminal_reward(
        state,
        reward_name=reward_name,
        target_x=target_x,
        property_name=property_name,
        property_target=property_target,
        property_regressor=property_regressor,
    )
    return rewards.view(batch_size, posterior_samples)


def differentiable_joint_glass_posterior(
    model,
    cond_state: TensorDict,
    t_cond: torch.Tensor,
    *,
    posterior_samples: int,
    n_steps: int,
    eps: float,
    prior_noise: TensorDict | None = None,
) -> TensorDict:
    atom_dim = cond_state["atomics"].shape[-1]
    cond_rep = repeat_state(cond_state, posterior_samples)
    t_rep = t_cond.repeat_interleave(posterior_samples)
    state = (
        model._sample_noise_like_batch(cond_rep)
        if prior_noise is None
        else clone_state(prior_noise)
    )
    codec = MultimodalStateCodec(cond_rep["padding_mask"])
    cond_block = codec.flatten_state(cond_rep)
    block = codec.flatten_state(state)

    def teacher_v(t_b: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        query = codec.unflatten_state(flat, atom_dim)
        pred = endpoint_prediction(model, query, t_b)
        endpoint_flat = codec.flatten_state(pred)
        state_flat = codec.flatten_state(query)
        denom = bcast((1.0 - t_b).clamp_min(eps), flat)
        return (endpoint_flat - state_flat) / denom

    s_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=t_cond.device)
    for i in range(n_steps):
        s = s_grid[i].expand(t_rep.shape[0])
        ds = s_grid[i + 1] - s_grid[i]
        velocity = glass_posterior_velocity(
            s, block, cond_block, t_rep, teacher_v, eps=eps
        )
        block = block + ds * velocity
        state_tmp = codec.unflatten_state(block, atom_dim)
        block = codec.flatten_state(state_tmp)
    return codec.unflatten_state(block, atom_dim)


def differentiable_mfm_posterior(
    student,
    cond_state: TensorDict,
    t_cond: torch.Tensor,
    *,
    posterior_samples: int,
    n_steps: int,
    diagonal: bool,
    prior_noise: TensorDict | None = None,
) -> TensorDict:
    cond_rep = repeat_state(cond_state, posterior_samples)
    t_rep = t_cond.repeat_interleave(posterior_samples)
    state = _prior_like(cond_rep) if prior_noise is None else clone_state(prior_noise)
    mask = cond_rep["padding_mask"]
    flow_t = torch.linspace(0.0, 1.0, n_steps + 1, device=t_cond.device)

    for i in range(n_steps):
        s = torch.full_like(t_rep, float(flow_t[i]))
        u = torch.full_like(t_rep, float(flow_t[i + 1]))
        if diagonal:
            u = s
        coords_v, atomics_v = student(
            state["coords"],
            state["atomics"],
            cond_rep["coords"],
            cond_rep["atomics"],
            mask,
            s,
            u,
            t_rep,
        )
        step = float(flow_t[i + 1] - flow_t[i])
        ds_coords = torch.full_like(s, step).view(
            -1, *([1] * (state["coords"].ndim - 1))
        )
        ds_atomics = torch.full_like(s, step).view(
            -1, *([1] * (state["atomics"].ndim - 1))
        )
        state["coords"] = mask_and_zero_com(
            state["coords"] + ds_coords * coords_v, mask
        )
        state["atomics"] = apply_mask(
            state["atomics"] + ds_atomics * atomics_v, mask
        )
    return state


def value_gradient(
    model,
    state: TensorDict,
    t: torch.Tensor,
    *,
    value_sampler: str,
    mfm_student=None,
    mfm_diagonal: bool = False,
    posterior_samples: int,
    n_steps: int,
    reward_name: str,
    reward_scale: float,
    target_x: float,
    property_name: str,
    property_target: float,
    property_regressor=None,
    guide_atomics: bool,
    eps: float,
    prior_noise: TensorDict | None = None,
    value_batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    batch_size = state["padding_mask"].shape[0]
    if 0 < value_batch_size < batch_size:
        grad_coord_chunks = []
        grad_atomic_chunks = [] if guide_atomics else None
        stats_chunks = []
        for start in range(0, batch_size, value_batch_size):
            stop = min(start + value_batch_size, batch_size)
            chunk_state = slice_state(state, slice(start, stop))
            chunk_t = t[start:stop]
            chunk_target = slice_batch_value(property_target, slice(start, stop))
            chunk_prior = slice_repeated_prior(
                prior_noise,
                start,
                stop,
                posterior_samples,
            )
            grad_coords, grad_atomics, stats = value_gradient(
                model,
                chunk_state,
                chunk_t,
                value_sampler=value_sampler,
                mfm_student=mfm_student,
                mfm_diagonal=mfm_diagonal,
                posterior_samples=posterior_samples,
                n_steps=n_steps,
                reward_name=reward_name,
                reward_scale=reward_scale,
                target_x=target_x,
                property_name=property_name,
                property_target=chunk_target,
                property_regressor=property_regressor,
                guide_atomics=guide_atomics,
                eps=eps,
                prior_noise=chunk_prior,
                value_batch_size=0,
            )
            grad_coord_chunks.append(grad_coords)
            if guide_atomics and grad_atomic_chunks is not None:
                grad_atomic_chunks.append(grad_atomics)
            stats_chunks.append((stop - start, stats))

        grad_coords = torch.cat(grad_coord_chunks, dim=0)
        grad_atomics = (
            torch.cat(grad_atomic_chunks, dim=0)
            if guide_atomics and grad_atomic_chunks is not None
            else None
        )
        total_batch = sum(size for size, _ in stats_chunks)
        reward_count = total_batch * posterior_samples
        value_mean = sum(
            size * stats["value_mean"] for size, stats in stats_chunks
        ) / max(total_batch, 1)
        reward_mean = sum(
            size * posterior_samples * stats["posterior_reward_mean"]
            for size, stats in stats_chunks
        ) / max(reward_count, 1)
        reward_second = sum(
            size
            * posterior_samples
            * (
                stats["posterior_reward_std"] ** 2
                + stats["posterior_reward_mean"] ** 2
            )
            for size, stats in stats_chunks
        ) / max(reward_count, 1)
        stats = {
            "value_mean": value_mean,
            "posterior_reward_mean": reward_mean,
            "posterior_reward_std": math.sqrt(
                max(0.0, reward_second - reward_mean**2)
            ),
            "grad_coords_rms": float(
                torch.sqrt(grad_coords.square().mean().clamp_min(0.0)).cpu()
            ),
            "value_batch_size": int(value_batch_size),
        }
        if grad_atomics is not None:
            stats["grad_atomics_rms"] = float(
                torch.sqrt(grad_atomics.square().mean().clamp_min(0.0)).cpu()
            )
        return grad_coords, grad_atomics, stats

    coords = state["coords"].detach().clone().requires_grad_(True)
    atomics = state["atomics"].detach().clone().requires_grad_(guide_atomics)
    cond_state = TensorDict(
        {
            "coords": coords,
            "atomics": atomics,
            "padding_mask": state["padding_mask"],
        },
        batch_size=state["padding_mask"].shape[0],
    ).to(state.device)

    if value_sampler == "glass":
        posterior = differentiable_joint_glass_posterior(
            model,
            cond_state,
            t,
            posterior_samples=posterior_samples,
            n_steps=n_steps,
            eps=eps,
            prior_noise=prior_noise,
        )
    elif value_sampler == "mfm":
        if mfm_student is None:
            raise ValueError("--value-sampler=mfm requires --mfm-checkpoint")
        posterior = differentiable_mfm_posterior(
            mfm_student,
            cond_state,
            t,
            posterior_samples=posterior_samples,
            n_steps=n_steps,
            diagonal=mfm_diagonal,
            prior_noise=prior_noise,
        )
    else:
        raise ValueError(f"Unknown value_sampler={value_sampler!r}")
    rewards = grouped_rewards(
        posterior,
        batch_size=state["padding_mask"].shape[0],
        posterior_samples=posterior_samples,
        reward_name=reward_name,
        target_x=target_x,
        property_name=property_name,
        property_target=property_target,
        property_regressor=property_regressor,
    )
    scaled_rewards = reward_scale * rewards
    values = torch.logsumexp(scaled_rewards, dim=-1) - math.log(posterior_samples)
    value_sum = values.sum()
    grad_targets = [coords, atomics] if guide_atomics else [coords]
    grads = torch.autograd.grad(value_sum, grad_targets, allow_unused=True)
    grad_coords = torch.zeros_like(coords) if grads[0] is None else grads[0]
    grad_coords = mask_and_zero_com(grad_coords, state["padding_mask"]).detach()
    grad_atomics = None
    if guide_atomics:
        grad_atomics = torch.zeros_like(atomics) if grads[1] is None else grads[1]
        grad_atomics = apply_mask(grad_atomics, state["padding_mask"]).detach()
    stats = {
        "value_mean": float(values.detach().mean().cpu()),
        "posterior_reward_mean": float(rewards.detach().mean().cpu()),
        "posterior_reward_std": float(rewards.detach().std(unbiased=False).cpu()),
        "grad_coords_rms": float(
            torch.sqrt(grad_coords.square().mean().clamp_min(0.0)).cpu()
        ),
    }
    if grad_atomics is not None:
        stats["grad_atomics_rms"] = float(
            torch.sqrt(grad_atomics.square().mean().clamp_min(0.0)).cpu()
        )
    return grad_coords, grad_atomics, stats


def clip_guidance_rms(
    grad: torch.Tensor,
    padding_mask: torch.Tensor,
    max_rms: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if max_rms <= 0.0:
        rms = torch.sqrt(grad.square().mean().clamp_min(0.0))
        return grad, {
            "rms_before": float(rms.detach().cpu()),
            "rms_after": float(rms.detach().cpu()),
            "scale_mean": 1.0,
            "scale_min": 1.0,
        }

    mask = (~padding_mask).unsqueeze(-1).to(dtype=grad.dtype)
    denom = (mask.sum(dim=1, keepdim=True) * grad.shape[-1]).clamp_min(1.0)
    per_sample_rms = torch.sqrt(
        ((grad.square() * mask).sum(dim=(1, 2), keepdim=True) / denom).clamp_min(0.0)
    )
    scale = torch.clamp(max_rms / per_sample_rms.clamp_min(1e-12), max=1.0)
    clipped = grad * scale
    before = torch.sqrt(grad.square().mean().clamp_min(0.0))
    after = torch.sqrt(clipped.square().mean().clamp_min(0.0))
    return clipped, {
        "rms_before": float(before.detach().cpu()),
        "rms_after": float(after.detach().cpu()),
        "scale_mean": float(scale.detach().mean().cpu()),
        "scale_min": float(scale.detach().min().cpu()),
    }


@torch.no_grad()
def production_step_with_optional_guidance(
    model,
    state: TensorDict,
    t: torch.Tensor,
    dt: torch.Tensor,
    *,
    mu: float,
    grad_coords: torch.Tensor | None,
    grad_atomics: torch.Tensor | None,
) -> TensorDict:
    pred = model._call_net(state, t)
    next_coords = model.coords_interpolant.step(state, pred, t, dt)
    next_atomics = model.atomics_interpolant.step(state, pred, t, dt)
    if grad_coords is not None:
        next_coords = mask_and_zero_com(
            next_coords + mu * grad_coords * bcast(dt, next_coords),
            state["padding_mask"],
        )
    if grad_atomics is not None:
        next_atomics = apply_mask(
            next_atomics + mu * grad_atomics * bcast(dt, next_atomics),
            state["padding_mask"],
        )
    return TensorDict(
        {
            "coords": next_coords,
            "atomics": next_atomics,
            "padding_mask": state["padding_mask"],
        },
        batch_size=state["padding_mask"].shape[0],
    ).to(state.device)


def guided_sample(
    model,
    init_state: TensorDict,
    *,
    value_sampler: str,
    mfm_student=None,
    mfm_diagonal: bool = False,
    num_steps: int,
    mu: float,
    reward_scale: float,
    value_samples: int,
    value_glass_steps: int,
    reward_name: str,
    target_x: float,
    property_name: str,
    property_target: float,
    property_regressor=None,
    guide_atomics: bool,
    guide_min_t: float,
    guide_max_t: float,
    guide_every: int,
    guidance_max_coord_rms: float,
    guidance_max_atom_rms: float,
    eps: float,
    value_batch_size: int = 0,
) -> tuple[TensorDict, list[dict]]:
    state = clone_state(init_state)
    schedule = model._get_sample_schedule(num_steps).to(state.device)
    logs = []
    for i in range(1, len(schedule)):
        t_value = schedule[i - 1]
        dt_value = schedule[i] - schedule[i - 1]
        t = t_value.expand(state["coords"].shape[0])
        should_guide = (
            mu != 0.0
            and value_samples > 0
            and guide_every > 0
            and (i - 1) % guide_every == 0
            and float(t_value) >= guide_min_t
            and float(t_value) <= guide_max_t
        )
        log_row = {"step": i, "t": float(t_value), "guided": bool(should_guide)}
        grad_coords = None
        grad_atomics = None
        rng_state = None
        if should_guide:
            rng_state = capture_torch_rng(state.device)
            with torch.enable_grad():
                grad_coords, grad_atomics, stats = value_gradient(
                    model,
                    state,
                    t,
                    value_sampler=value_sampler,
                    mfm_student=mfm_student,
                    mfm_diagonal=mfm_diagonal,
                    posterior_samples=value_samples,
                    n_steps=value_glass_steps,
                    reward_name=reward_name,
                    reward_scale=reward_scale,
                    target_x=target_x,
                    property_name=property_name,
                    property_target=property_target,
                    property_regressor=property_regressor,
                    guide_atomics=guide_atomics,
                    eps=eps,
                    value_batch_size=value_batch_size,
                )
            log_row.update(stats)
            if grad_coords is not None:
                grad_coords, clip_stats = clip_guidance_rms(
                    grad_coords,
                    state["padding_mask"],
                    guidance_max_coord_rms,
                )
                if guidance_max_coord_rms > 0.0:
                    log_row.update(
                        {
                            "grad_coords_rms_unclipped": clip_stats["rms_before"],
                            "grad_coords_clip_scale_mean": clip_stats["scale_mean"],
                            "grad_coords_clip_scale_min": clip_stats["scale_min"],
                            "grad_coords_rms": clip_stats["rms_after"],
                        }
                    )
            if grad_atomics is not None:
                grad_atomics, clip_stats = clip_guidance_rms(
                    grad_atomics,
                    state["padding_mask"],
                    guidance_max_atom_rms,
                )
                if guidance_max_atom_rms > 0.0:
                    log_row.update(
                        {
                            "grad_atomics_rms_unclipped": clip_stats["rms_before"],
                            "grad_atomics_clip_scale_mean": clip_stats["scale_mean"],
                            "grad_atomics_clip_scale_min": clip_stats["scale_min"],
                            "grad_atomics_rms": clip_stats["rms_after"],
                        }
                    )
            restore_torch_rng(rng_state, state.device)

        with torch.no_grad():
            dt = dt_value.expand(state["coords"].shape[0])
            state = production_step_with_optional_guidance(
                model,
                state,
                t,
                dt,
                mu=mu,
                grad_coords=grad_coords,
                grad_atomics=grad_atomics if guide_atomics else None,
            )
        logs.append(log_row)
    return state, logs


def summarize_state(
    state: TensorDict,
    *,
    reward_name: str,
    target_x: float,
    property_name: str,
    property_target: float,
    property_regressor=None,
    extra_property_regressors: dict[str, object] | None = None,
    run: str,
) -> dict:
    rewards = terminal_reward(
        state,
        reward_name=reward_name,
        target_x=target_x,
        property_name=property_name,
        property_target=property_target,
        property_regressor=property_regressor,
    ).detach()
    prop = predicted_property(
        state,
        property_name=property_name,
        property_regressor=property_regressor,
    ).detach()
    prop_target = target_like(property_target, prop).detach()
    first_x = state["coords"][:, 0, 0].detach()
    pairdist = pairwise_mean_distance(state["coords"], state["padding_mask"]).detach()
    rg = radius_gyration(state["coords"], state["padding_mask"]).detach()
    atom_cls = state["atomics"].argmax(dim=-1)
    real_mask = ~state["padding_mask"]
    dummy_idx = state["atomics"].shape[-1] - 1
    dummy_rate = ((atom_cls == dummy_idx) & real_mask).float().sum() / real_mask.float().sum().clamp_min(1.0)
    row = {
        "run": run,
        "reward_mean": float(rewards.mean().cpu()),
        "reward_std": float(rewards.std(unbiased=False).cpu()),
        "first_atom_x_mean": float(first_x.mean().cpu()),
        "first_atom_x_std": float(first_x.std(unbiased=False).cpu()),
        "property_mean": float(prop.mean().cpu()),
        "property_std": float(prop.std(unbiased=False).cpu()),
        "property_target": float(prop_target.mean().cpu()),
        "property_target_std": float(prop_target.std(unbiased=False).cpu())
        if prop_target.ndim > 0
        else 0.0,
        "property_abs_error_mean": float((prop - prop_target).abs().mean().cpu()),
        "property_abs_error_median": float((prop - prop_target).abs().median().cpu()),
        "pairdist_mean": float(pairdist.mean().cpu()),
        "radius_gyration_mean": float(rg.mean().cpu()),
        "dummy_rate": float(dummy_rate.cpu()),
        "batch_size": state["coords"].shape[0],
    }
    for label, regressor in (extra_property_regressors or {}).items():
        extra_prop = predicted_property(
            state,
            property_name=property_name,
            property_regressor=regressor,
        ).detach()
        extra_target = target_like(property_target, extra_prop).detach()
        row[f"{label}_property_mean"] = float(extra_prop.mean().cpu())
        row[f"{label}_property_std"] = float(extra_prop.std(unbiased=False).cpu())
        row[f"{label}_property_abs_error_mean"] = float(
            (extra_prop - extra_target).abs().mean().cpu()
        )
        row[f"{label}_property_abs_error_median"] = float(
            (extra_prop - extra_target).abs().median().cpu()
        )
    return row


def load_flow_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model_args = SimpleNamespace(**checkpoint["model_args"])
        data_stats = checkpoint["stats"]
        model = build_model(model_args, data_stats)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint and "data_stats" in checkpoint:
        model_args = SimpleNamespace(**checkpoint["args"])
        data_stats = checkpoint["data_stats"]
        model = build_model(model_args, data_stats)
        model.load_state_dict(checkpoint["model"])
    else:
        raise KeyError(
            "Unsupported checkpoint format. Expected either "
            "{model_state_dict, stats, model_args} or {model, data_stats, args}."
        )
    model.set_data_stats(data_stats)
    model = model.to(device).eval()
    return model, data_stats, model_args


def load_mfm_student(
    checkpoint_path: Path,
    *,
    data_stats: dict,
    teacher_args,
    device: torch.device,
    key: str,
) -> TabascoFlowMap:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("student_cfg") or {
        "hidden_dim": teacher_args.hidden_dim,
        "num_layers": teacher_args.num_layers,
        "num_heads": teacher_args.num_heads,
        "block_conditioning": "additive",
        "velocity_parametrization": "endpoint",
    }
    student = TabascoFlowMap(
        data_stats["spatial_dim"],
        data_stats["atom_dim"],
        cfg["num_heads"],
        cfg["num_layers"],
        cfg["hidden_dim"],
        max_num_atoms=data_stats["max_num_atoms"],
        block_conditioning=cfg.get("block_conditioning", "additive"),
        velocity_parametrization=cfg.get("velocity_parametrization", "endpoint"),
        encoder_depth=cfg.get("encoder_depth"),
        time_encoding_max_len=cfg.get("time_encoding_max_len", 200),
    ).to(device)
    state_key = key if key in checkpoint else ("ema" if "ema" in checkpoint else "model")
    student.load_state_dict(checkpoint[state_key])
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    print(
        f"loaded MFM value sampler from {checkpoint_path} "
        f"key={state_key} step={checkpoint.get('step')}"
    )
    return student


def build_posebusters(config_path: Path):
    install_posebusters_compat()
    import yaml
    from posebusters import PoseBusters

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return PoseBusters(config=config)


def pb_summary_for_state(
    state: TensorDict,
    *,
    data_stats: dict,
    posebusters,
    sanitize: bool,
) -> dict:
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    mols = []
    mol_indices = []
    batch_mols = converter.from_batch(
        state.detach().cpu(),
        sanitize=sanitize,
        rescale_coords=True,
    )
    for sample_idx, mol in enumerate(batch_mols):
        if mol is not None:
            mols.append(mol)
            mol_indices.append(sample_idx)
    return compute_pb_summary(
        posebusters=posebusters,
        mols=mols,
        mol_indices=mol_indices,
        total_generated=state["padding_mask"].shape[0],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model, data_stats, model_args = load_flow_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    mfm_student = None
    if args.value_sampler == "mfm":
        if args.mfm_checkpoint is None:
            raise ValueError("--value-sampler=mfm requires --mfm-checkpoint")
        mfm_student = load_mfm_student(
            args.mfm_checkpoint,
            data_stats=data_stats,
            teacher_args=model_args,
            device=device,
            key=args.mfm_key,
        )

    property_regressor = None
    extra_property_regressors = {}
    if is_tfg_property(args.property_name):
        property_regressor = load_tfg_regressor(
            args.regressor_kind,
            args.property_name,
            device,
        )
        property_regressor.soft_atom_temperature = args.soft_atom_temperature
        for param in property_regressor.parameters():
            param.requires_grad_(False)
        property_regressor.eval()
        if args.eval_oracle:
            eval_kind = "oracle" if args.regressor_kind == "guide" else "guide"
            eval_regressor = load_tfg_regressor(eval_kind, args.property_name, device)
            eval_regressor.soft_atom_temperature = args.soft_atom_temperature
            for param in eval_regressor.parameters():
                param.requires_grad_(False)
            eval_regressor.eval()
            extra_property_regressors[eval_kind] = eval_regressor

    posebusters = build_posebusters(args.posebusters_config) if args.compute_pb else None

    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.batch_size)
    init_state = init_state.to(device)

    unguided, unguided_logs = guided_sample(
        model,
        init_state,
        value_sampler=args.value_sampler,
        mfm_student=mfm_student,
        mfm_diagonal=args.mfm_diagonal,
        num_steps=args.sample_steps,
        mu=0.0,
        reward_scale=args.reward_scale,
        value_samples=0,
        value_glass_steps=args.value_glass_steps,
        reward_name=args.reward,
        target_x=args.target_x,
        property_name=args.property_name,
        property_target=args.property_target,
        property_regressor=property_regressor,
        guide_atomics=False,
        guide_min_t=args.guide_min_t,
        guide_max_t=args.guide_max_t,
        guide_every=args.guide_every,
        guidance_max_coord_rms=args.guidance_max_coord_rms,
        guidance_max_atom_rms=args.guidance_max_atom_rms,
        eps=args.eps,
        value_batch_size=args.value_batch_size,
    )
    seed_all(args.seed)
    guided, guided_logs = guided_sample(
        model,
        init_state,
        value_sampler=args.value_sampler,
        mfm_student=mfm_student,
        mfm_diagonal=args.mfm_diagonal,
        num_steps=args.sample_steps,
        mu=args.mu,
        reward_scale=args.reward_scale,
        value_samples=args.value_samples,
        value_glass_steps=args.value_glass_steps,
        reward_name=args.reward,
        target_x=args.target_x,
        property_name=args.property_name,
        property_target=args.property_target,
        property_regressor=property_regressor,
        guide_atomics=args.guide_atomics,
        guide_min_t=args.guide_min_t,
        guide_max_t=args.guide_max_t,
        guide_every=args.guide_every,
        guidance_max_coord_rms=args.guidance_max_coord_rms,
        guidance_max_atom_rms=args.guidance_max_atom_rms,
        eps=args.eps,
        value_batch_size=args.value_batch_size,
    )

    summary = [
        summarize_state(
            unguided,
            reward_name=args.reward,
            target_x=args.target_x,
            property_name=args.property_name,
            property_target=args.property_target,
            property_regressor=property_regressor,
            extra_property_regressors=extra_property_regressors,
            run="unguided",
        ),
        summarize_state(
            guided,
            reward_name=args.reward,
            target_x=args.target_x,
            property_name=args.property_name,
            property_target=args.property_target,
            property_regressor=property_regressor,
            extra_property_regressors=extra_property_regressors,
            run="guided",
        ),
    ]
    if posebusters is not None:
        for row, state in zip(summary, [unguided, guided], strict=True):
            row.update(
                {
                    f"pb_{key}": value
                    for key, value in pb_summary_for_state(
                        state,
                        data_stats=data_stats,
                        posebusters=posebusters,
                        sanitize=not args.no_sanitize,
                    ).items()
                }
            )
    delta = {
        "run": "guided_minus_unguided",
        "reward_mean": summary[1]["reward_mean"] - summary[0]["reward_mean"],
        "first_atom_x_mean": summary[1]["first_atom_x_mean"] - summary[0]["first_atom_x_mean"],
        "property_mean": summary[1]["property_mean"] - summary[0]["property_mean"],
        "property_abs_error_mean": summary[1]["property_abs_error_mean"] - summary[0]["property_abs_error_mean"],
        "pairdist_mean": summary[1]["pairdist_mean"] - summary[0]["pairdist_mean"],
        "radius_gyration_mean": summary[1]["radius_gyration_mean"] - summary[0]["radius_gyration_mean"],
        "dummy_rate": summary[1]["dummy_rate"] - summary[0]["dummy_rate"],
        "batch_size": args.batch_size,
    }
    summary.append(delta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "guided_vs_unguided_summary.csv", summary)
    write_csv(args.output_dir / "guided_step_logs.csv", guided_logs)
    with (args.output_dir / "guided_vs_unguided_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "summary": summary,
                "num_guided_steps": sum(1 for row in guided_logs if row.get("guided")),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    for row in summary:
        print(row)
    print(f"csv: {args.output_dir / 'guided_vs_unguided_summary.csv'}")
    print(f"guided steps: {sum(1 for row in guided_logs if row.get('guided'))}")


if __name__ == "__main__":
    main()
