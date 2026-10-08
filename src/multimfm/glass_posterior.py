"""Norway-style GLASS-vs-SDE posterior validation for TABASCO DFM models.

This mirrors the toy multimodal checks in the Norway repo:

* coords-only: coordinates evolve, atom types are held fixed at the data endpoint.
* atoms-only: atom continuous states evolve, coordinates are held fixed at the
  data endpoint.
* joint: coordinates and atom states evolve together.

For all modes the GLASS sampler integrates an auxiliary ODE in ``s`` from
Gaussian noise, while the SDE sampler integrates Euler-Maruyama in ``t`` from
``t_cond`` to ``1 - eps`` using the unsimplified stochastic-interpolant drift:

    u + ((1 - t) / t) * score,
    score = (t * endpoint_pred - x) / (1 - t)^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from multimfm.glass_endpoints import (
    load_data_endpoints,
    make_condition_state_from_endpoint,
    repeat_state,
)
from multimfm.model_loading import load_lightweight_model
from multimfm.train_base_flow import choose_device, set_seed
from tabasco.sample.glass import (
    MultimodalStateCodec,
    require_linear_dfm_atoms,
)
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


def parse_csv_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"coords", "atoms", "joint"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown modes {unknown}; allowed={sorted(allowed)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-conditions", type=int, default=8)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--posterior-batch-size", type=int, default=256)
    parser.add_argument("--t-cond-list", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--modes", type=str, default="coords,atoms,joint")
    parser.add_argument("--n-steps-glass", type=int, default=200)
    parser.add_argument("--n-steps-sde", type=int, default=500)
    parser.add_argument("--sde-eps", type=float, default=1e-3)
    parser.add_argument("--glass-eps", type=float, default=1e-6)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("cache/datasets/processed_geom_val.pt"),
    )
    parser.add_argument("--data-limit", type=int, default=4096)
    parser.add_argument("--condition-offset", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/glass_posterior_norway_style"),
    )
    return parser.parse_args()


def bcast(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *((1,) * (ref.ndim - 1)))


def seed_all(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def glass_posterior_velocity(
    s: torch.Tensor,
    state_s: torch.Tensor,
    cond_state: torch.Tensor,
    t_cond: torch.Tensor,
    teacher_v,
    *,
    eps: float,
) -> torch.Tensor:
    """Local Norway-style GLASS helper for an arbitrary Tensor state block."""
    s_b = bcast(s, state_s)
    t_b = bcast(t_cond, state_s)
    one_minus_s = 1.0 - s_b
    one_minus_t = 1.0 - t_b

    denom = (t_b.pow(2) * one_minus_s.pow(2) + one_minus_t.pow(2) * s_b.pow(2))
    denom = denom.clamp_min(eps)
    p_norm = one_minus_t.pow(2) / denom
    sqrt_p_norm = torch.sqrt(p_norm)

    t_star = 1.0 / (1.0 + one_minus_s * sqrt_p_norm)
    coeff_cond = t_star * one_minus_s.pow(2) * t_b / denom
    coeff_state = t_star * s_b * p_norm
    x_star = coeff_cond * cond_state + coeff_state * state_s

    v_star = teacher_v(t_star.view(s.shape[0]), x_star)
    term2 = t_star * sqrt_p_norm * v_star
    diff_div_x = (one_minus_t.pow(2) * (1.0 + s_b) - t_b.pow(2) * one_minus_s) / denom
    b_minus_1_div_x = (diff_div_x - (p_norm + sqrt_p_norm)) / (
        1.0 + one_minus_s * sqrt_p_norm
    )
    a_div_x = t_star * one_minus_s * t_b / denom
    term1 = a_div_x * cond_state + b_minus_1_div_x * state_s
    return term1 + term2


def slice_state(state: TensorDict, start: int, end: int) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"][start:end],
            "atomics": state["atomics"][start:end],
            "padding_mask": state["padding_mask"][start:end],
        },
        batch_size=end - start,
    )


def cat_states(states: list[TensorDict]) -> TensorDict:
    batch_size = sum(item["padding_mask"].shape[0] for item in states)
    return TensorDict(
        {
            "coords": torch.cat([item["coords"] for item in states], dim=0),
            "atomics": torch.cat([item["atomics"] for item in states], dim=0),
            "padding_mask": torch.cat([item["padding_mask"] for item in states], dim=0),
        },
        batch_size=batch_size,
    )


def flatten_joint_state(state: TensorDict, codec: MultimodalStateCodec) -> torch.Tensor:
    return codec.flatten_state(state)


def unflatten_joint_state(
    flat: torch.Tensor, codec: MultimodalStateCodec, atom_dim: int
) -> TensorDict:
    return codec.unflatten_state(flat, atom_dim)


def endpoint_prediction(model, state: TensorDict, t: torch.Tensor) -> TensorDict:
    pred = model._call_net(state, t)
    return TensorDict(
        {
            "coords": mask_and_zero_com(pred["coords"], state["padding_mask"]),
            "atomics": apply_mask(torch.softmax(pred["atomics"], dim=-1), state["padding_mask"]),
            "padding_mask": state["padding_mask"],
        },
        batch_size=state["padding_mask"].shape[0],
    )


def sample_noise_block(model, template: TensorDict, mode: str) -> TensorDict:
    noise = model._sample_noise_like_batch(template)
    if mode == "coords":
        return TensorDict(
            {
                "coords": noise["coords"],
                "atomics": template["atomics"],
                "padding_mask": template["padding_mask"],
            },
            batch_size=template["padding_mask"].shape[0],
        )
    if mode == "atoms":
        return TensorDict(
            {
                "coords": template["coords"],
                "atomics": noise["atomics"],
                "padding_mask": template["padding_mask"],
            },
            batch_size=template["padding_mask"].shape[0],
        )
    return noise


@torch.no_grad()
def sample_glass_chunk(
    model,
    endpoint: TensorDict,
    cond_state: TensorDict,
    t_cond: torch.Tensor,
    *,
    mode: str,
    n_steps: int,
    eps: float,
) -> TensorDict:
    atom_dim = endpoint["atomics"].shape[-1]
    state = sample_noise_block(model, endpoint, mode)

    if mode == "coords":
        cond_block = cond_state["coords"]

        def teacher_v(t_b: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            query = TensorDict(
                {
                    "coords": mask_and_zero_com(coords, endpoint["padding_mask"]),
                    "atomics": endpoint["atomics"],
                    "padding_mask": endpoint["padding_mask"],
                },
                batch_size=endpoint["padding_mask"].shape[0],
            )
            pred = endpoint_prediction(model, query, t_b)
            denom = bcast((1.0 - t_b).clamp_min(eps), coords)
            return mask_and_zero_com((pred["coords"] - query["coords"]) / denom, query["padding_mask"])

        block = state["coords"]
    elif mode == "atoms":
        cond_block = cond_state["atomics"]

        def teacher_v(t_b: torch.Tensor, atomics: torch.Tensor) -> torch.Tensor:
            query = TensorDict(
                {
                    "coords": endpoint["coords"],
                    "atomics": apply_mask(atomics, endpoint["padding_mask"]),
                    "padding_mask": endpoint["padding_mask"],
                },
                batch_size=endpoint["padding_mask"].shape[0],
            )
            pred = endpoint_prediction(model, query, t_b)
            denom = bcast((1.0 - t_b).clamp_min(eps), atomics)
            return apply_mask((pred["atomics"] - query["atomics"]) / denom, query["padding_mask"])

        block = state["atomics"]
    else:
        codec = MultimodalStateCodec(endpoint["padding_mask"])
        cond_block = flatten_joint_state(cond_state, codec)
        block = flatten_joint_state(state, codec)

        def teacher_v(t_b: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
            query = unflatten_joint_state(flat, codec, atom_dim)
            pred = endpoint_prediction(model, query, t_b)
            endpoint_flat = codec.flatten_state(pred)
            state_flat = codec.flatten_state(query)
            denom = bcast((1.0 - t_b).clamp_min(eps), flat)
            return (endpoint_flat - state_flat) / denom

    s_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=t_cond.device)
    for i in range(n_steps):
        s = s_grid[i].expand(t_cond.shape[0])
        ds = s_grid[i + 1] - s_grid[i]
        velocity = glass_posterior_velocity(
            s, block, cond_block, t_cond, teacher_v, eps=eps
        )
        block = block + ds * velocity
        if mode == "coords":
            block = mask_and_zero_com(block, endpoint["padding_mask"])
        elif mode == "atoms":
            block = apply_mask(block, endpoint["padding_mask"])
        else:
            state_tmp = unflatten_joint_state(block, codec, atom_dim)
            block = flatten_joint_state(state_tmp, codec)

    if mode == "coords":
        return TensorDict(
            {
                "coords": mask_and_zero_com(block, endpoint["padding_mask"]),
                "atomics": endpoint["atomics"],
                "padding_mask": endpoint["padding_mask"],
            },
            batch_size=endpoint["padding_mask"].shape[0],
        )
    if mode == "atoms":
        return TensorDict(
            {
                "coords": endpoint["coords"],
                "atomics": apply_mask(block, endpoint["padding_mask"]),
                "padding_mask": endpoint["padding_mask"],
            },
            batch_size=endpoint["padding_mask"].shape[0],
        )
    return unflatten_joint_state(block, codec, atom_dim)


@torch.no_grad()
def sample_sde_chunk(
    model,
    endpoint: TensorDict,
    cond_state: TensorDict,
    t_cond: torch.Tensor,
    *,
    mode: str,
    n_steps: int,
    terminal_eps: float,
    denom_eps: float = 1e-6,
) -> TensorDict:
    state = cond_state.clone()
    if mode == "coords":
        state["atomics"] = endpoint["atomics"]
    elif mode == "atoms":
        state["coords"] = endpoint["coords"]

    s_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=t_cond.device)
    final_t = torch.full_like(t_cond, 1.0 - terminal_eps)
    if torch.any(final_t <= t_cond):
        raise ValueError("--sde-eps leaves no positive integration interval.")

    for i in range(n_steps):
        frac0 = s_grid[i]
        frac1 = s_grid[i + 1]
        t_cur = t_cond + (final_t - t_cond) * frac0
        t_next = t_cond + (final_t - t_cond) * frac1
        dt = t_next - t_cur

        pred = endpoint_prediction(model, state, t_cur)
        noise = model._sample_noise_like_batch(state)

        if mode in {"coords", "joint"}:
            one_minus = bcast((1.0 - t_cur).clamp_min(denom_eps), state["coords"])
            u = (pred["coords"] - state["coords"]) / one_minus
            score = (bcast(t_cur, state["coords"]) * pred["coords"] - state["coords"]) / one_minus.pow(2)
            half_sigma_sq = bcast((1.0 - t_cur) / t_cur.clamp_min(denom_eps), state["coords"])
            drift = u + half_sigma_sq * score
            sigma = torch.sqrt(2.0 * half_sigma_sq)
            state["coords"] = mask_and_zero_com(
                state["coords"]
                + drift * bcast(dt, state["coords"])
                + sigma * torch.sqrt(bcast(dt, state["coords"])) * noise["coords"],
                state["padding_mask"],
            )

        if mode in {"atoms", "joint"}:
            one_minus = bcast((1.0 - t_cur).clamp_min(denom_eps), state["atomics"])
            u = (pred["atomics"] - state["atomics"]) / one_minus
            score = (bcast(t_cur, state["atomics"]) * pred["atomics"] - state["atomics"]) / one_minus.pow(2)
            half_sigma_sq = bcast((1.0 - t_cur) / t_cur.clamp_min(denom_eps), state["atomics"])
            drift = u + half_sigma_sq * score
            sigma = torch.sqrt(2.0 * half_sigma_sq)
            state["atomics"] = apply_mask(
                state["atomics"]
                + drift * bcast(dt, state["atomics"])
                + sigma * torch.sqrt(bcast(dt, state["atomics"])) * noise["atomics"],
                state["padding_mask"],
            )

        if mode == "coords":
            state["atomics"] = endpoint["atomics"]
        elif mode == "atoms":
            state["coords"] = endpoint["coords"]

    return state


@torch.no_grad()
def sample_posterior(
    sampler,
    model,
    endpoint: TensorDict,
    cond_state: TensorDict,
    t_cond: torch.Tensor,
    *,
    samples_per_condition: int,
    posterior_batch_size: int,
    seed: int,
    **kwargs,
) -> TensorDict:
    seed_all(seed)
    endpoint_rep = repeat_state(endpoint, samples_per_condition)
    cond_rep = repeat_state(cond_state, samples_per_condition)
    t_rep = t_cond.repeat_interleave(samples_per_condition)
    outputs = []
    total = t_rep.shape[0]
    for start in range(0, total, posterior_batch_size):
        end = min(start + posterior_batch_size, total)
        outputs.append(
            sampler(
                model,
                slice_state(endpoint_rep, start, end),
                slice_state(cond_rep, start, end),
                t_rep[start:end],
                **kwargs,
            )
        )
    return cat_states(outputs)


def real_flat_coords(state: TensorDict, condition: int, samples: int) -> torch.Tensor:
    coords = state["coords"].view(-1, samples, *state["coords"].shape[1:])
    mask = state["padding_mask"].view(-1, samples, state["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    return coords[condition, :, real, :].reshape(samples, -1)


def real_coord_samples(state: TensorDict, condition: int, samples: int) -> torch.Tensor:
    coords = state["coords"].view(-1, samples, *state["coords"].shape[1:])
    mask = state["padding_mask"].view(-1, samples, state["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    return coords[condition, :, real, :]


def endpoint_flat_coords(endpoint: TensorDict, condition: int) -> torch.Tensor:
    real = ~endpoint["padding_mask"][condition]
    return endpoint["coords"][condition, real, :].reshape(-1)


def endpoint_coord_samples(endpoint: TensorDict, condition: int) -> torch.Tensor:
    real = ~endpoint["padding_mask"][condition]
    return endpoint["coords"][condition, real, :]


def pairwise_distance_vector(coords: torch.Tensor) -> torch.Tensor:
    """Return upper-triangle pairwise distances for ``(..., N, 3)`` coordinates."""
    n_atoms = coords.shape[-2]
    if n_atoms < 2:
        return coords.new_zeros((*coords.shape[:-2], 0))
    distances = torch.cdist(coords, coords)
    row, col = torch.triu_indices(n_atoms, n_atoms, offset=1, device=coords.device)
    return distances[..., row, col]


def real_flat_atoms(state: TensorDict, condition: int, samples: int) -> torch.Tensor:
    atomics = state["atomics"].view(-1, samples, *state["atomics"].shape[1:])
    mask = state["padding_mask"].view(-1, samples, state["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    return atomics[condition, :, real, :].reshape(samples, -1)


def atom_site_tv(
    glass: TensorDict, sde: TensorDict, condition: int, samples: int
) -> tuple[float, float]:
    glass_atoms = glass["atomics"].view(-1, samples, *glass["atomics"].shape[1:])
    sde_atoms = sde["atomics"].view(-1, samples, *sde["atomics"].shape[1:])
    mask = glass["padding_mask"].view(-1, samples, glass["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    g_cls = glass_atoms[condition, :, real, :].argmax(dim=-1)
    s_cls = sde_atoms[condition, :, real, :].argmax(dim=-1)
    atom_dim = glass_atoms.shape[-1]
    tvs = []
    for atom_idx in range(g_cls.shape[1]):
        g_hist = torch.bincount(g_cls[:, atom_idx], minlength=atom_dim).float()
        s_hist = torch.bincount(s_cls[:, atom_idx], minlength=atom_dim).float()
        g_hist = g_hist / g_hist.sum().clamp_min(1.0)
        s_hist = s_hist / s_hist.sum().clamp_min(1.0)
        tvs.append(0.5 * (g_hist - s_hist).abs().sum())
    if not tvs:
        return 0.0, 0.0
    values = torch.stack(tvs)
    return float(values.mean().cpu()), float(values.max().cpu())


def atom_endpoint_accuracy(
    samples_state: TensorDict, endpoint: TensorDict, condition: int, samples: int
) -> float:
    atomics = samples_state["atomics"].view(-1, samples, *samples_state["atomics"].shape[1:])
    mask = samples_state["padding_mask"].view(-1, samples, samples_state["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    pred = atomics[condition, :, real, :].argmax(dim=-1)
    target = endpoint["atomics"][condition, real, :].argmax(dim=-1)
    return float((pred == target.unsqueeze(0)).float().mean().cpu())


def atom_vertex_distance(samples_state: TensorDict, condition: int, samples: int) -> float:
    atomics = samples_state["atomics"].view(-1, samples, *samples_state["atomics"].shape[1:])
    mask = samples_state["padding_mask"].view(-1, samples, samples_state["padding_mask"].shape[-1])
    real = ~mask[condition, 0]
    values = atomics[condition, :, real, :]
    if values.numel() == 0:
        return 0.0
    eye = torch.eye(values.shape[-1], device=values.device, dtype=values.dtype)
    dist = torch.cdist(values.reshape(-1, values.shape[-1]), eye)
    return float(dist.min(dim=-1).values.mean().cpu())


def summarize_mode(
    glass: TensorDict,
    sde: TensorDict,
    endpoint: TensorDict,
    *,
    mode: str,
    t_cond: float,
    samples_per_condition: int,
) -> dict:
    num_conditions = endpoint["padding_mask"].shape[0]
    coord_diffs = []
    coord_rels = []
    coord_rels_endpoint = []
    coord_glass_to_endpoint = []
    coord_sde_to_endpoint = []
    coord_glass_to_endpoint_rel = []
    coord_sde_to_endpoint_rel = []
    coord_std_diffs = []
    pairdist_diffs = []
    pairdist_rels_endpoint = []
    pairdist_glass_to_endpoint = []
    pairdist_sde_to_endpoint = []
    pairdist_glass_to_endpoint_rel = []
    pairdist_sde_to_endpoint_rel = []
    atom_diffs = []
    atom_tvs = []
    atom_tv_maxes = []
    joint_diffs = []
    joint_rels = []
    acc_glass = []
    acc_sde = []
    vertex_glass = []
    vertex_sde = []

    for i in range(num_conditions):
        if mode in {"coords", "joint"}:
            g = real_flat_coords(glass, i, samples_per_condition)
            s = real_flat_coords(sde, i, samples_per_condition)
            dim = max(1, g.shape[-1])
            g_mean = g.mean(dim=0)
            s_mean = s.mean(dim=0)
            endpoint_flat = endpoint_flat_coords(endpoint, i)
            endpoint_rms = torch.linalg.vector_norm(endpoint_flat) / np.sqrt(dim)
            endpoint_rms = endpoint_rms.clamp_min(1e-9)
            diff = torch.linalg.vector_norm(g_mean - s_mean)
            diff_per_dim = diff / np.sqrt(dim)
            g_endpoint = torch.linalg.vector_norm(g_mean - endpoint_flat) / np.sqrt(dim)
            s_endpoint = torch.linalg.vector_norm(s_mean - endpoint_flat) / np.sqrt(dim)
            coord_diffs.append(float(diff_per_dim.cpu()))
            coord_rels_endpoint.append(float((diff_per_dim / endpoint_rms).cpu()))
            coord_glass_to_endpoint.append(float(g_endpoint.cpu()))
            coord_sde_to_endpoint.append(float(s_endpoint.cpu()))
            coord_glass_to_endpoint_rel.append(float((g_endpoint / endpoint_rms).cpu()))
            coord_sde_to_endpoint_rel.append(float((s_endpoint / endpoint_rms).cpu()))
            coord_std = g.std(dim=0).mean().clamp_min(1e-9)
            coord_rels.append(float((diff / coord_std).cpu()))
            coord_std_diffs.append(
                float((torch.linalg.vector_norm(g.std(dim=0) - s.std(dim=0)) / np.sqrt(dim)).cpu())
            )

            g_coords = real_coord_samples(glass, i, samples_per_condition)
            s_coords = real_coord_samples(sde, i, samples_per_condition)
            endpoint_coords = endpoint_coord_samples(endpoint, i)
            g_pair = pairwise_distance_vector(g_coords).mean(dim=0)
            s_pair = pairwise_distance_vector(s_coords).mean(dim=0)
            endpoint_pair = pairwise_distance_vector(endpoint_coords)
            pair_dim = max(1, endpoint_pair.numel())
            pair_scale = (torch.linalg.vector_norm(endpoint_pair) / np.sqrt(pair_dim)).clamp_min(1e-9)
            pair_diff = torch.linalg.vector_norm(g_pair - s_pair) / np.sqrt(pair_dim)
            g_pair_endpoint = torch.linalg.vector_norm(g_pair - endpoint_pair) / np.sqrt(pair_dim)
            s_pair_endpoint = torch.linalg.vector_norm(s_pair - endpoint_pair) / np.sqrt(pair_dim)
            pairdist_diffs.append(float(pair_diff.cpu()))
            pairdist_rels_endpoint.append(float((pair_diff / pair_scale).cpu()))
            pairdist_glass_to_endpoint.append(float(g_pair_endpoint.cpu()))
            pairdist_sde_to_endpoint.append(float(s_pair_endpoint.cpu()))
            pairdist_glass_to_endpoint_rel.append(float((g_pair_endpoint / pair_scale).cpu()))
            pairdist_sde_to_endpoint_rel.append(float((s_pair_endpoint / pair_scale).cpu()))
        if mode in {"atoms", "joint"}:
            g = real_flat_atoms(glass, i, samples_per_condition)
            s = real_flat_atoms(sde, i, samples_per_condition)
            dim = max(1, g.shape[-1])
            diff = torch.linalg.vector_norm(g.mean(dim=0) - s.mean(dim=0))
            atom_diffs.append(float((diff / np.sqrt(dim)).cpu()))
            tv_mean, tv_max = atom_site_tv(glass, sde, i, samples_per_condition)
            atom_tvs.append(tv_mean)
            atom_tv_maxes.append(tv_max)
            acc_glass.append(atom_endpoint_accuracy(glass, endpoint, i, samples_per_condition))
            acc_sde.append(atom_endpoint_accuracy(sde, endpoint, i, samples_per_condition))
            vertex_glass.append(atom_vertex_distance(glass, i, samples_per_condition))
            vertex_sde.append(atom_vertex_distance(sde, i, samples_per_condition))
        if mode == "joint":
            g = torch.cat(
                [
                    real_flat_coords(glass, i, samples_per_condition),
                    real_flat_atoms(glass, i, samples_per_condition),
                ],
                dim=-1,
            )
            s = torch.cat(
                [
                    real_flat_coords(sde, i, samples_per_condition),
                    real_flat_atoms(sde, i, samples_per_condition),
                ],
                dim=-1,
            )
            dim = max(1, g.shape[-1])
            diff = torch.linalg.vector_norm(g.mean(dim=0) - s.mean(dim=0))
            joint_diffs.append(float((diff / np.sqrt(dim)).cpu()))
            joint_std = g.std(dim=0).mean().clamp_min(1e-9)
            joint_rels.append(float((diff / joint_std).cpu()))

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def std(values: list[float]) -> float | None:
        return float(np.std(values)) if values else None

    return {
        "mode": mode,
        "t_cond": t_cond,
        "num_conditions": num_conditions,
        "posterior_samples": samples_per_condition,
        "coord_mean_l2_per_sqrt_dim": mean(coord_diffs),
        "coord_mean_l2_std": std(coord_diffs),
        "coord_mean_l2_relative_to_glass_std": mean(coord_rels),
        "coord_mean_l2_relative_to_endpoint_rms": mean(coord_rels_endpoint),
        "coord_glass_mean_l2_to_endpoint_per_sqrt_dim": mean(coord_glass_to_endpoint),
        "coord_sde_mean_l2_to_endpoint_per_sqrt_dim": mean(coord_sde_to_endpoint),
        "coord_glass_mean_l2_to_endpoint_relative_to_endpoint_rms": mean(coord_glass_to_endpoint_rel),
        "coord_sde_mean_l2_to_endpoint_relative_to_endpoint_rms": mean(coord_sde_to_endpoint_rel),
        "coord_std_l2_per_sqrt_dim": mean(coord_std_diffs),
        "pairdist_mean_l2_per_pair_sqrt": mean(pairdist_diffs),
        "pairdist_mean_l2_relative_to_endpoint_rms": mean(pairdist_rels_endpoint),
        "pairdist_glass_mean_l2_to_endpoint_per_pair_sqrt": mean(pairdist_glass_to_endpoint),
        "pairdist_sde_mean_l2_to_endpoint_per_pair_sqrt": mean(pairdist_sde_to_endpoint),
        "pairdist_glass_mean_l2_to_endpoint_relative_to_endpoint_rms": mean(pairdist_glass_to_endpoint_rel),
        "pairdist_sde_mean_l2_to_endpoint_relative_to_endpoint_rms": mean(pairdist_sde_to_endpoint_rel),
        "atom_mean_l2_per_sqrt_dim": mean(atom_diffs),
        "atom_site_tv_mean": mean(atom_tvs),
        "atom_site_tv_max": max(atom_tv_maxes) if atom_tv_maxes else None,
        "atom_endpoint_acc_glass": mean(acc_glass),
        "atom_endpoint_acc_sde": mean(acc_sde),
        "atom_vertex_distance_glass": mean(vertex_glass),
        "atom_vertex_distance_sde": mean(vertex_sde),
        "joint_mean_l2_per_sqrt_dim": mean(joint_diffs),
        "joint_mean_l2_relative_to_glass_std": mean(joint_rels),
    }


def compact_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if value is not None}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    modes = parse_modes(args.modes)
    t_cond_values = parse_csv_numbers(args.t_cond_list, float)
    device = choose_device(args.device)
    model, data_stats, model_args = load_lightweight_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)

    endpoint = load_data_endpoints(
        args.data_path,
        data_stats=data_stats,
        num_conditions=args.num_conditions,
        offset=args.condition_offset,
        limit=args.data_limit,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    full = {
        "checkpoint": str(args.checkpoint),
        "atomics_mode": getattr(model_args, "atomics_mode", None),
        "dfm_beta_schedule": getattr(model_args, "dfm_beta_schedule", None),
        "dfm_beta_power": getattr(model_args, "dfm_beta_power", None),
        "num_conditions": args.num_conditions,
        "posterior_samples": args.posterior_samples,
        "posterior_batch_size": args.posterior_batch_size,
        "n_steps_glass": args.n_steps_glass,
        "n_steps_sde": args.n_steps_sde,
        "sde_eps": args.sde_eps,
        "glass_eps": args.glass_eps,
        "t_cond_values": t_cond_values,
        "modes": modes,
        "data_path": str(args.data_path),
        "data_limit": args.data_limit,
        "condition_offset": args.condition_offset,
        "device": str(device),
        "results": [],
    }

    for t_index, t_cond_value in enumerate(t_cond_values):
        seed_all(args.seed + 10_000 + t_index)
        cond_state, t_cond = make_condition_state_from_endpoint(
            model, endpoint, t_cond=t_cond_value
        )
        for mode_index, mode in enumerate(modes):
            glass = sample_posterior(
                sample_glass_chunk,
                model,
                endpoint,
                cond_state,
                t_cond,
                samples_per_condition=args.posterior_samples,
                posterior_batch_size=args.posterior_batch_size,
                seed=args.seed + 100_000 + 100 * t_index + mode_index,
                mode=mode,
                n_steps=args.n_steps_glass,
                eps=args.glass_eps,
            )
            sde = sample_posterior(
                sample_sde_chunk,
                model,
                endpoint,
                cond_state,
                t_cond,
                samples_per_condition=args.posterior_samples,
                posterior_batch_size=args.posterior_batch_size,
                seed=args.seed + 200_000 + 100 * t_index + mode_index,
                mode=mode,
                n_steps=args.n_steps_sde,
                terminal_eps=args.sde_eps,
            )
            row = compact_row(
                summarize_mode(
                    glass,
                    sde,
                    endpoint,
                    mode=mode,
                    t_cond=t_cond_value,
                    samples_per_condition=args.posterior_samples,
                )
            )
            rows.append(row)
            full["results"].append(row)

            if mode == "coords":
                print(
                    "coords t={t:.2f} mean_l2={mean:.6f} rel_size={rel_size:.4f} "
                    "pair_l2={pair:.6f} endpoint_g/s={eg:.6f}/{es:.6f}".format(
                        t=t_cond_value,
                        mean=row["coord_mean_l2_per_sqrt_dim"],
                        rel_size=row["coord_mean_l2_relative_to_endpoint_rms"],
                        pair=row["pairdist_mean_l2_per_pair_sqrt"],
                        eg=row["coord_glass_mean_l2_to_endpoint_per_sqrt_dim"],
                        es=row["coord_sde_mean_l2_to_endpoint_per_sqrt_dim"],
                    )
                )
            elif mode == "atoms":
                print(
                    "atoms  t={t:.2f} mean_l2={mean:.6f} tv={tv:.6f} acc_g={ag:.3f} acc_s={as_:.3f}".format(
                        t=t_cond_value,
                        mean=row["atom_mean_l2_per_sqrt_dim"],
                        tv=row["atom_site_tv_mean"],
                        ag=row["atom_endpoint_acc_glass"],
                        as_=row["atom_endpoint_acc_sde"],
                    )
                )
            else:
                print(
                    "joint  t={t:.2f} joint_l2={joint:.6f} coord_l2={coord:.6f} atom_tv={tv:.6f}".format(
                        t=t_cond_value,
                        joint=row["joint_mean_l2_per_sqrt_dim"],
                        coord=row["coord_mean_l2_per_sqrt_dim"],
                        tv=row["atom_site_tv_mean"],
                    )
                )

    csv_path = args.output_dir / "norway_style_glass_sweep.csv"
    json_path = args.output_dir / "norway_style_glass_sweep.json"
    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(full, handle, indent=2, sort_keys=True)

    print("Norway-style GLASS posterior validation complete")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
