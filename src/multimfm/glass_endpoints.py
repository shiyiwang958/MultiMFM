"""Validate TABASCO GLASS posterior sampling against reverse-SDE sampling.

The comparison is distributional. For each conditioning state ``(Y_t, t)``, this
script draws posterior samples with:

1. GLASS posterior ODE over an auxiliary time ``s``.
2. Reverse SDE integration from ``Y_t`` to time 1 using the same trained DFM
   diagonal velocity.

The model is expected to be a lightweight DFM checkpoint from
``train_pb_valid_curve.py`` with linear atom beta(t)=t.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from tensordict import TensorDict

from multimfm.model_loading import load_lightweight_model
from multimfm.train_base_flow import choose_device, extract_mol, get_atom_names, set_seed
from tabasco.chem.convert import MoleculeConverter
from tabasco.sample.glass import (
    diagonal_velocity,
    extract_glass_velocity,
    require_linear_dfm_atoms,
)
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-conditions", type=int, default=4)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--t-cond", type=float, default=0.4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--condition-source",
        choices=["data", "generated"],
        default="data",
        help="Use dataset endpoints for Y1 by default; generated is a debug mode.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("cache/datasets/processed_geom_val.pt"),
        help="Processed GEOM .pt file used when --condition-source=data.",
    )
    parser.add_argument(
        "--data-limit",
        type=int,
        default=4096,
        help="Maximum raw dataset records to scan for condition endpoints.",
    )
    parser.add_argument(
        "--condition-offset",
        type=int,
        default=0,
        help="Offset into the usable dataset endpoints before taking conditions.",
    )
    parser.add_argument(
        "--endpoint-sample-steps",
        type=int,
        default=100,
        help="Steps used only when --condition-source=generated.",
    )
    parser.add_argument(
        "--keep-generated-atom-probs",
        action="store_true",
        help="Do not argmax generated atom endpoints before making Y_t.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/glass_posterior_validation/summary.json"),
    )
    return parser.parse_args()


def repeat_state(state: TensorDict, repeats: int) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"].repeat_interleave(repeats, dim=0),
            "atomics": state["atomics"].repeat_interleave(repeats, dim=0),
            "padding_mask": state["padding_mask"].repeat_interleave(repeats, dim=0),
        },
        batch_size=state["padding_mask"].shape[0] * repeats,
    )


def discretize_generated_atoms(state: TensorDict) -> TensorDict:
    atom_dim = state["atomics"].shape[-1]
    atom_class = state["atomics"].argmax(dim=-1)
    atomics = F.one_hot(atom_class, num_classes=atom_dim).to(state["atomics"].dtype)
    atomics = apply_mask(atomics, state["padding_mask"])
    out = state.clone()
    out["atomics"] = atomics
    return out


def atom_names_from_stats(data_stats: dict) -> list[str]:
    atom_names = data_stats.get("atom_names")
    if atom_names is not None:
        return list(atom_names)

    atom_dim = int(data_stats.get("atom_dim", 0))
    heavy_atom_names = get_atom_names(include_hydrogens=False)
    hydrogen_atom_names = get_atom_names(include_hydrogens=True)
    if atom_dim == len(heavy_atom_names):
        return heavy_atom_names
    if atom_dim == len(hydrogen_atom_names):
        return hydrogen_atom_names

    include_hydrogens = bool(data_stats.get("include_hydrogens", False))
    fallback = get_atom_names(include_hydrogens=include_hydrogens)
    if atom_dim and len(fallback) != atom_dim:
        raise ValueError(
            f"Cannot infer atom vocabulary for checkpoint atom_dim={atom_dim}."
        )
    return fallback


def load_data_endpoints(
    data_path: Path,
    *,
    data_stats: dict,
    num_conditions: int,
    offset: int,
    limit: int,
    device: torch.device,
) -> TensorDict:
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    if limit > 0:
        raw = raw[:limit]

    atom_names = atom_names_from_stats(data_stats)
    converter = MoleculeConverter(atom_names=atom_names)
    max_num_atoms = int(data_stats["max_num_atoms"])
    include_hydrogens = "H" in atom_names
    remove_hydrogens = not include_hydrogens

    tensors = []
    skipped_too_large = 0
    for item in raw:
        mol = extract_mol(item)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol_copy = Chem.Mol(mol)
        if remove_hydrogens:
            mol_copy = Chem.RemoveAllHs(mol_copy)
        if mol_copy.GetNumAtoms() > max_num_atoms:
            skipped_too_large += 1
            continue
        try:
            tensor = converter.to_tensor(
                mol_copy,
                pad_to_size=max_num_atoms,
                remove_hydrogens=remove_hydrogens,
            )
        except Exception:
            continue
        tensors.append(tensor)
        if len(tensors) >= offset + num_conditions:
            break

    selected = tensors[offset : offset + num_conditions]
    if len(selected) < num_conditions:
        raise ValueError(
            "Could not load enough dataset endpoints from "
            f"{data_path}; got {len(selected)} usable after offset={offset}. "
            f"Skipped {skipped_too_large} molecules larger than max_num_atoms={max_num_atoms}."
        )

    batch = TensorDict(
        {
            "coords": torch.stack([item["coords"] for item in selected], dim=0),
            "atomics": torch.stack([item["atomics"] for item in selected], dim=0),
            "padding_mask": torch.stack(
                [item["padding_mask"] for item in selected], dim=0
            ).bool(),
        },
        batch_size=num_conditions,
    ).to(device)
    batch["coords"] = mask_and_zero_com(batch["coords"], batch["padding_mask"])
    batch["atomics"] = apply_mask(batch["atomics"].float(), batch["padding_mask"])
    return batch


@torch.no_grad()
def make_condition_state_from_endpoint(
    model,
    endpoint: TensorDict,
    *,
    t_cond: float,
) -> tuple[TensorDict, Tensor]:
    noise = model._sample_noise_like_batch(endpoint)
    t = torch.full((endpoint.batch_size[0],), float(t_cond), device=endpoint.device)
    path = model._create_path(endpoint, t=t, noise_batch=noise)
    return path.x_t, t


@torch.no_grad()
def make_generated_condition_state(
    model,
    *,
    batch_size: int,
    t_cond: float,
    endpoint_sample_steps: int,
    discretize_atoms: bool,
) -> tuple[TensorDict, Tensor]:
    endpoint = model.sample(batch_size=batch_size, num_steps=endpoint_sample_steps)
    endpoint["coords"] = mask_and_zero_com(endpoint["coords"], endpoint["padding_mask"])
    endpoint["atomics"] = apply_mask(endpoint["atomics"], endpoint["padding_mask"])
    if discretize_atoms:
        endpoint = discretize_generated_atoms(endpoint)
    return make_condition_state_from_endpoint(model, endpoint, t_cond=t_cond)


@torch.no_grad()
def sample_glass_posterior(
    model,
    cond_state: TensorDict,
    t_cond: Tensor,
    *,
    samples_per_condition: int,
    num_steps: int,
) -> TensorDict:
    cond_rep = repeat_state(cond_state, samples_per_condition)
    t_rep = t_cond.repeat_interleave(samples_per_condition)
    state = model._sample_noise_like_batch(cond_rep)

    s_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=t_rep.device)
    for i in range(num_steps):
        s_cur = torch.full_like(t_rep, s_steps[i])
        ds = s_steps[i + 1] - s_steps[i]
        vel = extract_glass_velocity(model, state, cond_rep, s_cur, t_rep)
        state["coords"] = mask_and_zero_com(
            state["coords"] + vel["coords"] * ds, state["padding_mask"]
        )
        state["atomics"] = apply_mask(
            state["atomics"] + vel["atomics"] * ds, state["padding_mask"]
        )
    return state


def sigma_t_sq(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.clamp((1.0 / (t + 1e-8) - 1.0), min=0.0, max=25.0)


@torch.no_grad()
def sample_sde_posterior(
    model,
    cond_state: TensorDict,
    t_cond: Tensor,
    *,
    samples_per_condition: int,
    num_steps: int,
) -> TensorDict:
    state = repeat_state(cond_state, samples_per_condition)
    t_rep = t_cond.repeat_interleave(samples_per_condition)
    final_t = torch.ones_like(t_rep)

    for i in range(num_steps):
        frac0 = i / num_steps
        frac1 = (i + 1) / num_steps
        t_cur = t_rep + (final_t - t_rep) * frac0
        t_next = t_rep + (final_t - t_rep) * frac1
        dt = t_next - t_cur

        vel = diagonal_velocity(model, state, t_cur)
        t_coords = t_cur.view(-1, 1, 1).clamp_min(1e-4)
        drift_coords = 2.0 * vel["coords"] - state["coords"] / t_coords
        drift_atoms = 2.0 * vel["atomics"] - state["atomics"] / t_coords

        noise = model._sample_noise_like_batch(state)
        diffusion = torch.sqrt(sigma_t_sq(t_cur)).view(-1, 1, 1)
        step_noise = diffusion * torch.sqrt(dt).view(-1, 1, 1)

        state["coords"] = mask_and_zero_com(
            state["coords"]
            + drift_coords * dt.view(-1, 1, 1)
            + step_noise * noise["coords"],
            state["padding_mask"],
        )
        state["atomics"] = apply_mask(
            state["atomics"]
            + drift_atoms * dt.view(-1, 1, 1)
            + step_noise * noise["atomics"],
            state["padding_mask"],
        )
    return state


def flatten_grouped(state: TensorDict, n_conditions: int, samples_per_condition: int):
    coords = state["coords"].view(n_conditions, samples_per_condition, *state["coords"].shape[1:])
    atomics = state["atomics"].view(
        n_conditions, samples_per_condition, *state["atomics"].shape[1:]
    )
    padding_mask = state["padding_mask"].view(
        n_conditions, samples_per_condition, state["padding_mask"].shape[-1]
    )
    coords = mask_and_zero_com(coords, padding_mask)
    atomics = apply_mask(atomics, padding_mask)
    flat = torch.cat(
        [coords.reshape(n_conditions, samples_per_condition, -1), atomics.reshape(n_conditions, samples_per_condition, -1)],
        dim=-1,
    )
    return flat, coords, atomics, padding_mask


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).mean()
    yy = torch.cdist(y, y).mean()
    return 2.0 * xy - xx - yy


def compare_samples(
    glass: TensorDict,
    sde: TensorDict,
    *,
    n_conditions: int,
    samples_per_condition: int,
) -> dict:
    glass_flat, glass_coords, glass_atoms, glass_mask = flatten_grouped(
        glass, n_conditions, samples_per_condition
    )
    sde_flat, sde_coords, sde_atoms, _ = flatten_grouped(
        sde, n_conditions, samples_per_condition
    )

    summaries = []
    for i in range(n_conditions):
        flat_dim = max(1, glass_flat.shape[-1])
        e_dist = energy_distance(glass_flat[i], sde_flat[i])
        mean_l2 = torch.linalg.vector_norm(
            glass_flat[i].mean(dim=0) - sde_flat[i].mean(dim=0)
        ) / np.sqrt(flat_dim)
        std_l2 = torch.linalg.vector_norm(
            glass_flat[i].std(dim=0) - sde_flat[i].std(dim=0)
        ) / np.sqrt(flat_dim)

        real_mask = ~glass_mask[i]
        glass_atom_classes = glass_atoms[i].argmax(dim=-1)[real_mask]
        sde_atom_classes = sde_atoms[i].argmax(dim=-1)[real_mask]
        atom_dim = glass_atoms.shape[-1]
        glass_hist = torch.bincount(glass_atom_classes, minlength=atom_dim).float()
        sde_hist = torch.bincount(sde_atom_classes, minlength=atom_dim).float()
        glass_hist = glass_hist / glass_hist.sum().clamp_min(1.0)
        sde_hist = sde_hist / sde_hist.sum().clamp_min(1.0)

        summaries.append(
            {
                "condition": i,
                "energy_distance": float(e_dist.cpu()),
                "energy_distance_per_sqrt_dim": float((e_dist / np.sqrt(flat_dim)).cpu()),
                "flat_mean_l2_per_sqrt_dim": float(mean_l2.cpu()),
                "flat_std_l2_per_sqrt_dim": float(std_l2.cpu()),
                "coord_mean_abs_diff": float(
                    (glass_coords[i].mean(dim=0) - sde_coords[i].mean(dim=0))
                    .abs()
                    .mean()
                    .cpu()
                ),
                "atom_argmax_hist_l1": float((glass_hist - sde_hist).abs().sum().cpu()),
            }
        )

    aggregate = {
        "mean_energy_distance_per_sqrt_dim": float(
            np.mean([row["energy_distance_per_sqrt_dim"] for row in summaries])
        ),
        "mean_flat_mean_l2_per_sqrt_dim": float(
            np.mean([row["flat_mean_l2_per_sqrt_dim"] for row in summaries])
        ),
        "mean_flat_std_l2_per_sqrt_dim": float(
            np.mean([row["flat_std_l2_per_sqrt_dim"] for row in summaries])
        ),
        "mean_coord_mean_abs_diff": float(
            np.mean([row["coord_mean_abs_diff"] for row in summaries])
        ),
        "mean_atom_argmax_hist_l1": float(
            np.mean([row["atom_argmax_hist_l1"] for row in summaries])
        ),
    }
    return {"per_condition": summaries, **aggregate}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    model, data_stats, model_args = load_lightweight_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)

    if args.condition_source == "data":
        endpoint = load_data_endpoints(
            args.data_path,
            data_stats=data_stats,
            num_conditions=args.num_conditions,
            offset=args.condition_offset,
            limit=args.data_limit,
            device=device,
        )
        cond_state, t_cond = make_condition_state_from_endpoint(
            model, endpoint, t_cond=args.t_cond
        )
    else:
        cond_state, t_cond = make_generated_condition_state(
            model,
            batch_size=args.num_conditions,
            t_cond=args.t_cond,
            endpoint_sample_steps=args.endpoint_sample_steps,
            discretize_atoms=not args.keep_generated_atom_probs,
        )

    glass = sample_glass_posterior(
        model,
        cond_state,
        t_cond,
        samples_per_condition=args.posterior_samples,
        num_steps=args.num_steps,
    )
    sde = sample_sde_posterior(
        model,
        cond_state,
        t_cond,
        samples_per_condition=args.posterior_samples,
        num_steps=args.num_steps,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "atomics_mode": getattr(model_args, "atomics_mode", None),
        "dfm_beta_schedule": getattr(model_args, "dfm_beta_schedule", None),
        "dfm_beta_power": getattr(model_args, "dfm_beta_power", None),
        "num_conditions": args.num_conditions,
        "posterior_samples": args.posterior_samples,
        "t_cond": args.t_cond,
        "condition_source": args.condition_source,
        "data_path": str(args.data_path) if args.condition_source == "data" else None,
        "data_limit": args.data_limit if args.condition_source == "data" else None,
        "condition_offset": (
            args.condition_offset if args.condition_source == "data" else None
        ),
        "num_steps": args.num_steps,
        "endpoint_sample_steps": args.endpoint_sample_steps,
        "device": str(device),
        "stats_max_num_atoms": data_stats.get("max_num_atoms"),
        **compare_samples(
            glass,
            sde,
            n_conditions=args.num_conditions,
            samples_per_condition=args.posterior_samples,
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("GLASS posterior validation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"t_cond: {args.t_cond}")
    print(f"conditions: {args.num_conditions}")
    print(f"posterior_samples: {args.posterior_samples}")
    print(
        "mean_energy_distance_per_sqrt_dim: "
        f"{summary['mean_energy_distance_per_sqrt_dim']:.6f}"
    )
    print(
        "mean_flat_mean_l2_per_sqrt_dim: "
        f"{summary['mean_flat_mean_l2_per_sqrt_dim']:.6f}"
    )
    print(
        "mean_flat_std_l2_per_sqrt_dim: "
        f"{summary['mean_flat_std_l2_per_sqrt_dim']:.6f}"
    )
    print(f"mean_coord_mean_abs_diff: {summary['mean_coord_mean_abs_diff']:.6f}")
    print(f"mean_atom_argmax_hist_l1: {summary['mean_atom_argmax_hist_l1']:.6f}")
    print(f"summary_json: {args.output_json}")


if __name__ == "__main__":
    main()
