#!/usr/bin/env python
"""MC gradient-accuracy ablation using a learned dMFM posterior sampler.

The dMFM sampler can be either one composed flow-map step or a diagonal
velocity integrated by RK4. Finite-MC estimates may be compared with an
independent high-MC estimate from the same sampler or with GLASS.

Ported from dirichlet-flow-matching/scripts/ablate_dmfm_one_step_gradient_mc.py (Table 18).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmfm.experiments.ablate_glass_gradient_mc import (
    DEFAULT_CKPTS,
    bootstrap_ci,
    calibrate_score,
    estimate_value_gradient as estimate_glass_value_gradient,
    load_model,
    make_figures,
    motif_tensor,
    seeded_randn,
    target_reward,
)
from dmfm import paths
from dmfm.utils.model_loading import load_args_json, load_student
from dmfm.utils.flow_utils import gaussian_beta
from dmfm.utils.yeast_splits import load_yeast_split_indices
from dmfm.utils.torch_io import torch_load


# 4-step ESD students of Table 18; original dirs
# workdir/yeast_parent_a_L{L}_dmfm_4step_esd_gap_upto025_corrected_20260727/.
DEFAULT_DMFMS = {L: str(paths.dmfm4_ckpt(L)) for L in paths.LENGTHS}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 400])
    p.add_argument("--teacher_ckpt", action="append", default=[], metavar="L=PATH")
    p.add_argument("--dmfm_ckpt", action="append", default=[], metavar="L=PATH")
    p.add_argument("--output_dir", default=str(paths.OUTPUTS / "dmfm_one_step_gradient_accuracy_gc"))
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--probe_data_dir", default=str(paths.DATA / "yeast_parent_disjoint"))
    p.add_argument("--probe_split", default="test")
    p.add_argument("--nfe_probe", type=int, default=32)
    p.add_argument("--nfe_calibration", type=int, default=64)
    p.add_argument("--calibration_samples", type=int, default=1024)
    p.add_argument("--sample_batch_size", type=int, default=32)
    p.add_argument("--n_probes", type=int, default=32)
    p.add_argument("--n_repeats", type=int, default=8)
    p.add_argument("--reference_mc", type=int, default=2048)
    p.add_argument("--reference_sampler", choices=["dmfm", "glass"], default="dmfm")
    p.add_argument("--dmfm_sampler", choices=["one_step", "flow_map", "diagonal_rk4"], default="one_step")
    p.add_argument("--dmfm_steps", type=int, default=16)
    p.add_argument("--dmfm_end_time", type=float, default=0.999)
    p.add_argument("--reference_glass_steps", type=int, default=100)
    p.add_argument("--reference_glass_solver", choices=["euler", "rk4"], default="rk4")
    p.add_argument(
        "--glass_end_time",
        type=float,
        default=0.999,
        help="GLASS endpoint below one, avoiding the singular linear-DFM data-time velocity.",
    )
    p.add_argument("--mc_values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument(
        "--nested_mc_pools",
        action="store_true",
        help="Use shared max-MC noise pools and their prefixes for paired MC comparisons.",
    )
    p.add_argument("--mc_chunk", type=int, default=16)
    p.add_argument("--z_target", type=float, default=1.0)
    p.add_argument("--reward_beta", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--reward", choices=["gc", "motif", "conjunction"], default="gc")
    p.add_argument("--reward_objective", choices=["target", "maximize"], default="target")
    p.add_argument("--motif", default="TTTTTC")
    p.add_argument("--motif2", default="AAAATT")
    p.add_argument("--motif_tau", type=float, default=0.10)
    p.add_argument("--conjunction_tau", type=float, default=0.10)
    p.add_argument("--target_percentile", type=float, default=0.90)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--no_store_gradient_tensors", dest="store_gradient_tensors", action="store_false")
    p.set_defaults(store_gradient_tensors=True)
    return p.parse_args(argv)


def parse_paths(raw: list[str], defaults: dict[int, str], flag: str) -> dict[int, str]:
    paths = dict(defaults)
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Invalid {flag} {spec!r}; expected L=PATH.")
        length, path = spec.split("=", 1)
        paths[int(length)] = path
    return paths


def dmfm_args_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    args_path = path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing dMFM args at {args_path}")
    return args_path


def load_dmfm(checkpoint: str, device: torch.device):
    student_args = load_args_json(dmfm_args_path(checkpoint))
    student, incompat = load_student(student_args, alphabet_size=4, device=device, student_ckpt=checkpoint)
    if incompat is not None and (incompat.missing_keys or incompat.unexpected_keys):
        raise RuntimeError(
            f"Incompatible dMFM checkpoint {checkpoint}: "
            f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
        )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student


def one_step_dmfm(student, terminal_noise: torch.Tensor, x_cond: torch.Tensor, t_cond: torch.Tensor) -> torch.Tensor:
    batch = terminal_noise.shape[0]
    zeros = torch.zeros(batch, device=terminal_noise.device, dtype=terminal_noise.dtype)
    ones = torch.ones(batch, device=terminal_noise.device, dtype=terminal_noise.dtype)
    return student(zeros, ones, terminal_noise, t_cond, x_cond)


def flow_map_dmfm(
    student,
    terminal_noise: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    *,
    n_steps: int,
    end_time: float,
) -> torch.Tensor:
    """Compose learned two-time flow maps on a uniform grid."""
    if int(n_steps) <= 0:
        raise ValueError(f"flow-map n_steps must be positive, got {n_steps}.")
    if not 0.0 < float(end_time) <= 1.0:
        raise ValueError(f"flow-map end_time must lie in (0, 1], got {end_time}.")
    state = terminal_noise
    grid = torch.linspace(0.0, float(end_time), int(n_steps) + 1, device=state.device, dtype=state.dtype)
    for start, end in zip(grid[:-1], grid[1:]):
        batch = state.shape[0]
        state = student(start.expand(batch), end.expand(batch), state, t_cond, x_cond)
    return state


def diagonal_rk4_dmfm(
    student,
    terminal_noise: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    *,
    n_steps: int,
    end_time: float,
) -> torch.Tensor:
    """Integrate the learned diagonal dMFM velocity with uniform-grid RK4."""
    if not 0.0 < float(end_time) < 1.0:
        raise ValueError(f"dMFM end_time must lie in (0, 1), got {end_time}.")
    x = terminal_noise
    grid = torch.linspace(0.0, float(end_time), int(n_steps) + 1, device=x.device, dtype=x.dtype)

    def velocity(r: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        times = r.expand(state.shape[0])
        return student.v(times, times, state, t_cond, x_cond)

    for r0, r1 in zip(grid[:-1], grid[1:]):
        step = r1 - r0
        midpoint = r0 + 0.5 * step
        k1 = velocity(r0, x)
        k2 = velocity(midpoint, x + 0.5 * step * k1)
        k3 = velocity(midpoint, x + 0.5 * step * k2)
        k4 = velocity(r1, x + step * k3)
        x = x + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def sample_dmfm(
    student,
    terminal_noise: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    *,
    sampler: str,
    n_steps: int,
    end_time: float,
) -> torch.Tensor:
    if sampler == "one_step":
        return one_step_dmfm(student, terminal_noise, x_cond, t_cond)
    if sampler == "flow_map":
        return flow_map_dmfm(
            student,
            terminal_noise,
            x_cond,
            t_cond,
            n_steps=n_steps,
            end_time=end_time,
        )
    if sampler == "diagonal_rk4":
        return diagonal_rk4_dmfm(
            student,
            terminal_noise,
            x_cond,
            t_cond,
            n_steps=n_steps,
            end_time=end_time,
        )
    raise ValueError(f"Unknown dMFM sampler: {sampler}")


def make_heldout_probe_states(
    length: int,
    cfg,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample held-out DNA sources and forward-noise them to the evaluation time."""
    root = Path(args.probe_data_dir)
    data_path = root / f"yeast_parent_L{length}.pt"
    split_path = root / f"yeast_parent_L{length}_split_seed0.pt"
    payload = torch_load(data_path, map_location="cpu")
    sequences = payload["seqs"] if isinstance(payload, dict) else payload
    sequences = sequences.long()[load_yeast_split_indices(split_path, args.probe_split)]
    if len(sequences) < args.n_probes:
        raise ValueError(f"Held-out split has {len(sequences)} sequences, fewer than n_probes={args.n_probes}.")
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1_000_000 + length)
    indices = torch.randperm(len(sequences), generator=generator)[: args.n_probes]
    sources = sequences[indices].contiguous()
    one_hot = F.one_hot(sources, num_classes=4).float().to(device)
    t = torch.full((args.n_probes,), args.t_eval, device=device)
    noise = seeded_randn((args.n_probes, length, 4), args.seed + 2_000_000 + length, device)
    beta = gaussian_beta(cfg, t)[:, None, None]
    return beta * one_hot + (1.0 - beta) * noise, indices


def estimate_value_gradient(
    student,
    x: torch.Tensor,
    *,
    t_eval: float,
    eps_pool: torch.Tensor,
    score_center: float,
    score_std: float,
    z_target: float,
    reward_beta: float,
    reward_scale: float,
    reward: str,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    motif_tau: float,
    conjunction_tau: float,
    mc_chunk: int,
    dmfm_sampler: str = "one_step",
    dmfm_steps: int = 16,
    dmfm_end_time: float = 0.999,
    reward_objective: str = "target",
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, mc = eps_pool.shape[:2]
    chunk = max(1, min(int(mc_chunk), mc))
    with torch.no_grad():
        log_rewards: list[torch.Tensor] = []
        for start in range(0, mc, chunk):
            end = min(start + chunk, mc)
            width = end - start
            terminal_noise = eps_pool[:, start:end].reshape(batch * width, *x.shape[1:])
            x_rep = x.detach().repeat_interleave(width, dim=0)
            t_rep = torch.full((batch * width,), t_eval, device=x.device, dtype=x.dtype)
            endpoint = sample_dmfm(
                student,
                terminal_noise,
                x_rep,
                t_rep,
                sampler=dmfm_sampler,
                n_steps=dmfm_steps,
                end_time=dmfm_end_time,
            )
            reward_value = target_reward(
                endpoint,
                score_center=score_center,
                score_std=score_std,
                z_target=z_target,
                beta=reward_beta,
                reward=reward,
                motif=motif,
                motif2=motif2,
                motif_tau=motif_tau,
                conjunction_tau=conjunction_tau,
                objective=reward_objective,
            )
            log_rewards.append((float(reward_scale) * reward_value).view(batch, width))
        log_rewards_full = torch.cat(log_rewards, dim=1)
        value = torch.logsumexp(log_rewards_full, dim=1) - math.log(mc)
        weights = torch.softmax(log_rewards_full, dim=1)

    x_leaf = x.detach().clone().requires_grad_(True)
    gradient = torch.zeros_like(x_leaf)
    for start in range(0, mc, chunk):
        end = min(start + chunk, mc)
        width = end - start
        terminal_noise = eps_pool[:, start:end].reshape(batch * width, *x.shape[1:])
        x_rep = x_leaf.repeat_interleave(width, dim=0)
        t_rep = torch.full((batch * width,), t_eval, device=x.device, dtype=x.dtype)
        endpoint = sample_dmfm(
            student,
            terminal_noise,
            x_rep,
            t_rep,
            sampler=dmfm_sampler,
            n_steps=dmfm_steps,
            end_time=dmfm_end_time,
        )
        reward_value = target_reward(
            endpoint,
            score_center=score_center,
            score_std=score_std,
            z_target=z_target,
            beta=reward_beta,
            reward=reward,
            motif=motif,
            motif2=motif2,
            motif_tau=motif_tau,
            conjunction_tau=conjunction_tau,
            objective=reward_objective,
        ).view(batch, width)
        objective = (weights[:, start:end].detach() * (float(reward_scale) * reward_value)).sum()
        gradient = gradient + torch.autograd.grad(objective, x_leaf, create_graph=False)[0]
    return value.detach(), gradient.detach()


def run_length(length: int, teacher_path: str, dmfm_path: str, args: argparse.Namespace, root: Path, device: torch.device):
    print(f"Loading L={length}: DFM={teacher_path}; dMFM={dmfm_path}", flush=True)
    teacher, cfg = load_model(teacher_path, device)
    if int(cfg.seq_len) != length or int(cfg.alphabet_size) != 4:
        raise ValueError(f"Unexpected teacher shape for L={length}")
    student = load_dmfm(dmfm_path, device)
    out_dir = root / f"L{length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    motif = motif_tensor(args.motif, device)
    motif2 = motif_tensor(args.motif2, device)
    calibration = calibrate_score(
        teacher,
        cfg,
        n_samples=args.calibration_samples,
        batch_size=args.sample_batch_size,
        nfe=args.nfe_calibration,
        seed=args.seed + 100_000 * length,
        device=device,
        reward=args.reward,
        motif=motif,
        motif2=motif2,
        motif_tau=args.motif_tau,
        conjunction_tau=args.conjunction_tau,
    )
    score_mean, score_std = float(calibration.mean()), float(calibration.std(ddof=1))
    if not np.isfinite(score_std) or score_std <= 0.0:
        raise RuntimeError(f"Invalid calibration score std at L={length}: {score_std}")
    if args.reward_objective == "maximize":
        score_center, reward_z_target = score_mean, 0.0
    elif args.reward == "gc":
        score_center, reward_z_target = score_mean, float(args.z_target)
    else:
        score_center, reward_z_target = float(np.quantile(calibration, args.target_percentile)), 0.0
    np.save(out_dir / "score_calibration_samples.npy", calibration)
    states, source_indices = make_heldout_probe_states(length, cfg, args, device)
    rows: list[dict[str, float | int]] = []
    reference_gradients: list[torch.Tensor] = []
    cache_reference_eps: list[torch.Tensor] = []
    cache_reference_gradients: list[torch.Tensor] = []
    cache_candidate_eps: list[torch.Tensor] = []
    estimate_gradients = (
        torch.empty((args.n_probes, len(args.mc_values), args.n_repeats, length, 4), dtype=torch.float32)
        if args.store_gradient_tensors else None
    )
    common = dict(
        score_center=score_center, score_std=score_std, z_target=reward_z_target,
        reward_beta=args.reward_beta, reward_scale=args.reward_scale, reward=args.reward,
        motif=motif, motif2=motif2, motif_tau=args.motif_tau, conjunction_tau=args.conjunction_tau,
        mc_chunk=args.mc_chunk, dmfm_sampler=args.dmfm_sampler, dmfm_steps=args.dmfm_steps,
        dmfm_end_time=args.dmfm_end_time, reward_objective=args.reward_objective,
    )
    for probe_idx in range(args.n_probes):
        x = states[probe_idx : probe_idx + 1]
        reference_eps = seeded_randn(
            (1, args.reference_mc, length, 4),
            args.seed + 10_000_000 + 10_000 * length + probe_idx, device,
        )
        cache_reference_eps.append(reference_eps.cpu())
        if args.reference_sampler == "dmfm":
            _, reference_grad = estimate_value_gradient(student, x, t_eval=args.t_eval, eps_pool=reference_eps, **common)
        else:
            glass_common = {
                key: value for key, value in common.items() if not key.startswith("dmfm_")
            }
            _, reference_grad = estimate_glass_value_gradient(
                teacher,
                x,
                t_eval=args.t_eval,
                eps_pool=reference_eps,
                nfe_value=args.reference_glass_steps,
                glass_end_time=args.glass_end_time,
                glass_solver=args.reference_glass_solver,
                **glass_common,
            )
        reference_l2 = reference_grad.flatten().norm().item()
        cache_reference_gradients.append(reference_grad.cpu())
        if args.store_gradient_tensors:
            reference_gradients.append(reference_grad.cpu())
        for repeat in range(args.n_repeats):
            nested_eps = None
            if args.nested_mc_pools:
                nested_eps = seeded_randn(
                    (1, max(args.mc_values), length, 4),
                    args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe_idx + repeat,
                    device,
                )
                cache_candidate_eps.append(nested_eps.cpu())
            for mc_idx, mc in enumerate(args.mc_values):
                if nested_eps is not None:
                    eps_pool = nested_eps[:, :mc]
                else:
                    eps_pool = seeded_randn(
                        (1, mc, length, 4),
                        args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe_idx + 100 * mc + repeat,
                        device,
                    )
                _, estimate_grad = estimate_value_gradient(student, x, t_eval=args.t_eval, eps_pool=eps_pool, **common)
                error_l2 = (estimate_grad - reference_grad).flatten().norm().item()
                estimate_l2 = estimate_grad.flatten().norm().item()
                dot_product = (estimate_grad * reference_grad).sum().item()
                rows.append({
                    "length": length, "probe": probe_idx, "repeat": repeat, "mc": mc,
                    "e_rms": error_l2 / math.sqrt(4 * length), "error_l2": error_l2,
                    "reference_l2": reference_l2, "reference_rms": reference_l2 / math.sqrt(4 * length),
                    "estimate_l2": estimate_l2, "gradient_dot_product": dot_product,
                    "cosine_similarity": dot_product / max(estimate_l2 * reference_l2, 1e-12),
                    "relative_l2_error": error_l2 / max(reference_l2, 1e-12),
                })
                if estimate_gradients is not None:
                    estimate_gradients[probe_idx, mc_idx, repeat] = estimate_grad.cpu()
        print(f"L={length}: probe {probe_idx + 1}/{args.n_probes}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "gradient_errors.csv", index=False)
    if estimate_gradients is not None:
        torch.save({
            "reference_gradients": torch.cat(reference_gradients, dim=0),
            "estimate_gradients": estimate_gradients,
            "mc_values": torch.tensor(args.mc_values, dtype=torch.long),
            "layout": "estimate_gradients[probe, mc_index, repeat, position, alphabet]",
        }, out_dir / "gradient_pairs.pt")
    torch.save({
        "format_version": 1,
        "conditioning_states": states.detach().cpu(),
        "source_indices": torch.as_tensor(source_indices, dtype=torch.long),
        "t_eval": float(args.t_eval),
        "score_mean": score_mean,
        "score_std": score_std,
        "score_center": score_center,
        "reward_z_target": reward_z_target,
        "reference_eps": torch.cat(cache_reference_eps, dim=0),
        "reference_gradients": torch.cat(cache_reference_gradients, dim=0),
        "candidate_eps_pools": (
            torch.cat(cache_candidate_eps, dim=0).view(args.n_probes, args.n_repeats, max(args.mc_values), length, 4)
            if cache_candidate_eps else None
        ),
        "reward": args.reward,
        "reward_objective": args.reward_objective,
        "mc_values": torch.tensor(args.mc_values, dtype=torch.long),
        "nested_mc_pools": args.nested_mc_pools,
    }, out_dir / "evaluation_cache.pt")
    summary_rows = []
    for mc, group in raw.groupby("mc", sort=True):
        rms = group.groupby("probe")["e_rms"].mean().to_numpy()
        l2 = group.groupby("probe")["error_l2"].mean().to_numpy()
        relative = group.groupby("probe")["relative_l2_error"].mean().to_numpy()
        rms_low, rms_high = bootstrap_ci(rms, args.bootstrap, args.seed + length + int(mc))
        l2_low, l2_high = bootstrap_ci(l2, args.bootstrap, args.seed + 10_000 + length + int(mc))
        rel_low, rel_high = bootstrap_ci(relative, args.bootstrap, args.seed + 20_000 + length + int(mc))
        summary_rows.append({
            "length": length, "mc": int(mc), "mean_e_rms": float(rms.mean()), "ci95_low": rms_low, "ci95_high": rms_high,
            "std_across_probes": float(rms.std(ddof=1)) if len(rms) > 1 else 0.0,
            "mean_error_l2": float(l2.mean()), "error_l2_ci95_low": l2_low, "error_l2_ci95_high": l2_high,
            "mean_relative_l2_error": float(relative.mean()), "relative_l2_error_ci95_low": rel_low,
            "relative_l2_error_ci95_high": rel_high, "n_probes": args.n_probes, "n_repeats": args.n_repeats,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "gradient_error_summary.csv", index=False)
    metadata = {
        "length": length, "teacher_checkpoint": teacher_path, "dmfm_checkpoint": dmfm_path,
        "posterior_sampler": args.dmfm_sampler, "dmfm_steps": args.dmfm_steps,
        "dmfm_end_time": args.dmfm_end_time, "reference_sampler": args.reference_sampler,
        "reference_glass_steps": args.reference_glass_steps, "score_mean": score_mean, "score_std": score_std,
        "reference_glass_solver": args.reference_glass_solver,
        "glass_end_time": args.glass_end_time,
        "score_center": score_center, "dimensions": 4 * length, "reference_mc": args.reference_mc,
        "normalization": "E_RMS = ||gradient_estimate - gradient_reference||_2 / sqrt(4L)",
        "reference": f"independent high-MC {args.reference_sampler} estimator with reference_mc terminal-noise samples",
        "reward": {"name": args.reward, "objective": args.reward_objective,
                   "z_target": reward_z_target, "beta": args.reward_beta, "lambda": args.reward_scale,
                   "motif": args.motif, "motif2": args.motif2 if args.reward == "conjunction" else None,
                   "motif_tau": args.motif_tau if args.reward != "gc" else None,
                   "conjunction_tau": args.conjunction_tau if args.reward == "conjunction" else None,
                   "target_percentile": args.target_percentile if args.reward != "gc" else None},
        "t_eval": args.t_eval, "nfe_probe": args.nfe_probe, "n_probes": args.n_probes,
        "n_repeats": args.n_repeats, "mc_values": args.mc_values,
        "nested_mc_pools": args.nested_mc_pools,
        "probe_states": "x_t = beta(t) * one_hot(held_out_test_sequence) + (1 - beta(t)) * Gaussian_noise",
        "probe_split": args.probe_split,
        "probe_source_indices": source_indices.tolist(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return summary, metadata


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.t_eval < 1.0:
        raise ValueError("--t_eval must be in (0, 1)")
    if max(args.mc_values) >= args.reference_mc:
        raise ValueError("--reference_mc must exceed every candidate MC value")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    teachers = parse_paths(args.teacher_ckpt, DEFAULT_CKPTS, "--teacher_ckpt")
    students = parse_paths(args.dmfm_ckpt, DEFAULT_DMFMS, "--dmfm_ckpt")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries, metadata = [], []
    for length in args.lengths:
        summary, meta = run_length(length, teachers[length], students[length], args, root, device)
        summaries.append(summary)
        metadata.append(meta)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(root / "gradient_error_summary_all_lengths.csv", index=False)
    required = make_figures(root, all_summary, args.thresholds)
    with open(root / "run_metadata.json", "w") as f:
        json.dump({"args": vars(args), "per_length": metadata, "elapsed_seconds": time.time() - started,
                   "required_mc_definition": "smallest tested N whose bootstrap upper 95% CI for mean E_RMS is <= delta"}, f, indent=2, sort_keys=True)
    print(all_summary.to_string(index=False))
    print(required.to_string(index=False))
    print(f"Outputs written to {root.resolve()} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
