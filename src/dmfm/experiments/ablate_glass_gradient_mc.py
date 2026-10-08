#!/usr/bin/env python
"""Measure Monte Carlo error in GLASS value-gradient estimates across DNA lengths.

The experiment uses the trained diagonal DFM models only.  It evaluates a
standardized GC-content reward at intermediate states from each base flow,
then compares finite-MC GLASS gradients against an independent high-MC
reference.  No learned property model is involved.

Ported from dirichlet-flow-matching/scripts/ablate_glass_gradient_mc.py (Fig 5, Tables 14-15).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step
from dmfm.utils.torch_io import torch_load


# Base DFM teachers; originally workdir/yeast_parent_a_L{L}_dfm_dit_h192_b4/best.pt.
DEFAULT_CKPTS = {L: str(paths.base_ckpt(L)) for L in paths.LENGTHS}
DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 400])
    p.add_argument(
        "--ckpt",
        action="append",
        default=[],
        metavar="L=PATH",
        help="Override a checkpoint path, e.g. --ckpt 50=checkpoints/dna/base/L50/best.pt.",
    )
    p.add_argument("--output_dir", default=str(paths.OUTPUTS / "gradient_accuracy_gc"))
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Base-flow states and per-length reward calibration.
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--nfe_probe", type=int, default=32)
    p.add_argument("--nfe_calibration", type=int, default=64)
    p.add_argument("--calibration_samples", type=int, default=1024)
    p.add_argument("--sample_batch_size", type=int, default=32)

    # GLASS estimator and comparison design.
    p.add_argument("--nfe_value", type=int, default=8)
    p.add_argument(
        "--glass_end_time",
        type=float,
        default=0.999,
        help="Posterior-flow endpoint. Keep below one for RK4 because the linear DFM velocity is singular "
        "at t=1; 1.0 with the Euler solver reproduces the pre-flag runs of Fig 5 / Tables 14-15.",
    )
    p.add_argument("--glass_solver", choices=["euler", "rk4"], default="euler")
    p.add_argument("--n_probes", type=int, default=32)
    p.add_argument("--n_repeats", type=int, default=8)
    p.add_argument("--reference_mc", type=int, default=2048)
    p.add_argument("--mc_values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--mc_chunk", type=int, default=16)
    p.add_argument("--z_target", type=float, default=1.0)
    p.add_argument("--reward_beta", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0, help="Lambda in exp(lambda * reward).")
    p.add_argument("--reward", choices=["gc", "motif", "conjunction"], default="gc")
    p.add_argument(
        "--reward_objective",
        choices=["target", "maximize"],
        default="target",
        help="Use a quadratic standardized target or directly maximize the standardized score.",
    )
    p.add_argument(
        "--motif",
        default="TTTTTC",
        help="Fixed A/C/G/T motif for the nonlinear reward and first conjunction component.",
    )
    p.add_argument(
        "--motif2",
        default="AAAATT",
        help="Second fixed A/C/G/T motif, used only by --reward conjunction.",
    )
    p.add_argument("--motif_tau", type=float, default=0.10, help="Smooth-max temperature for motif occurrence.")
    p.add_argument("--conjunction_tau", type=float, default=0.10, help="Smooth-min temperature for conjunction.")
    p.add_argument(
        "--target_percentile",
        type=float,
        default=0.90,
        help="Unconditional score percentile targeted by motif and conjunction rewards.",
    )
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--no_store_gradient_tensors",
        dest="store_gradient_tensors",
        action="store_false",
        help="Do not write compact per-probe reference/estimate gradient tensors.",
    )
    p.set_defaults(store_gradient_tensors=True)
    return p.parse_args(argv)


def seeded_randn(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def parse_checkpoints(raw: list[str]) -> dict[int, str]:
    checkpoints = dict(DEFAULT_CKPTS)
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Invalid --ckpt {spec!r}; expected L=PATH.")
        length, path = spec.split("=", 1)
        checkpoints[int(length)] = path
    return checkpoints


def load_model(path: str, device: torch.device) -> tuple[DiTSequenceModel, SimpleNamespace]:
    checkpoint = torch_load(path, map_location="cpu")
    model_cfg = checkpoint["model_cfg"]
    cfg = model_cfg if isinstance(model_cfg, SimpleNamespace) else SimpleNamespace(**model_cfg)
    cfg.flow_temp = float(getattr(cfg, "flow_temp", 1.0))
    cfg.gaussian_beta_schedule = getattr(cfg, "gaussian_beta_schedule", "linear")
    cfg.gaussian_beta_table_path = getattr(cfg, "gaussian_beta_table_path", None)

    model = DiTSequenceModel(cfg, alphabet_size=cfg.alphabet_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def flow_to_time(
    model: DiTSequenceModel,
    cfg: SimpleNamespace,
    x: torch.Tensor,
    t_end: float,
    nfe: int,
) -> torch.Tensor:
    """Integrate the trained base diagonal DFM from Gaussian noise to `t_end`."""
    grid = torch.linspace(0.0, float(t_end), int(nfe) + 1, device=x.device, dtype=x.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        s = s0.expand(x.shape[0])
        t = s1.expand(x.shape[0])
        x, _, _ = gaussian_denoiser_flow_step(cfg, model, x, s, t)
    return x


def bcast_like(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while coef.ndim < x.ndim:
        coef = coef[..., None]
    return coef.to(device=x.device, dtype=x.dtype)


def glass_velocity_diff(
    model: DiTSequenceModel,
    s: torch.Tensor,
    terminal_noise: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Conditional GLASS velocity used by the existing DNA guidance sampler."""
    ss = bcast_like(s, terminal_noise)
    tt = bcast_like(t_cond, terminal_noise)
    one_minus_s = 1.0 - ss
    denom = (tt.pow(2) * one_minus_s.pow(2) + (1.0 - tt).pow(2) * ss.pow(2)).clamp_min(eps)
    p = (1.0 - tt).pow(2) / denom
    sqrt_p = p.sqrt()
    t_star = 1.0 / (1.0 + one_minus_s * sqrt_p)
    coeff_cond = t_star * one_minus_s.pow(2) * tt / denom
    coeff_terminal = t_star * ss * p
    x_star = coeff_cond * x_cond + coeff_terminal * terminal_noise

    t_star_vec = t_star.reshape(terminal_noise.shape[0], -1)[:, 0]
    v_star = model.v(t_star_vec, t_star_vec, x_star, t_cond, x_cond)

    term2 = t_star * sqrt_p * v_star
    diff_div_x = ((1.0 - tt).pow(2) * (1.0 + ss) - tt.pow(2) * one_minus_s) / denom
    b_minus_1_div_x = (diff_div_x - (p + sqrt_p)) / (1.0 + one_minus_s * sqrt_p)
    a_div_x = t_star * one_minus_s * tt / denom
    return a_div_x * x_cond + b_minus_1_div_x * terminal_noise + term2


def glass_integrate_diff(
    model: DiTSequenceModel,
    terminal_noise: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    n_steps: int,
    end_time: float = 0.999,
    solver: str = "euler",
) -> torch.Tensor:
    """Integrate the GLASS posterior ODE without evaluating the singular data-time drift.

    Port change: ``end_time=1.0`` is also accepted for the Euler solver. The runs behind
    Fig 5 / Tables 14-15 (2026-07-26 morning) predate ``--glass_end_time`` and integrated
    ``linspace(0, 1, n_steps + 1)`` with Euler, like ``sample_c0_guidance.glass_integrate_diff``;
    Euler never evaluates the drift at s=1, so this is safe. RK4 still requires end_time < 1.
    """
    legacy_euler_to_one = float(end_time) == 1.0 and solver == "euler"
    if not (0.0 < float(end_time) < 1.0 or legacy_euler_to_one):
        raise ValueError(f"GLASS end_time must lie in (0, 1) (or equal 1 for Euler), got {end_time}.")
    if solver not in {"euler", "rk4"}:
        raise ValueError(f"Unknown GLASS solver {solver!r}.")
    x = terminal_noise
    grid = torch.linspace(0.0, float(end_time), int(n_steps) + 1, device=x.device, dtype=x.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        step = float(s1 - s0)
        s = torch.full((x.shape[0],), float(s0), device=x.device, dtype=x.dtype)
        if solver == "euler":
            x = x + step * glass_velocity_diff(model, s, x, x_cond, t_cond)
            continue
        midpoint = torch.full((x.shape[0],), float(s0 + 0.5 * step), device=x.device, dtype=x.dtype)
        s_next = torch.full((x.shape[0],), float(s1), device=x.device, dtype=x.dtype)
        k1 = glass_velocity_diff(model, s, x, x_cond, t_cond)
        k2 = glass_velocity_diff(model, midpoint, x + 0.5 * step * k1, x_cond, t_cond)
        k3 = glass_velocity_diff(model, midpoint, x + 0.5 * step * k2, x_cond, t_cond)
        k4 = glass_velocity_diff(model, s_next, x + step * k3, x_cond, t_cond)
        x = x + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def motif_tensor(motif: str, device: torch.device) -> torch.Tensor:
    motif = motif.upper()
    if not motif or any(base not in DNA_TO_INDEX for base in motif):
        raise ValueError(f"Motif must be a non-empty A/C/G/T string, got {motif!r}.")
    return torch.tensor([DNA_TO_INDEX[base] for base in motif], device=device, dtype=torch.long)


def soft_motif_score(endpoint: torch.Tensor, motif: torch.Tensor, tau: float) -> torch.Tensor:
    """Normalized soft maximum of a motif's local fractional match scores."""
    length = endpoint.shape[1]
    width = motif.numel()
    if width > length:
        raise ValueError(f"Motif length {width} exceeds sequence length {length}.")
    if tau <= 0.0:
        raise ValueError("Motif temperature must be positive.")
    kernel = F.one_hot(motif, num_classes=endpoint.shape[-1]).to(dtype=endpoint.dtype).T.unsqueeze(0)
    local_match = F.conv1d(endpoint.transpose(1, 2), kernel).squeeze(1) / float(width)
    return float(tau) * (torch.logsumexp(local_match / float(tau), dim=-1) - math.log(local_match.shape[-1]))


def reward_score(
    endpoint: torch.Tensor,
    *,
    reward: str,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    motif_tau: float,
    conjunction_tau: float,
) -> torch.Tensor:
    if reward == "gc":
        # DNA coordinate order is A, C, G, T, so C/G are channels 1 and 2.
        return endpoint[..., 1:3].sum(dim=-1).mean(dim=-1)
    score1 = soft_motif_score(endpoint, motif, motif_tau)
    if reward == "motif":
        return score1
    score2 = soft_motif_score(endpoint, motif2, motif_tau)
    if conjunction_tau <= 0.0:
        raise ValueError("Conjunction temperature must be positive.")
    scores = torch.stack([score1, score2], dim=-1)
    return -float(conjunction_tau) * (
        torch.logsumexp(-scores / float(conjunction_tau), dim=-1) - math.log(2.0)
    )


def target_reward(
    endpoint: torch.Tensor,
    *,
    score_center: float,
    score_std: float,
    z_target: float,
    beta: float,
    reward: str,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    motif_tau: float,
    conjunction_tau: float,
    objective: str = "target",
) -> torch.Tensor:
    score = reward_score(
        endpoint,
        reward=reward,
        motif=motif,
        motif2=motif2,
        motif_tau=motif_tau,
        conjunction_tau=conjunction_tau,
    )
    z = (score - float(score_center)) / float(score_std)
    if objective == "maximize":
        return float(beta) * z
    if objective != "target":
        raise ValueError(f"Unknown reward objective: {objective}")
    return -float(beta) * (z - float(z_target)).square()


def estimate_value_gradient(
    model: DiTSequenceModel,
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
    nfe_value: int,
    mc_chunk: int,
    glass_end_time: float = 0.999,
    glass_solver: str = "euler",
    reward_objective: str = "target",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked exact-autodiff gradient of the finite-MC log-mean-exp value."""
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
            endpoint = glass_integrate_diff(
                model, terminal_noise, x_rep, t_rep, nfe_value, end_time=glass_end_time, solver=glass_solver
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
        endpoint = glass_integrate_diff(
            model, terminal_noise, x_rep, t_rep, nfe_value, end_time=glass_end_time, solver=glass_solver
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


@torch.no_grad()
def calibrate_score(
    model: DiTSequenceModel,
    cfg: SimpleNamespace,
    *,
    n_samples: int,
    batch_size: int,
    nfe: int,
    seed: int,
    device: torch.device,
    reward: str,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    motif_tau: float,
    conjunction_tau: float,
) -> np.ndarray:
    values: list[torch.Tensor] = []
    for start in range(0, n_samples, batch_size):
        count = min(batch_size, n_samples - start)
        x0 = seeded_randn((count, cfg.seq_len, cfg.alphabet_size), seed + start, device)
        endpoint = flow_to_time(model, cfg, x0, t_end=0.999, nfe=nfe)
        tokens = endpoint.argmax(dim=-1)
        hard = F.one_hot(tokens, num_classes=cfg.alphabet_size).float()
        score = reward_score(
            hard,
            reward=reward,
            motif=motif,
            motif2=motif2,
            motif_tau=motif_tau,
            conjunction_tau=conjunction_tau,
        )
        values.append(score.cpu())
    return torch.cat(values).numpy()


@torch.no_grad()
def make_probe_states(
    model: DiTSequenceModel,
    cfg: SimpleNamespace,
    *,
    n_probes: int,
    t_eval: float,
    nfe: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    x0 = seeded_randn((n_probes, cfg.seq_len, cfg.alphabet_size), seed, device)
    return flow_to_time(model, cfg, x0, t_end=t_eval, nfe=nfe)


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, len(values), size=(int(bootstrap), len(values)))
    means = values[sample_ids].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_length(
    length: int,
    path: str,
    args: argparse.Namespace,
    root: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"Loading L={length} checkpoint: {path}", flush=True)
    model, cfg = load_model(path, device)
    if int(cfg.seq_len) != int(length):
        raise ValueError(f"Checkpoint {path} has seq_len={cfg.seq_len}, expected L={length}.")
    if int(cfg.alphabet_size) != 4:
        raise ValueError(f"Expected DNA alphabet size 4, got {cfg.alphabet_size}.")

    out_dir = root / f"L{length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    motif = motif_tensor(args.motif, device)
    motif2 = motif_tensor(args.motif2, device)
    score_calibration = calibrate_score(
        model,
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
    score_mean = float(score_calibration.mean())
    score_std = float(score_calibration.std(ddof=1))
    if not np.isfinite(score_std) or score_std <= 0.0:
        raise RuntimeError(f"Invalid {args.reward} score standard deviation for L={length}: {score_std}")
    if args.reward_objective == "maximize":
        score_center = score_mean
        reward_z_target = 0.0
    elif args.reward == "gc":
        score_center = score_mean
        reward_z_target = float(args.z_target)
    else:
        score_center = float(np.quantile(score_calibration, args.target_percentile))
        reward_z_target = 0.0
    np.save(out_dir / "score_calibration_samples.npy", score_calibration)
    print(
        f"L={length}: {args.reward} calibration mean={score_mean:.6f}, std={score_std:.6f}, "
        f"center={score_center:.6f}",
        flush=True,
    )

    states = make_probe_states(
        model,
        cfg,
        n_probes=args.n_probes,
        t_eval=args.t_eval,
        nfe=args.nfe_probe,
        seed=args.seed + 1_000_000 + length,
        device=device,
    )
    rows: list[dict[str, float | int]] = []
    reference_gradients: list[torch.Tensor] = []
    estimate_gradients = (
        torch.empty(
            (args.n_probes, len(args.mc_values), args.n_repeats, length, cfg.alphabet_size), dtype=torch.float32
        )
        if args.store_gradient_tensors
        else None
    )
    for probe_idx in range(args.n_probes):
        x = states[probe_idx : probe_idx + 1]
        reference_eps = seeded_randn(
            (1, args.reference_mc, length, cfg.alphabet_size),
            args.seed + 10_000_000 + 10_000 * length + probe_idx,
            device,
        )
        _, reference_grad = estimate_value_gradient(
            model,
            x,
            t_eval=args.t_eval,
            eps_pool=reference_eps,
            score_center=score_center,
            score_std=score_std,
            z_target=reward_z_target,
            reward_beta=args.reward_beta,
            reward_scale=args.reward_scale,
            reward=args.reward,
            motif=motif,
            motif2=motif2,
            motif_tau=args.motif_tau,
            conjunction_tau=args.conjunction_tau,
            nfe_value=args.nfe_value,
            glass_end_time=args.glass_end_time,
            glass_solver=args.glass_solver,
            reward_objective=args.reward_objective,
            mc_chunk=args.mc_chunk,
        )
        reference_l2 = reference_grad.flatten().norm().item()
        reference_rms = reference_l2 / math.sqrt(length * cfg.alphabet_size)
        if args.store_gradient_tensors:
            reference_gradients.append(reference_grad.cpu())
        for mc_idx, mc in enumerate(args.mc_values):
            for repeat in range(args.n_repeats):
                eps_pool = seeded_randn(
                    (1, mc, length, cfg.alphabet_size),
                    args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe_idx + 100 * mc + repeat,
                    device,
                )
                _, estimate_grad = estimate_value_gradient(
                    model,
                    x,
                    t_eval=args.t_eval,
                    eps_pool=eps_pool,
                    score_center=score_center,
                    score_std=score_std,
                    z_target=reward_z_target,
                    reward_beta=args.reward_beta,
                    reward_scale=args.reward_scale,
                    reward=args.reward,
                    motif=motif,
                    motif2=motif2,
                    motif_tau=args.motif_tau,
                    conjunction_tau=args.conjunction_tau,
                    nfe_value=args.nfe_value,
                    glass_end_time=args.glass_end_time,
                    glass_solver=args.glass_solver,
                    reward_objective=args.reward_objective,
                    mc_chunk=args.mc_chunk,
                )
                error_l2 = (estimate_grad - reference_grad).flatten().norm().item()
                estimate_l2 = estimate_grad.flatten().norm().item()
                dot_product = (estimate_grad * reference_grad).sum().item()
                cosine_similarity = dot_product / max(estimate_l2 * reference_l2, 1e-12)
                if estimate_gradients is not None:
                    estimate_gradients[probe_idx, mc_idx, repeat] = estimate_grad.cpu()
                rows.append(
                    {
                        "length": length,
                        "probe": probe_idx,
                        "repeat": repeat,
                        "mc": mc,
                        "e_rms": error_l2 / math.sqrt(length * cfg.alphabet_size),
                        "error_l2": error_l2,
                        "reference_l2": reference_l2,
                        "reference_rms": reference_rms,
                        "estimate_l2": estimate_l2,
                        "gradient_dot_product": dot_product,
                        "cosine_similarity": cosine_similarity,
                        "relative_l2_error": error_l2 / max(reference_l2, 1e-12),
                    }
                )
        print(f"L={length}: probe {probe_idx + 1}/{args.n_probes}", flush=True)
        del reference_eps, reference_grad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "gradient_errors.csv", index=False)
    if estimate_gradients is not None:
        torch.save(
            {
                "reference_gradients": torch.cat(reference_gradients, dim=0),
                "estimate_gradients": estimate_gradients,
                "mc_values": torch.tensor(args.mc_values, dtype=torch.long),
                "layout": "estimate_gradients[probe, mc_index, repeat, position, alphabet]",
            },
            out_dir / "gradient_pairs.pt",
        )
    summary_rows: list[dict[str, float | int]] = []
    for mc, group in raw.groupby("mc", sort=True):
        # Repeat estimates share a reference; bootstrap at the independent probe level.
        probe_means = group.groupby("probe")["e_rms"].mean().to_numpy()
        ci_low, ci_high = bootstrap_ci(probe_means, args.bootstrap, args.seed + length + int(mc))
        l2_probe_means = group.groupby("probe")["error_l2"].mean().to_numpy()
        l2_ci_low, l2_ci_high = bootstrap_ci(l2_probe_means, args.bootstrap, args.seed + 10_000 + length + int(mc))
        relative_probe_means = group.groupby("probe")["relative_l2_error"].mean().to_numpy()
        relative_ci_low, relative_ci_high = bootstrap_ci(
            relative_probe_means, args.bootstrap, args.seed + 20_000 + length + int(mc)
        )
        summary_rows.append(
            {
                "length": length,
                "mc": int(mc),
                "mean_e_rms": float(probe_means.mean()),
                "std_across_probes": float(probe_means.std(ddof=1)) if len(probe_means) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_error_l2": float(l2_probe_means.mean()),
                "error_l2_ci95_low": l2_ci_low,
                "error_l2_ci95_high": l2_ci_high,
                "mean_relative_l2_error": float(relative_probe_means.mean()),
                "relative_l2_error_ci95_low": relative_ci_low,
                "relative_l2_error_ci95_high": relative_ci_high,
                "n_probes": int(args.n_probes),
                "n_repeats": int(args.n_repeats),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "gradient_error_summary.csv", index=False)
    metadata: dict[str, object] = {
        "length": length,
        "checkpoint": path,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_center": score_center,
        "dimensions": int(length * cfg.alphabet_size),
        "reference_mc": args.reference_mc,
        "normalization": "E_RMS = ||gradient_estimate - gradient_reference||_2 / sqrt(4L)",
        "reference": "independent finite-MC GLASS estimator with reference_mc samples",
        "reward": {
            "name": args.reward,
            "objective": args.reward_objective,
            "z_target": reward_z_target,
            "beta": args.reward_beta,
            "lambda": args.reward_scale,
            "motif": args.motif,
            "motif2": args.motif2 if args.reward == "conjunction" else None,
            "motif_tau": args.motif_tau if args.reward != "gc" else None,
            "conjunction_tau": args.conjunction_tau if args.reward == "conjunction" else None,
            "target_percentile": args.target_percentile if args.reward != "gc" else None,
        },
        "t_eval": args.t_eval,
        "nfe_probe": args.nfe_probe,
        "nfe_value": args.nfe_value,
        "glass_end_time": args.glass_end_time,
        "glass_solver": args.glass_solver,
        "n_probes": args.n_probes,
        "n_repeats": args.n_repeats,
        "mc_values": args.mc_values,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return summary, metadata


def make_figures(root: Path, summary: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for length, group in summary.groupby("length", sort=True):
        group = group.sort_values("mc")
        axis.plot(group["mc"], group["mean_e_rms"], marker="o", label=f"L={length}")
        axis.fill_between(group["mc"], group["ci95_low"], group["ci95_high"], alpha=0.18)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Monte Carlo samples per value-gradient estimate")
    axis.set_ylabel(r"normalized gradient error $E_{RMS}$")
    axis.grid(alpha=0.25)
    axis.legend(title="sequence length")
    fig.tight_layout()
    fig.savefig(root / "gradient_error_vs_mc.png", dpi=220)
    plt.close(fig)

    required_rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        for length, group in summary.groupby("length", sort=True):
            group = group.sort_values("mc")
            mean_ok = group[group["mean_e_rms"] <= threshold]
            conservative_ok = group[group["ci95_high"] <= threshold]
            required_rows.append(
                {
                    "length": int(length),
                    "threshold": float(threshold),
                    "criterion": "mean",
                    "required_mc": int(mean_ok.iloc[0]["mc"]) if len(mean_ok) else -1,
                }
            )
            required_rows.append(
                {
                    "length": int(length),
                    "threshold": float(threshold),
                    "criterion": "upper_95_ci",
                    "required_mc": int(conservative_ok.iloc[0]["mc"]) if len(conservative_ok) else -1,
                }
            )
    required = pd.DataFrame(required_rows)
    required.to_csv(root / "required_mc_by_threshold.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    plotted = required[required["criterion"] == "upper_95_ci"]
    for threshold, group in plotted.groupby("threshold", sort=True):
        valid = group[group["required_mc"] > 0].sort_values("length")
        if len(valid):
            axis.plot(valid["length"], valid["required_mc"], marker="o", label=fr"$\delta={threshold:g}$")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("sequence length L")
    axis.set_ylabel("required MC samples (upper 95% CI below threshold)")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(title="error threshold")
    fig.tight_layout()
    fig.savefig(root / "required_mc_vs_length.png", dpi=220)
    plt.close(fig)
    return required


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.t_eval < 1.0:
        raise ValueError("--t_eval must be strictly between zero and one.")
    if max(args.mc_values) >= args.reference_mc:
        raise ValueError("--reference_mc must exceed every --mc_values entry.")
    if any(mc <= 0 for mc in args.mc_values):
        raise ValueError("--mc_values must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoints = parse_checkpoints(args.ckpt)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()

    summaries: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for length in args.lengths:
        if length not in checkpoints:
            raise ValueError(f"No checkpoint configured for L={length}.")
        summary, result_metadata = run_length(length, checkpoints[length], args, root, device)
        summaries.append(summary)
        metadata.append(result_metadata)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(root / "gradient_error_summary_all_lengths.csv", index=False)
    required = make_figures(root, all_summary, args.thresholds)
    run_metadata = {
        "args": vars(args),
        "per_length": metadata,
        "elapsed_seconds": time.time() - start,
        "required_mc_definition": "smallest tested N whose bootstrap upper 95% CI for mean E_RMS across probes is <= delta",
    }
    with open(root / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True)
    print("\nGradient accuracy summary:")
    print(all_summary.to_string(index=False))
    print("\nRequired MC:")
    print(required.to_string(index=False))
    print(f"Outputs written to {root.resolve()} in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
