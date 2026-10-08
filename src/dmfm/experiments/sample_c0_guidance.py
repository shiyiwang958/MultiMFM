#!/usr/bin/env python
"""Ported from dirichlet-flow-matching/scripts/sample_yeast_c0_guidance_hist.py
(recovered in scratch_dfm_recon_20260925; produces Table 1 / Fig 3 / Fig 7 samples).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

from dmfm import paths
from dmfm.regressors.c0 import ParkC0Regressor
from dmfm.models.dna_models import DiTSequenceModel
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step
from dmfm.utils.torch_io import torch_load


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired unguided/guided yeast sampling with C0 histogram output.")
    p.add_argument("--ckpt", required=True)
    # Original default was the purged TensorFlow-converted dinko/C0free_torch.pt;
    # every paper run passed the parent-disjoint guide explicitly.
    p.add_argument("--c0_ckpt", default=str(paths.c0_guide()))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default="yeast_c0_guidance_hist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--target_c0", type=float, default=0.30)
    p.add_argument(
        "--target_c0_second",
        type=float,
        default=None,
        help=(
            "Optional second C0 target. If set, reward is a smooth either/or "
            "mixture: log(0.5 exp(r_target1) + 0.5 exp(r_target2))."
        ),
    )
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--mc", type=int, default=128)
    p.add_argument(
        "--mc_chunk",
        type=int,
        default=16,
        help="Chunk size for MC guidance gradient to reduce GPU memory (<= mc).",
    )
    p.add_argument("--nfe_traj", type=int, default=128)
    p.add_argument("--nfe_value", type=int, default=12)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--guide_t_start", type=float, default=0.50)
    p.add_argument("--guide_t_end", type=float, default=0.95)
    p.add_argument("--guidance_frac", type=float, default=1.0)
    p.add_argument("--coeff_cap", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--taper", action="store_true")
    p.add_argument("--hist_bins", type=int, default=50)
    p.add_argument("--use_flash_attn", action="store_true")
    return p.parse_args(argv)


def init_distributed() -> tuple[int, int, int, torch.device]:
    # Default: run single-process. Only enable DDP when explicitly requested, to
    # avoid accidental multiproc launches via inherited env vars.
    enable = os.environ.get("ENABLE_DISTRIBUTED", "0").lower() in {"1", "true", "yes", "y"}

    if enable and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if has_cuda:
            # Guard against misconfigured nproc_per_node vs allocated GPUs.
            local_rank = min(local_rank, torch.cuda.device_count() - 1)
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return rank, world_size, local_rank, torch.device("cuda", local_rank)

        dist.init_process_group(backend="gloo")
        return rank, world_size, local_rank, torch.device("cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seeded_randn(shape, seed: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=dtype).to(device)


def bcast_like(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while coef.ndim < x.ndim:
        coef = coef[..., None]
    return coef.to(device=x.device, dtype=x.dtype)


def dna_probs_to_park_input(x_acgt: torch.Tensor) -> torch.Tensor:
    x_atgc = x_acgt[..., [0, 3, 2, 1]]
    return x_atgc.reshape(x_acgt.shape[0], 200, 1)


def cyclizability_reward(c0_model: ParkC0Regressor, x_acgt: torch.Tensor) -> torch.Tensor:
    # ParkC0Regressor.forward() applies the same ACGT->ATGC permute + reshape
    # (B, 200, 1) + transpose internally, in the same order, so this is the
    # bit-for-bit equivalent of the original dinko C0 wrapper call.
    return c0_model(x_acgt)


def target_reward(
    scores: torch.Tensor,
    target_c0: float,
    target_c0_second: float | None,
    reward_sigma: float,
) -> torch.Tensor:
    r1 = -0.5 * ((scores - float(target_c0)) / float(reward_sigma)).pow(2)
    if target_c0_second is None:
        return r1

    r2 = -0.5 * ((scores - float(target_c0_second)) / float(reward_sigma)).pow(2)
    # This is a smooth max / equal mixture over target modes. A plain r1 + r2
    # would reward the midpoint between targets, which is exactly not what we want.
    return torch.logsumexp(torch.stack([r1, r2], dim=0), dim=0) - math.log(2.0)


def target_distances_np(scores: np.ndarray, targets: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    return np.min(np.abs(arr[:, None] - target_arr[None, :]), axis=1)


def score_hard(c0_model: ParkC0Regressor, x: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    hard = F.one_hot(x.argmax(-1), alphabet_size).float()
    return cyclizability_reward(c0_model, hard)


def hard_token_strings(x: torch.Tensor, alphabet_size: int) -> list[str]:
    """Discrete sequences matching `score_hard` (argmax over last dim)."""
    idx = x.argmax(-1).detach().cpu().numpy()
    if alphabet_size == 4:
        table = np.array(["A", "C", "G", "T"], dtype="U1")
        return ["".join(table[row].tolist()) for row in idx]
    return [" ".join(str(int(t)) for t in row) for row in idx]


def glass_velocity_diff(
    gen_model: DiTSequenceModel,
    s: torch.Tensor,
    Is: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ss = bcast_like(s, Is)
    tt = bcast_like(t_cond, Is)
    oms = 1.0 - ss

    denom = (tt.pow(2) * oms.pow(2) + (1.0 - tt).pow(2) * ss.pow(2)).clamp_min(eps)
    P = (1.0 - tt).pow(2) / denom
    sqrtP = P.sqrt()

    t_star = 1.0 / (1.0 + oms * sqrtP)
    coeff_cond = t_star * oms.pow(2) * tt / denom
    coeff_Is = t_star * ss * P
    x_star = coeff_cond * x_cond + coeff_Is * Is

    t_star_vec = t_star.reshape(Is.shape[0], -1)[:, 0]
    v_star = gen_model.v(t_star_vec, t_star_vec, x_star, t_cond, x_cond)

    term2 = t_star * sqrtP * v_star
    diff_div_x = ((1.0 - tt).pow(2) * (1.0 + ss) - tt.pow(2) * oms) / denom
    b_minus_1_div_x = (diff_div_x - (P + sqrtP)) / (1.0 + oms * sqrtP)
    a_div_x = t_star * oms * tt / denom

    return a_div_x * x_cond + b_minus_1_div_x * Is + term2


def glass_integrate_diff(
    gen_model: DiTSequenceModel,
    eps0: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    x = eps0
    grid = torch.linspace(0, 1, n_steps + 1, device=x.device, dtype=x.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        s = torch.full((x.shape[0],), float(s0), device=x.device, dtype=x.dtype)
        x = x + float(s1 - s0) * glass_velocity_diff(gen_model, s, x, x_cond, t_cond)
    return x


def estimate_v_and_grad(
    gen_model: DiTSequenceModel,
    c0_model: ParkC0Regressor,
    x: torch.Tensor,
    t_scalar: float,
    eps_pool: torch.Tensor,
    target_c0: float,
    target_c0_second: float | None,
    reward_sigma: float,
    reward_scale: float,
    nfe_value: int,
    mc_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    mc = eps_pool.shape[1]
    mc_chunk = int(max(1, min(mc_chunk, mc)))

    # Two-pass, chunked MC estimator:
    # 1) compute rewards for all eps with no grad to get softmax weights
    # 2) recompute per-chunk rewards with grad and accumulate weighted gradient
    with torch.no_grad():
        rewards_all: list[torch.Tensor] = []
        for j0 in range(0, mc, mc_chunk):
            j1 = min(mc, j0 + mc_chunk)
            eps_rep = eps_pool[:, j0:j1].reshape(batch * (j1 - j0), *x.shape[1:])
            x_rep = x.detach().repeat_interleave(j1 - j0, 0)
            t_rep = torch.full((batch * (j1 - j0),), float(t_scalar), device=x.device, dtype=x.dtype)
            x1 = glass_integrate_diff(gen_model, eps_rep, x_rep, t_rep, n_steps=nfe_value)
            soft_scores = cyclizability_reward(c0_model, x1)
            reward = target_reward(soft_scores, target_c0, target_c0_second, reward_sigma)
            rewards_all.append(reward.view(batch, j1 - j0))
        reward_full = reward_scale * torch.cat(rewards_all, dim=1)  # (batch, mc)
        v = torch.logsumexp(reward_full, dim=1) - math.log(mc)
        weights = torch.softmax(reward_full, dim=1)  # (batch, mc)

    x_leaf = x.detach().clone().requires_grad_(True)
    grad_accum = torch.zeros_like(x_leaf)

    for j0 in range(0, mc, mc_chunk):
        j1 = min(mc, j0 + mc_chunk)
        eps_rep = eps_pool[:, j0:j1].reshape(batch * (j1 - j0), *x.shape[1:])
        x_rep = x_leaf.repeat_interleave(j1 - j0, 0)
        t_rep = torch.full((batch * (j1 - j0),), float(t_scalar), device=x.device, dtype=x.dtype)

        x1 = glass_integrate_diff(gen_model, eps_rep, x_rep, t_rep, n_steps=nfe_value)
        soft_scores = cyclizability_reward(c0_model, x1)
        reward = target_reward(soft_scores, target_c0, target_c0_second, reward_sigma)
        reward = reward_scale * reward.view(batch, j1 - j0)

        w = weights[:, j0:j1].detach()
        obj = (w * reward).sum()
        g = torch.autograd.grad(obj, x_leaf, retain_graph=False, create_graph=False)[0]
        grad_accum = grad_accum + g

    return v.detach(), grad_accum.detach()


def sample_paired_batch(
    sample_ids: list[int],
    args: argparse.Namespace,
    gen_model: DiTSequenceModel,
    c0_model: ParkC0Regressor,
    device: torch.device,
    seq_len: int,
    alphabet_size: int,
) -> list[dict[str, Any]]:
    batch = len(sample_ids)
    x0 = torch.cat(
        [seeded_randn((1, seq_len, alphabet_size), args.seed + idx, device) for idx in sample_ids],
        dim=0,
    )
    eps_pool = torch.stack(
        [
            seeded_randn(
                (args.mc, seq_len, alphabet_size),
                args.seed + 1_000_000 + idx,
                device,
            )
            for idx in sample_ids
        ],
        dim=0,
    )

    x_base = x0.clone()
    x_guided = x0.clone()
    grid = torch.linspace(0, args.t_max, args.nfe_traj + 1, device=device)

    for s0, s1 in zip(grid[:-1], grid[1:]):
        t_now = float(s0)
        s = s0.expand(batch)
        t_next = s1.expand(batch)
        dt = float(s1 - s0)

        with torch.no_grad():
            x_base, _, _ = gaussian_denoiser_flow_step(args, gen_model, x_base, s, t_next)
            x_guided_base, _, _ = gaussian_denoiser_flow_step(args, gen_model, x_guided, s, t_next)

        do_guide = args.guide_t_start <= t_now <= args.guide_t_end
        if do_guide:
            _, g = estimate_v_and_grad(
                gen_model,
                c0_model,
                x_guided,
                t_now,
                eps_pool,
                target_c0=args.target_c0,
                target_c0_second=args.target_c0_second,
                reward_sigma=args.reward_sigma,
                reward_scale=args.reward_scale,
                nfe_value=args.nfe_value,
                mc_chunk=args.mc_chunk,
            )
            gnorm = g.flatten(1).norm(dim=1).clamp_min(1e-8)
            if args.grad_clip is not None:
                g = g * (args.grad_clip / gnorm).clamp(max=1.0)[:, None, None]

            coeff = args.guidance_frac * gen_model.sde_sigma_sq(s)
            if args.coeff_cap is not None:
                coeff = coeff.clamp(max=args.coeff_cap)
            if args.taper:
                width = max(args.guide_t_end - args.guide_t_start, 1e-8)
                taper = ((args.guide_t_end - s) / width).clamp(min=0.0, max=1.0)
                coeff = coeff * taper

            guidance = coeff[:, None, None] * g
            x_guided = x_guided_base + dt * guidance
        else:
            x_guided = x_guided_base

    with torch.no_grad():
        unguided_scores = score_hard(c0_model, x_base, alphabet_size).detach().cpu().numpy()
        guided_scores = score_hard(c0_model, x_guided, alphabet_size).detach().cpu().numpy()
        seq_unguided = hard_token_strings(x_base, alphabet_size)
        seq_guided = hard_token_strings(x_guided, alphabet_size)

    targets = [float(args.target_c0)]
    if args.target_c0_second is not None:
        targets.append(float(args.target_c0_second))
    unguided_dist = target_distances_np(unguided_scores, targets)
    guided_dist = target_distances_np(guided_scores, targets)

    rows = []
    for idx, ung, gui, ung_dist, gui_dist, su, sg in zip(
        sample_ids,
        unguided_scores,
        guided_scores,
        unguided_dist,
        guided_dist,
        seq_unguided,
        seq_guided,
    ):
        rows.append(
            {
                "sample_idx": int(idx),
                "seed": int(args.seed + idx),
                "unguided": float(ung),
                "guided": float(gui),
                "delta": float(gui - ung),
                "unguided_abs_to_target": float(ung_dist),
                "guided_abs_to_target": float(gui_dist),
                "abs_to_target_improvement": float(ung_dist - gui_dist),
                "seq_unguided": su,
                "seq_guided": sg,
            }
        )
    return rows


def summarize_scores(scores: np.ndarray, targets: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    d = target_distances_np(arr, targets)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean_abs_to_target": float(d.mean()),
        "frac_within_0.05": float((d <= 0.05).mean()),
        "frac_within_0.10": float((d <= 0.10).mean()),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world_size, local_rank, device = init_distributed()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.output_dir is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = str(paths.OUTPUTS / f"{args.run_name}_{stamp}")
    out_dir = Path(args.output_dir)

    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir.resolve()}", flush=True)
    if world_size > 1:
        dist.barrier()

    ckpt = torch_load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["model_cfg"]
    if isinstance(model_cfg, SimpleNamespace):
        cfg = model_cfg
    else:
        cfg = SimpleNamespace(**model_cfg)
    cfg.use_flash_attn = bool(args.use_flash_attn or getattr(cfg, "use_flash_attn", False))
    cfg.flow_temp = float(getattr(cfg, "flow_temp", 1.0))
    args.gaussian_beta_schedule = getattr(cfg, "gaussian_beta_schedule", "linear")
    args.gaussian_beta_table_path = getattr(cfg, "gaussian_beta_table_path", None)
    args.flow_temp = float(getattr(cfg, "flow_temp", 1.0))

    gen_model = DiTSequenceModel(cfg, alphabet_size=cfg.alphabet_size).to(device)
    gen_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    gen_model.eval()

    c0_model = ParkC0Regressor().to(device).eval()
    c0_state = torch_load(args.c0_ckpt, map_location=device)
    c0_model.load_state_dict(c0_state, strict=True)
    for p in c0_model.parameters():
        p.requires_grad_(False)

    if is_main_process(rank):
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        targets = [float(args.target_c0)]
        if args.target_c0_second is not None:
            targets.append(float(args.target_c0_second))
        print(
            f"Sampling {args.n_samples} paired trajectories "
            f"on world_size={world_size}, batch_size={args.batch_size}, "
            f"guide=[{args.guide_t_start}, {args.guide_t_end}], "
            f"targets={targets}",
            flush=True,
        )

    local_ids = list(range(rank, args.n_samples, world_size))
    local_rows: list[dict[str, Any]] = []

    for start in range(0, len(local_ids), args.batch_size):
        sample_ids = local_ids[start : start + args.batch_size]
        rows = sample_paired_batch(
            sample_ids=sample_ids,
            args=args,
            gen_model=gen_model,
            c0_model=c0_model,
            device=device,
            seq_len=cfg.seq_len,
            alphabet_size=cfg.alphabet_size,
        )
        local_rows.extend(rows)
        if is_main_process(rank):
            done = min(start + len(sample_ids), len(local_ids))
            print(f"rank0 progress: {done}/{len(local_ids)} local samples", flush=True)
        torch.cuda.empty_cache()

    if world_size > 1:
        gathered: list[list[dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_rows)
        all_rows = [row for part in gathered for row in part]
    else:
        all_rows = local_rows

    if is_main_process(rank):
        df = pd.DataFrame(all_rows).sort_values("sample_idx").reset_index(drop=True)
        df.to_csv(out_dir / "sample_scores.csv", index=False)

        with open(out_dir / "sequences_unguided.fa", "w") as f_ung, open(
            out_dir / "sequences_guided.fa", "w"
        ) as f_gui:
            for _, row in df.iterrows():
                sid = int(row["sample_idx"])
                f_ung.write(f">sample_{sid} seed={int(row['seed'])}\n{row['seq_unguided']}\n")
                f_gui.write(f">sample_{sid} seed={int(row['seed'])}\n{row['seq_guided']}\n")

        unguided = df["unguided"].to_numpy()
        guided = df["guided"].to_numpy()
        targets = [float(args.target_c0)]
        if args.target_c0_second is not None:
            targets.append(float(args.target_c0_second))

        summary = {
            "n_samples": int(len(df)),
            "targets": targets,
            "unguided": summarize_scores(unguided, targets),
            "guided": summarize_scores(guided, targets),
            "delta": {
                "mean": float(df["delta"].mean()),
                "std": float(df["delta"].std(ddof=0)),
                "median": float(df["delta"].median()),
                "p05": float(df["delta"].quantile(0.05)),
                "p95": float(df["delta"].quantile(0.95)),
                "frac_score_increased": float((df["delta"] > 0).mean()),
                "abs_to_target_improvement_mean": float(df["abs_to_target_improvement"].mean()),
                "abs_to_target_improvement_median": float(df["abs_to_target_improvement"].median()),
                "frac_improved": float((df["abs_to_target_improvement"] > 0).mean()),
            },
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        bins = np.linspace(
            min(float(unguided.min()), float(guided.min())),
            max(float(unguided.max()), float(guided.max())),
            args.hist_bins,
        )
        plt.figure(figsize=(8, 4.5))
        plt.hist(unguided, bins=bins, alpha=0.55, density=True, label="unguided")
        plt.hist(guided, bins=bins, alpha=0.55, density=True, label="guided")
        for i, target in enumerate(targets):
            label = f"target {target:g}" if i == 0 else f"target2 {target:g}"
            plt.axvline(target, color="black", linestyle="--", label=label)
        plt.xlabel("Cyclizability C0 score")
        plt.ylabel("density")
        plt.title("Unguided vs guided score distribution")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "histogram.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        xs = np.sort(unguided)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label="unguided")
        xs = np.sort(guided)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label="guided")
        for i, target in enumerate(targets):
            label = f"target {target:g}" if i == 0 else f"target2 {target:g}"
            plt.axvline(target, color="black", linestyle="--", label=label)
        plt.xlabel("Cyclizability C0 score")
        plt.ylabel("empirical CDF")
        plt.title("Unguided vs guided empirical CDF")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "cdf.png", dpi=200)
        plt.close()

        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
