#!/usr/bin/env python
"""Paired reward-attainment sampling for the parent-disjoint DNA models.

Rewards are the analytic motif and two-motif-conjunction objectives from the
gradient-accuracy rebuttal. Guidance uses a single composed dMFM posterior map
for every Monte Carlo future. Reward attainment is evaluated exactly on the
decoded sequence, while diversity is compared with the parent-disjoint test
partition.

Ported from dirichlet-flow-matching/scripts/sample_parent_reward_attainment.py (Table 13).

Port change (default only): the original imported ``DEFAULT_DMFMS`` from
``ablate_dmfm_one_step_gradient_mc``. When Table 13 ran (2026-07-26) that map held the
selected diagonal dMFM students (recorded as ``dmfm_checkpoint`` in every Table 13
``metadata.json``); it was repointed at the 4-step students the next day. The port's
default is therefore the diagonal students (``TABLE13_DMFMS``), i.e. what the paper used.
Pass ``--dmfm_ckpt L=PATH`` to use another student.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmfm import paths
from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import load_dmfm, one_step_dmfm
from dmfm.experiments.ablate_glass_gradient_mc import (
    DEFAULT_CKPTS,
    calibrate_score,
    load_model,
    motif_tensor,
    reward_score,
    target_reward,
)
from dmfm.utils.model_loading import load_data_seqs
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step


DNA = np.asarray(list("ACGT"))

# Selected diagonal dMFM students used by Table 13 (see module docstring).
TABLE13_DMFMS = {L: str(paths.dmfm_ckpt(L)) for L in paths.LENGTHS}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--teacher_ckpt", action="append", default=[], metavar="L=PATH")
    parser.add_argument("--dmfm_ckpt", action="append", default=[], metavar="L=PATH")
    parser.add_argument("--reward", choices=["motif", "conjunction"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nfe_traj", type=int, default=64)
    parser.add_argument("--t_end", type=float, default=0.999)
    parser.add_argument("--guide_t_start", type=float, default=0.50)
    parser.add_argument("--guide_t_end", type=float, default=0.95)
    parser.add_argument("--mc", type=int, default=8)
    parser.add_argument("--mc_chunk", type=int, default=4)
    parser.add_argument("--guidance_frac", type=float, default=8.0)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--coeff_cap", type=float, default=10.0)
    parser.add_argument("--calibration_samples", type=int, default=1024)
    parser.add_argument("--nfe_calibration", type=int, default=64)
    parser.add_argument("--calibration_batch_size", type=int, default=32)
    parser.add_argument("--target_percentile", type=float, default=0.90)
    parser.add_argument("--reward_beta", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--motif", default="TTTTTC")
    parser.add_argument("--motif2", default="AAAATT")
    parser.add_argument("--motif_tau", type=float, default=0.10)
    parser.add_argument("--conjunction_tau", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def parse_paths(raw: list[str], defaults: dict[int, str], flag: str) -> dict[int, str]:
    paths = dict(defaults)
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Invalid {flag} {spec!r}; expected L=PATH.")
        length, path = spec.split("=", 1)
        paths[int(length)] = path
    return paths


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def strings(tokens: torch.Tensor) -> list[str]:
    return ["".join(DNA[row].tolist()) for row in tokens.detach().cpu().numpy()]


def value_gradient(
    student,
    x: torch.Tensor,
    t: torch.Tensor,
    eps_pool: torch.Tensor,
    *,
    score_center: float,
    score_std: float,
    reward: str,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    motif_tau: float,
    conjunction_tau: float,
    reward_beta: float,
    reward_scale: float,
    mc_chunk: int,
) -> torch.Tensor:
    """Finite-MC gradient of the one-map dMFM value estimate."""
    batch, mc = eps_pool.shape[:2]
    chunk = max(1, min(int(mc_chunk), mc))
    with torch.no_grad():
        log_rewards = []
        for start in range(0, mc, chunk):
            end = min(start + chunk, mc)
            width = end - start
            endpoint = one_step_dmfm(
                student,
                eps_pool[:, start:end].reshape(batch * width, *x.shape[1:]),
                x.detach().repeat_interleave(width, dim=0),
                t.repeat_interleave(width),
            )
            reward_value = target_reward(
                endpoint,
                score_center=score_center,
                score_std=score_std,
                z_target=0.0,
                beta=reward_beta,
                reward=reward,
                motif=motif,
                motif2=motif2,
                motif_tau=motif_tau,
                conjunction_tau=conjunction_tau,
            )
            log_rewards.append((reward_scale * reward_value).view(batch, width))
        weights = torch.softmax(torch.cat(log_rewards, dim=1), dim=1)

    x_leaf = x.detach().clone().requires_grad_(True)
    gradient = torch.zeros_like(x_leaf)
    for start in range(0, mc, chunk):
        end = min(start + chunk, mc)
        width = end - start
        endpoint = one_step_dmfm(
            student,
            eps_pool[:, start:end].reshape(batch * width, *x.shape[1:]),
            x_leaf.repeat_interleave(width, dim=0),
            t.repeat_interleave(width),
        )
        reward_value = target_reward(
            endpoint,
            score_center=score_center,
            score_std=score_std,
            z_target=0.0,
            beta=reward_beta,
            reward=reward,
            motif=motif,
            motif2=motif2,
            motif_tau=motif_tau,
            conjunction_tau=conjunction_tau,
        ).view(batch, width)
        gradient = gradient + torch.autograd.grad((weights[:, start:end] * reward_scale * reward_value).sum(), x_leaf)[0]
    return gradient.detach()


def paired_generate(
    teacher,
    student,
    cfg,
    *,
    length: int,
    cli: argparse.Namespace,
    score_center: float,
    score_std: float,
    motif: torch.Tensor,
    motif2: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    unguided_parts, guided_parts = [], []
    for start in range(0, cli.n_samples, cli.batch_size):
        count = min(cli.batch_size, cli.n_samples - start)
        generator = torch.Generator(device=device).manual_seed(cli.seed + 100_000 * length + start)
        x_unguided = torch.randn((count, length, 4), generator=generator, device=device)
        x_guided = x_unguided.clone()
        eps_pool = torch.randn((count, cli.mc, length, 4), generator=generator, device=device)
        grid = torch.linspace(0.0, cli.t_end, cli.nfe_traj + 1, device=device)
        for s0, s1 in zip(grid[:-1], grid[1:]):
            s = s0.expand(count)
            with torch.no_grad():
                x_unguided, _, _ = gaussian_denoiser_flow_step(cfg, teacher, x_unguided, s, s1.expand(count))
                x_base, _, _ = gaussian_denoiser_flow_step(cfg, teacher, x_guided, s, s1.expand(count))
            if cli.guide_t_start <= float(s0) <= cli.guide_t_end:
                gradient = value_gradient(
                    student, x_guided, s, eps_pool,
                    score_center=score_center, score_std=score_std, reward=cli.reward,
                    motif=motif, motif2=motif2, motif_tau=cli.motif_tau,
                    conjunction_tau=cli.conjunction_tau, reward_beta=cli.reward_beta,
                    reward_scale=cli.reward_scale, mc_chunk=cli.mc_chunk,
                )
                norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-8)
                gradient = gradient * (cli.grad_clip / norm).clamp(max=1.0)[:, None, None]
                coefficient = (cli.guidance_frac * teacher.sde_sigma_sq(s)).clamp(max=cli.coeff_cap)
                x_guided = x_base + float(s1 - s0) * coefficient[:, None, None] * gradient
            else:
                x_guided = x_base
        unguided_parts.append(x_unguided.argmax(dim=-1).cpu())
        guided_parts.append(x_guided.argmax(dim=-1).cpu())
        print(f"L={length}: generated {start + count}/{cli.n_samples}", flush=True)
    return torch.cat(unguided_parts), torch.cat(guided_parts)


def diversity_metrics(generated: torch.Tensor, test_sequences: torch.Tensor) -> dict[str, float]:
    count, length = generated.shape
    pairwise = (generated[:, None, :] != generated[None, :, :]).float().mean(dim=-1)
    upper = pairwise[torch.triu_indices(count, count, offset=1).unbind()]
    nearest = torch.full((count,), float("inf"))
    for start in range(0, len(test_sequences), 512):
        distance = (generated[:, None, :] != test_sequences[start : start + 512][None, :, :]).float().mean(dim=-1)
        nearest = torch.minimum(nearest, distance.min(dim=1).values)
    return {
        "pairwise_hamming_frac": float(upper.mean()),
        "nearest_test_hamming_frac": float(nearest.mean()),
        "exact_unique_frac": float(torch.unique(generated, dim=0).shape[0] / count),
        "length": int(length),
    }


@torch.inference_mode()
def score_tokens(tokens: torch.Tensor, *, cli: argparse.Namespace, motif: torch.Tensor, motif2: torch.Tensor, score_center: float, score_std: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    one_hot = F.one_hot(tokens.to(device), num_classes=4).float()
    score = reward_score(one_hot, reward=cli.reward, motif=motif, motif2=motif2, motif_tau=cli.motif_tau, conjunction_tau=cli.conjunction_tau)
    shaped = target_reward(
        one_hot, score_center=score_center, score_std=score_std, z_target=0.0,
        beta=cli.reward_beta, reward=cli.reward, motif=motif, motif2=motif2,
        motif_tau=cli.motif_tau, conjunction_tau=cli.conjunction_tau,
    )
    return score.cpu().numpy(), shaped.cpu().numpy()


def main(argv=None) -> None:
    cli = parse_args(argv)
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if not 0.0 < cli.t_end < 1.0:
        raise ValueError("--t_end must lie in (0, 1)")
    device = torch.device(cli.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    teacher_paths = parse_paths(cli.teacher_ckpt, DEFAULT_CKPTS, "--teacher_ckpt")
    student_paths = parse_paths(cli.dmfm_ckpt, TABLE13_DMFMS, "--dmfm_ckpt")
    root = Path(cli.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_summary = []

    for length in cli.lengths:
        teacher, cfg = load_model(teacher_paths[length], device)
        student = load_dmfm(student_paths[length], device)
        motif = motif_tensor(cli.motif, device)
        motif2 = motif_tensor(cli.motif2, device)
        calibration = calibrate_score(
            teacher, cfg, n_samples=cli.calibration_samples, batch_size=cli.calibration_batch_size,
            nfe=cli.nfe_calibration, seed=cli.seed + 10_000 * length, device=device,
            reward=cli.reward, motif=motif, motif2=motif2, motif_tau=cli.motif_tau,
            conjunction_tau=cli.conjunction_tau,
        )
        score_center = float(np.quantile(calibration, cli.target_percentile))
        score_std = float(calibration.std(ddof=1))
        if not np.isfinite(score_std) or score_std <= 0:
            raise RuntimeError(f"Invalid score standard deviation at L={length}: {score_std}")
        print(f"L={length}: target score={score_center:.6f}; scale={score_std:.6f}", flush=True)
        unguided, guided = paired_generate(
            teacher, student, cfg, length=length, cli=cli, score_center=score_center,
            score_std=score_std, motif=motif, motif2=motif2, device=device,
        )
        # Port fix: the original path data/yeast_parent_disjoint/... is data/dna/... here.
        test_sequences = load_data_seqs(
            paths.data_pt(length),
            split_pt=paths.split_pt(length),
            split="test",
        )
        reference_ids = torch.randperm(len(test_sequences), generator=torch.Generator().manual_seed(cli.seed + length))[: cli.n_samples]
        test_reference = test_sequences[reference_ids]
        scores = {}
        for name, tokens in {"unguided": unguided, "guided": guided, "heldout_test": test_reference}.items():
            raw_score, shaped_reward = score_tokens(
                tokens, cli=cli, motif=motif, motif2=motif2, score_center=score_center,
                score_std=score_std, device=device,
            )
            scores[name] = (raw_score, shaped_reward)

        paired_error_delta = np.abs(scores["unguided"][0] - score_center) - np.abs(scores["guided"][0] - score_center)
        delta_low, delta_high = bootstrap_mean(paired_error_delta, cli.bootstrap, cli.seed + length)
        per_sample = pd.DataFrame({
            "length": length,
            "sample_id": np.arange(cli.n_samples),
            "sequence_unguided": strings(unguided),
            "sequence_guided": strings(guided),
            "score_unguided": scores["unguided"][0],
            "score_guided": scores["guided"][0],
            "reward_unguided": scores["unguided"][1],
            "reward_guided": scores["guided"][1],
            "target_score": score_center,
        })
        out_dir = root / f"L{length}"
        out_dir.mkdir(exist_ok=True)
        per_sample.to_csv(out_dir / "reward_attainment_samples.csv", index=False)
        pd.DataFrame({"length": length, "test_id": reference_ids.numpy(), "sequence": strings(test_reference), "score": scores["heldout_test"][0], "reward": scores["heldout_test"][1]}).to_csv(out_dir / "heldout_test_reference.csv", index=False)
        rows = []
        for name, tokens in {"unguided": unguided, "guided": guided}.items():
            raw_score, shaped_reward = scores[name]
            error = np.abs(raw_score - score_center)
            reward_low, reward_high = bootstrap_mean(shaped_reward, cli.bootstrap, cli.seed + length + (0 if name == "unguided" else 1))
            error_low, error_high = bootstrap_mean(error, cli.bootstrap, cli.seed + length + 10 + (0 if name == "unguided" else 1))
            rows.append({
                "length": length, "set": name, "n_samples": cli.n_samples, "score_mean": float(raw_score.mean()),
                "target_score": score_center, "target_abs_error_mean": float(error.mean()),
                "target_abs_error_ci95_low": error_low, "target_abs_error_ci95_high": error_high,
                "analytic_reward_mean": float(shaped_reward.mean()), "analytic_reward_ci95_low": reward_low,
                "analytic_reward_ci95_high": reward_high, **diversity_metrics(tokens, test_sequences),
            })
        rows.append({
            "length": length, "set": "paired_guidance_effect", "n_samples": cli.n_samples,
            "paired_target_error_improvement_mean": float(paired_error_delta.mean()),
            "paired_target_error_improvement_ci95_low": delta_low,
            "paired_target_error_improvement_ci95_high": delta_high,
            "fraction_guided_target_improved": float((paired_error_delta > 0).mean()),
        })
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "reward_attainment_summary.csv", index=False)
        all_summary.extend(rows)
        np.save(out_dir / "unconditional_score_calibration.npy", calibration)
        with (out_dir / "metadata.json").open("w") as handle:
            json.dump({
                "teacher_checkpoint": teacher_paths[length], "dmfm_checkpoint": student_paths[length],
                "reward": cli.reward, "motif": cli.motif, "motif2": cli.motif2,
                "motif_tau": cli.motif_tau, "conjunction_tau": cli.conjunction_tau,
                "target_percentile": cli.target_percentile, "target_score": score_center,
                "score_std": score_std, "reward_evaluator": "exact analytic reward on decoded sequences",
                "posterior_sampler_for_guidance": "one_step_composed_dmfm_flow_map",
                "generation": {key: value for key, value in vars(cli).items() if key not in {"teacher_ckpt", "dmfm_ckpt"}},
            }, handle, indent=2, sort_keys=True)
        print(summary.to_string(index=False), flush=True)
        del teacher, student
        torch.cuda.empty_cache()
    pd.DataFrame(all_summary).to_csv(root / "reward_attainment_summary_all_lengths.csv", index=False)
    print(f"wrote {root}", flush=True)


if __name__ == "__main__":
    main()
