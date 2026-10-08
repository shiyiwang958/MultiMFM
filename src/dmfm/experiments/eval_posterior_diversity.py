#!/usr/bin/env python
"""Reward-free posterior look-ahead diversity evaluation across DNA lengths.

For fixed held-out forward-process observations x_t, compare multiple dMFM
diagonal-Euler futures with matched GLASS futures.  Diversity is intentionally
computed within each (source, t) posterior condition rather than after pooling
different conditions.

Ported from dirichlet-flow-matching/scripts/eval_parent_posterior_diversity.py (Table 17).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmfm.utils.model_loading import load_args_json, load_data_seqs, load_student, load_teacher
from dmfm import paths
from dmfm.experiments.sample_c0_guidance import glass_integrate_diff
from dmfm.utils.flow_utils import gaussian_beta


# Frozen dMFM selection of the paper (original run dirs, in order:
# workdir/yeast_parent_a_L{50,100,200}_dmfm_diagonal_h192_b4 and
# workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726); the same files
# now live under checkpoints/dna/dmfm/L*/ with their args.json.
DEFAULT_STUDENTS = {L: str(paths.dmfm_ckpt(L)) for L in paths.LENGTHS}
DNA = np.asarray(list("ACGT"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200, 400])
    p.add_argument("--times", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    p.add_argument("--n_sources", type=int, default=32)
    p.add_argument("--n_futures", type=int, default=16)
    p.add_argument("--source_batch", type=int, default=4)
    p.add_argument("--dmfm_steps", type=int, default=1, help="Number of dMFM posterior flow-map steps.")
    p.add_argument(
        "--dmfm_sampler",
        choices=["flow_map", "diagonal_euler"],
        default="flow_map",
        help="Compose two-time dMFM maps (default) or integrate the diagonal velocity.",
    )
    p.add_argument("--glass_steps", type=int, default=100, help="Posterior Euler steps for the GLASS numerical reference.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--out_dir", default=str(paths.OUTPUTS / "posterior_diversity"))
    return p.parse_args(argv)


def randn_cpu(shape, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator)


@torch.inference_mode()
def dmfm_diagonal_euler(student, eps: torch.Tensor, x_cond: torch.Tensor, t_cond: torch.Tensor, n_steps: int) -> torch.Tensor:
    x = eps
    grid = torch.linspace(0, 1, int(n_steps) + 1, device=x.device, dtype=x.dtype)
    for r0, r1 in zip(grid[:-1], grid[1:]):
        b = x.shape[0]
        x = x + (r1 - r0) * student.v(r0.expand(b), r0.expand(b), x, t_cond, x_cond)
    return x


@torch.inference_mode()
def dmfm_flow_map(student, eps: torch.Tensor, x_cond: torch.Tensor, t_cond: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Compose the learned F(r_k, r_{k+1}, . | t_cond, x_cond) on a uniform grid."""
    x = eps
    grid = torch.linspace(0, 1, int(n_steps) + 1, device=x.device, dtype=x.dtype)
    for r0, r1 in zip(grid[:-1], grid[1:]):
        b = x.shape[0]
        x = student(r0.expand(b), r1.expand(b), x, t_cond, x_cond)
    return x


def seq_strings(tokens: torch.Tensor) -> list[str]:
    arr = tokens.detach().cpu().numpy()
    return ["".join(DNA[row].tolist()) for row in arr]


def condition_metrics(tokens: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    """Metrics for one fixed posterior condition, tokens [K,L]."""
    k, length = tokens.shape
    mismatch = (tokens[:, None, :] != tokens[None, :, :]).float().mean(dim=-1)
    tri = torch.triu_indices(k, k, offset=1, device=tokens.device)
    pair = mismatch[tri[0], tri[1]]
    nearest = mismatch.masked_fill(torch.eye(k, dtype=torch.bool, device=tokens.device), float("inf")).min(dim=1).values
    source_dist = (tokens != source[None, :]).float().mean(dim=-1)
    return {
        "exact_unique_frac": float(torch.unique(tokens, dim=0).shape[0] / k),
        "pairwise_hamming_frac": float(pair.mean()),
        "nearest_generated_frac": float(nearest.mean()),
        "source_hamming_frac": float(source_dist.mean()),
        "length": int(length),
    }


def bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) <= 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        means[b] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_conditions(frame: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = ["exact_unique_frac", "pairwise_hamming_frac", "nearest_generated_frac", "source_hamming_frac"]
    rows = []
    for (length, t, method), group in frame.groupby(["length", "t", "method"], sort=True):
        row = {"length": int(length), "t": float(t), "method": method, "n_conditions": int(len(group))}
        for j, metric in enumerate(metrics):
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values, n_bootstrap=bootstrap, seed=seed + int(length) * 1000 + int(round(float(t) * 100)) * 10 + j)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["pairwise_hamming_normalized"] = np.nan
    for (length, method), group in summary.groupby(["length", "method"], sort=False):
        base = group.loc[np.isclose(group["t"], 0.0), "pairwise_hamming_frac_mean"]
        if len(base):
            mask = (summary["length"] == length) & (summary["method"] == method)
            summary.loc[mask, "pairwise_hamming_normalized"] = summary.loc[mask, "pairwise_hamming_frac_mean"] / float(base.iloc[0])
    return summary.sort_values(["length", "method", "t"], ignore_index=True)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    panels = [
        ("pairwise_hamming_frac_mean", "conditional pairwise Hamming fraction"),
        ("pairwise_hamming_normalized", r"pairwise diversity / diversity at $t=0$"),
        ("source_hamming_frac_mean", "Hamming fraction to held-out source"),
    ]
    colors = {50: "#0072B2", 100: "#D55E00", 200: "#009E73", 400: "#CC79A7"}
    styles = {"dmfm": "-", "glass": "--"}
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.2), sharex=True)
    for axis, (metric, ylabel) in zip(axes, panels):
        for (length, method), group in summary.groupby(["length", "method"], sort=True):
            group = group.sort_values("t")
            axis.plot(group["t"], group[metric], color=colors[int(length)], linestyle=styles[method], marker="o", label=f"L={length} {method}")
        axis.set_xlabel("conditioning time t")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_diversity_vs_time.png", dpi=240)
    plt.close(fig)


def main(argv=None) -> None:
    cli = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_rows: list[dict] = []
    sample_rows: list[dict] = []

    for length in cli.lengths:
        if length not in DEFAULT_STUDENTS:
            raise ValueError(f"No frozen student selection for L={length}")
        student_ckpt = Path(DEFAULT_STUDENTS[length])
        student_dir = student_ckpt.parent
        cfg, teacher, _, teacher_incompat = load_teacher(
            paths.base_ckpt(length), paths.base_args(length), alphabet_size=4, device=device
        )
        if teacher_incompat.missing_keys or teacher_incompat.unexpected_keys:
            raise RuntimeError(f"Teacher incompatibility at L={length}: {teacher_incompat}")
        student, student_incompat = load_student(load_args_json(student_dir / "args.json"), alphabet_size=4, device=device, student_ckpt=student_ckpt)
        if student_incompat.missing_keys or student_incompat.unexpected_keys:
            raise RuntimeError(f"Student incompatibility at L={length}: {student_incompat}")
        for model in (teacher, student):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        # Port fix: the original path data/yeast_parent_disjoint/... is data/dna/... here.
        all_test = load_data_seqs(
            paths.data_pt(length),
            split_pt=paths.split_pt(length),
            split="test",
        )
        if cli.n_sources > len(all_test):
            raise ValueError(f"Requested {cli.n_sources} sources but only {len(all_test)} test sequences at L={length}")
        pick = torch.randperm(len(all_test), generator=torch.Generator().manual_seed(cli.seed + length))[: cli.n_sources]
        sources_cpu = all_test[pick].contiguous()
        source_strings = seq_strings(sources_cpu)
        print(f"L={length}: sources={len(sources_cpu)} student={student_ckpt.name}", flush=True)

        for t_index, t_value in enumerate(cli.times):
            cond_noise = randn_cpu((cli.n_sources, length, 4), cli.seed + length * 10000 + t_index * 100)
            terminal_noise = randn_cpu((cli.n_sources, cli.n_futures, length, 4), cli.seed + length * 10000 + t_index * 100 + 1)
            t_tensor_cpu = torch.full((cli.n_sources,), float(t_value))
            x1_cpu = F.one_hot(sources_cpu, num_classes=4).float()
            beta = gaussian_beta(cfg, t_tensor_cpu).reshape(-1, 1, 1).cpu()
            xcond_cpu = beta * x1_cpu + (1.0 - beta) * cond_noise

            for method in ("dmfm", "glass"):
                decoded_batches = []
                for start in range(0, cli.n_sources, cli.source_batch):
                    end = min(start + cli.source_batch, cli.n_sources)
                    b = end - start
                    x_cond = xcond_cpu[start:end].to(device)
                    eps = terminal_noise[start:end].to(device)
                    if float(t_value) == 1.0:
                        decoded = sources_cpu[start:end, None, :].expand(b, cli.n_futures, length).clone()
                    else:
                        eps_flat = eps.reshape(b * cli.n_futures, length, 4)
                        x_rep = x_cond.repeat_interleave(cli.n_futures, dim=0)
                        t_rep = torch.full((b * cli.n_futures,), float(t_value), device=device)
                        if method == "dmfm":
                            stepper = dmfm_flow_map if cli.dmfm_sampler == "flow_map" else dmfm_diagonal_euler
                            endpoint = stepper(student, eps_flat, x_rep, t_rep, cli.dmfm_steps)
                        else:
                            endpoint = glass_integrate_diff(teacher, eps_flat, x_rep, t_rep, n_steps=cli.glass_steps)
                        decoded = endpoint.argmax(dim=-1).reshape(b, cli.n_futures, length).cpu()
                    decoded_batches.append(decoded)
                decoded_all = torch.cat(decoded_batches, dim=0)
                for source_local in range(cli.n_sources):
                    metrics = condition_metrics(decoded_all[source_local], sources_cpu[source_local])
                    condition_rows.append({"length": length, "t": float(t_value), "method": method, "source_id": int(pick[source_local]), **metrics})
                    strings = seq_strings(decoded_all[source_local])
                    for future_id, sequence in enumerate(strings):
                        sample_rows.append({
                            "length": length, "t": float(t_value), "method": method,
                            "source_id": int(pick[source_local]), "future_id": future_id,
                            "source_sequence": source_strings[source_local], "sequence": sequence,
                        })
                print(f"  t={t_value:g} method={method} complete", flush=True)
        del teacher, student
        torch.cuda.empty_cache()

    conditions = pd.DataFrame(condition_rows)
    samples = pd.DataFrame(sample_rows)
    summary = summarize_conditions(conditions, cli.bootstrap, cli.seed)
    conditions.to_csv(out_dir / "posterior_condition_metrics.csv", index=False)
    samples.to_csv(out_dir / "posterior_futures.csv", index=False)
    summary.to_csv(out_dir / "posterior_diversity_summary.csv", index=False)
    plot_summary(summary, out_dir)
    with open(out_dir / "run_config.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["length", "student_dir", "student_ckpt", "n_sources", "n_futures", "dmfm_sampler", "dmfm_steps", "glass_steps", "times", "seed"])
        writer.writeheader()
        for length in cli.lengths:
            writer.writerow({"length": length, "student_dir": paths.rel(Path(DEFAULT_STUDENTS[length]).parent), "student_ckpt": Path(DEFAULT_STUDENTS[length]).name, "n_sources": cli.n_sources, "n_futures": cli.n_futures, "dmfm_sampler": cli.dmfm_sampler, "dmfm_steps": cli.dmfm_steps, "glass_steps": cli.glass_steps, "times": " ".join(map(str, cli.times)), "seed": cli.seed})
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
