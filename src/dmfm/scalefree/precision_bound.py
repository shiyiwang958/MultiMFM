#!/usr/bin/env python
"""How much does producing a Table 14/15 cell on CPU instead of an H100 move it?

The CPU route (``scripts/dna/scalefree/fig5_dmfm_gradient_mc_cpu.sbatch``) computes the
finite-N dMFM gradients in fp32 on CPU, whereas the published run computed its finite-N
gradients in TF32 on an H100. ``check_published_protocol`` bounds that gap at 0.6-1.9 % on
individual gradient *norms*; this module answers the question that actually matters, on the
*metric*: recompute a subset of the published run's own finite-N GLASS estimates on CPU in
fp32, then score the Table 14/15 cells (pooled relative L1, Eq 46, and mean cosine, Eq 47)
over that subset twice - once from the published TF32 tensors, once from the CPU fp32
recomputation - against the same published GLASS-2048 reference, and report the difference.

The published run is the ideal control: its ``gradient_pairs.pt`` stores both the reference
and the TF32 finite-N estimates at the same conditioning states, so nothing has to be
assumed. The GLASS estimator integrates 8 Euler steps and differentiates through all of
them, while the dMFM estimator of the rerun is a single flow-map evaluation, so the number
this prints is an **upper bound** on the CPU-versus-GPU movement of a dMFM cell.

CPU only. Roughly 6 s per (state, repeat) estimate at N=8, L=50::

    python -m dmfm.scalefree.precision_bound \
        --published-root /n/holylabs/.../dirichlet-flow-matching/workdir \
        --reward motif --length 50 --mc 1 8 --n_states 8 \
        --out results/dna/scale_free_mc/dmfm/precision_bound.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dmfm.experiments.ablate_glass_gradient_mc import (
    estimate_value_gradient,
    make_probe_states,
    motif_tensor,
    seeded_randn,
)
from dmfm.scalefree.run_shard import PUBLISHED_SEED, _load_published_calibration
from dmfm.scalefree.verify_reference import DEFAULT_PUBLISHED_RUNS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--published-root", required=True)
    p.add_argument("--published-run", action="append", default=[], metavar="REWARD=DIRNAME")
    p.add_argument("--reward", choices=sorted(DEFAULT_PUBLISHED_RUNS), default="motif")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--mc", type=int, nargs="+", default=[1, 8], help="Which N columns to score.")
    p.add_argument("--n_states", type=int, default=8, help="First n of the 32 conditioning states.")
    p.add_argument("--n_repeats", type=int, default=8)
    p.add_argument("--n_probes", type=int, default=32)
    p.add_argument("--torch_threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=PUBLISHED_SEED)
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def cells(estimates: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Eq 46 (pooled relative L1) and Eq 47 (mean cosine) over [states, repeats, L, 4]."""
    estimates = estimates.double()
    reference = reference.double()
    ref = reference[:, None]
    abs_error = (estimates - ref).abs().sum(dim=(-1, -2))
    relative_l1 = abs_error.sum().item() / (reference.abs().sum(dim=(-1, -2)).sum().item() * estimates.shape[1])
    cosine = (estimates * ref).sum(dim=(-1, -2)) / (
        estimates.square().sum(dim=(-1, -2)).sqrt() * ref.square().sum(dim=(-1, -2)).sqrt()
    ).clamp_min(1e-30)
    return {"relative_l1": float(relative_l1), "cosine": float(cosine.mean(dim=1).mean())}


def main(argv=None) -> None:
    args = parse_args(argv)
    runs = dict(DEFAULT_PUBLISHED_RUNS)
    for spec in args.published_run:
        reward, name = spec.split("=", 1)
        runs[reward] = name
    published = Path(args.published_root) / runs[args.reward] / f"L{args.length}"
    pairs = torch.load(published / "gradient_pairs.pt", map_location="cpu", weights_only=False)
    mc_values = pairs["mc_values"].tolist()
    reference = pairs["reference_gradients"][: args.n_states]

    torch.set_num_threads(int(args.torch_threads))
    device = torch.device("cpu")
    length = int(args.length)

    from dmfm import api

    base, cfg = api.load_base(length, device)
    motif = motif_tensor("TTTTTC", device)
    motif2 = motif_tensor("AAAATT", device)
    calibration = _load_published_calibration(published.parent, length, args.reward)
    states = make_probe_states(
        base, cfg, n_probes=args.n_probes, t_eval=args.t_eval, nfe=32,
        seed=args.seed + 1_000_000 + length, device=device,
    )
    common = dict(
        score_center=calibration["score_center"], score_std=calibration["score_std"],
        z_target=calibration["z_target"], reward_beta=1.0, reward_scale=1.0, reward=args.reward,
        motif=motif, motif2=motif2, motif_tau=0.10, conjunction_tau=0.10, nfe_value=8,
        glass_end_time=1.0, glass_solver="euler", mc_chunk=16, reward_objective="target",
    )

    report: dict[str, object] = {
        "published_run": str(published), "reward": args.reward, "length": length,
        "n_states": args.n_states, "n_repeats": args.n_repeats,
        "what": "published TF32-GPU finite-N GLASS estimates vs a CPU fp32 recomputation of the "
                "same estimates, scored as Table 14/15 cells against the same GLASS-2048 reference. "
                "An upper bound for the dMFM rerun (8 differentiated network evals vs 1).",
        "per_mc": {},
    }
    worst_rel, worst_cos = 0.0, 0.0
    for mc in args.mc:
        if mc not in mc_values:
            raise ValueError(f"N={mc} is not in the published sweep {mc_values}.")
        mc_index = mc_values.index(mc)
        gpu = pairs["estimate_gradients"][: args.n_states, mc_index, : args.n_repeats]
        cpu = torch.empty_like(gpu)
        for state in range(args.n_states):
            for repeat in range(args.n_repeats):
                eps = seeded_randn(
                    (1, mc, length, 4),
                    args.seed + 20_000_000 + 1_000_000 * length + 10_000 * state + 100 * mc + repeat,
                    device,
                )
                _, gradient = estimate_value_gradient(
                    base, states[state : state + 1], t_eval=args.t_eval, eps_pool=eps, **common
                )
                cpu[state, repeat] = gradient[0]
            print(f"  N={mc} state {state + 1}/{args.n_states}", flush=True)
        gpu_cells, cpu_cells = cells(gpu, reference), cells(cpu, reference)
        norm_dev = (
            (cpu - gpu).flatten(2).norm(dim=-1) / gpu.flatten(2).norm(dim=-1).clamp_min(1e-30)
        )
        entry = {
            "gpu_tf32": gpu_cells,
            "cpu_fp32": cpu_cells,
            "delta_relative_l1": cpu_cells["relative_l1"] - gpu_cells["relative_l1"],
            "delta_cosine": cpu_cells["cosine"] - gpu_cells["cosine"],
            "gradient_norm_deviation": {
                "mean": float(norm_dev.mean()), "max": float(norm_dev.max()),
            },
        }
        report["per_mc"][str(mc)] = entry
        worst_rel = max(worst_rel, abs(entry["delta_relative_l1"]))
        worst_cos = max(worst_cos, abs(entry["delta_cosine"]))
        print(
            f"N={mc}: relative L1 {gpu_cells['relative_l1']:.4f} (GPU TF32) vs "
            f"{cpu_cells['relative_l1']:.4f} (CPU fp32), delta {entry['delta_relative_l1']:+.4f}; "
            f"cosine {gpu_cells['cosine']:.4f} vs {cpu_cells['cosine']:.4f}, "
            f"delta {entry['delta_cosine']:+.4f}; per-gradient norm deviation "
            f"mean {norm_dev.mean():.2%} max {norm_dev.max():.2%}",
            flush=True,
        )
    report["worst_abs_delta_relative_l1"] = worst_rel
    report["worst_abs_delta_cosine"] = worst_cos
    print(
        f"\nUPPER BOUND on CPU-vs-GPU movement of a cell: relative L1 {worst_rel:.4f}, "
        f"cosine {worst_cos:.4f} (Tables 14-15 print both to 2 decimals)."
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
