#!/usr/bin/env python
"""Value-gradient accuracy over the (steps, MC) grid at matched NFE -- Table 18's own question.

Table 18 prints the coordinate-wise MAE of the dMFM finite-MC value gradient against a
GLASS-128 (RK4, 50 steps) reference, with the dMFM posterior fixed at **4** composed
flow-map steps and MC in {1, 2, 4, 8}.  The value gradient costs ``steps x MC`` network
calls, so its published top point, 4 x 8, is one of four ways to spend NFE 32; the others
(1 x 32, 2 x 16, 8 x 4) were never measured.  This module measures all of them.

It reuses ``ablate_dmfm_one_step_gradient_mc``'s own functions -- the probe construction, the
reward calibration, the dMFM estimator and the GLASS reference estimator -- and its seeds, so
with the same ``--seed``, ``--n_probes`` and ``--t_eval`` the conditioning states and the
reference gradients are *identical* to the published Table 18 run's and the ``steps = 4`` rows
are directly comparable with it.  The reference does not depend on the dMFM sampler, so it is
computed once per probe and shared by the whole grid, which is what makes the grid affordable.

The candidate noise pools are nested (``--nested_mc_pools`` in the published run): the MC = m
pool is the first m columns of one draw, so the MC axis is a refinement of one sample and not
independent draws.  The same pool is reused across step counts, so a paired comparison between
two splits at fixed (probe, repeat) differs only in the posterior sampler.

Students: convention 1 of ``docs/REPRODUCIBILITY_PLAN.md`` section 7 item 8 selects the
diagonal student at one composed step and the 4-step ESD student above that.  Both are run at
one step so that the effect of the split is separated from the effect of the student swap.

    python -m dmfm.experiments.operating_point_gradient_grid --length 50 --reward gc \
        --output_dir results/dna/operating_point/table18_grid/gc
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

from dmfm import paths
from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import (
    estimate_value_gradient as estimate_dmfm_value_gradient,
    load_dmfm,
    make_heldout_probe_states,
)
from dmfm.experiments.ablate_glass_gradient_mc import (
    DEFAULT_CKPTS,
    calibrate_score,
    estimate_value_gradient as estimate_glass_value_gradient,
    load_model,
    motif_tensor,
    seeded_randn,
)
from dmfm.experiments.parity_dmfm_glass_c0 import run_env


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--reward", choices=["gc", "motif", "conjunction"], default="gc")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # Everything below reproduces scripts/dna/table18_dmfm4_gradient_mc.sbatch.
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--probe_data_dir", default=str(paths.DATA / "yeast_parent_disjoint"))
    p.add_argument("--probe_split", default="test")
    p.add_argument("--nfe_probe", type=int, default=32)
    p.add_argument("--nfe_calibration", type=int, default=32)
    p.add_argument("--calibration_samples", type=int, default=128)
    p.add_argument("--sample_batch_size", type=int, default=16)
    p.add_argument("--n_probes", type=int, default=32)
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--reference_mc", type=int, default=128)
    p.add_argument("--reference_glass_steps", type=int, default=50)
    p.add_argument("--reference_glass_solver", choices=["euler", "rk4"], default="rk4")
    p.add_argument("--glass_end_time", type=float, default=0.999)
    p.add_argument("--dmfm_end_time", type=float, default=0.999)
    p.add_argument("--mc_chunk", type=int, default=16)
    p.add_argument("--z_target", type=float, default=1.0)
    p.add_argument("--reward_beta", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--reward_objective", choices=["target", "maximize"], default="target")
    p.add_argument("--motif", default="TTTTTC")
    p.add_argument("--motif2", default="AAAATT")
    p.add_argument("--motif_tau", type=float, default=0.10)
    p.add_argument("--conjunction_tau", type=float, default=0.10)
    p.add_argument("--target_percentile", type=float, default=0.90)
    p.add_argument("--bootstrap", type=int, default=10000)
    # The grid itself.
    p.add_argument("--mc_values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--steps_values", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--also_diagonal_one_step", action="store_true", default=True,
                   help="Also run the diagonal student at one step (convention 1's artifact there).")
    p.add_argument("--glass_steps_values", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Give GLASS the same matched-NFE grid; [] to skip it.")
    return p.parse_args(argv)


def student_kinds(args) -> list[tuple[str, int, str]]:
    """``(student_key, steps, checkpoint)`` combinations to evaluate."""
    out: list[tuple[str, int, str]] = []
    for steps in args.steps_values:
        out.append(("dmfm4", int(steps), str(paths.dmfm4_ckpt(args.length))))
    if args.also_diagonal_one_step and 1 in [int(s) for s in args.steps_values]:
        out.append(("dmfm", 1, str(paths.dmfm_ckpt(args.length))))
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir) / f"L{args.length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    length = int(args.length)

    teacher, cfg = load_model(DEFAULT_CKPTS[length], device)
    motif = motif_tensor(args.motif, device)
    motif2 = motif_tensor(args.motif2, device)
    calibration = calibrate_score(
        teacher, cfg, n_samples=args.calibration_samples, batch_size=args.sample_batch_size,
        nfe=args.nfe_calibration, seed=args.seed + 100_000 * length, device=device,
        reward=args.reward, motif=motif, motif2=motif2, motif_tau=args.motif_tau,
        conjunction_tau=args.conjunction_tau,
    )
    score_mean, score_std = float(calibration.mean()), float(calibration.std(ddof=1))
    if args.reward_objective == "maximize":
        score_center, reward_z_target = score_mean, 0.0
    elif args.reward == "gc":
        score_center, reward_z_target = score_mean, float(args.z_target)
    else:
        score_center, reward_z_target = float(np.quantile(calibration, args.target_percentile)), 0.0

    states, source_indices = make_heldout_probe_states(length, cfg, args, device)
    combos = student_kinds(args)
    students = {key: load_dmfm(ckpt, device) for key, _, ckpt in combos}
    ckpt_of = {key: ckpt for key, _, ckpt in combos}

    reward_common = dict(
        score_center=score_center, score_std=score_std, z_target=reward_z_target,
        reward_beta=args.reward_beta, reward_scale=args.reward_scale, reward=args.reward,
        motif=motif, motif2=motif2, motif_tau=args.motif_tau,
        conjunction_tau=args.conjunction_tau, mc_chunk=args.mc_chunk,
        reward_objective=args.reward_objective,
    )
    rows: list[dict] = []
    t_start = time.time()
    for probe_idx in range(args.n_probes):
        x = states[probe_idx : probe_idx + 1]
        # Identical seed to ablate_dmfm_one_step_gradient_mc, so the reference gradient of the
        # published Table 18 run is reproduced bit-for-bit up to CUDA nondeterminism.
        reference_eps = seeded_randn(
            (1, args.reference_mc, length, 4),
            args.seed + 10_000_000 + 10_000 * length + probe_idx, device,
        )
        _, reference_grad = estimate_glass_value_gradient(
            teacher, x, t_eval=args.t_eval, eps_pool=reference_eps,
            nfe_value=args.reference_glass_steps, glass_end_time=args.glass_end_time,
            glass_solver=args.reference_glass_solver, **reward_common,
        )
        reference_l2 = reference_grad.flatten().norm().item()
        for repeat in range(args.n_repeats):
            nested_eps = seeded_randn(
                (1, max(args.mc_values), length, 4),
                args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe_idx + repeat, device,
            )
            for mc in args.mc_values:
                eps_pool = nested_eps[:, :mc]
                for key, steps, _ in combos:
                    _, grad = estimate_dmfm_value_gradient(
                        students[key], x, t_eval=args.t_eval, eps_pool=eps_pool,
                        dmfm_sampler="flow_map", dmfm_steps=int(steps),
                        dmfm_end_time=args.dmfm_end_time, **reward_common,
                    )
                    rows.append(_row(grad, reference_grad, reference_l2, length, probe_idx,
                                     repeat, mc, steps, f"{key}_fm{steps}", "dmfm", key))
                for gsteps in args.glass_steps_values:
                    _, grad = estimate_glass_value_gradient(
                        teacher, x, t_eval=args.t_eval, eps_pool=eps_pool,
                        nfe_value=int(gsteps), glass_end_time=args.glass_end_time,
                        glass_solver="euler", **reward_common,
                    )
                    rows.append(_row(grad, reference_grad, reference_l2, length, probe_idx,
                                     repeat, mc, gsteps, f"glass_euler{gsteps}", "glass", "base"))
        print(f"L={length} {args.reward}: probe {probe_idx + 1}/{args.n_probes} "
              f"({time.time() - t_start:.0f}s)", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out_dir / "gradient_grid.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "gradient_grid.csv", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "args": vars(args), "run_env": run_env(device),
        "teacher_ckpt": str(DEFAULT_CKPTS[length]),
        "students": {k: paths.rel(v) for k, v in ckpt_of.items()},
        "score_mean": score_mean, "score_std": score_std, "score_center": score_center,
        "reward_z_target": reward_z_target,
        "source_indices": torch.as_tensor(source_indices).tolist(),
        "_what": "Coordinate-wise MAE of the dMFM/GLASS finite-MC value gradient against the "
                 "GLASS-128 RK4-50 reference, over the (posterior steps, MC) grid. NFE = steps*mc.",
    }, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_dir / 'gradient_grid.csv'}", flush=True)


def _row(grad, reference_grad, reference_l2, length, probe, repeat, mc, steps, config, kind, student):
    diff = (grad - reference_grad)
    err_l2 = diff.flatten().norm().item()
    est_l2 = grad.flatten().norm().item()
    dot = (grad * reference_grad).sum().item()
    return {
        "length": length, "probe": probe, "repeat": repeat, "mc": int(mc),
        "steps": int(steps), "nfe": int(steps) * int(mc), "config": config,
        "kind": kind, "student": student,
        "mae": diff.abs().mean().item(),
        "e_rms": err_l2 / math.sqrt(4 * length), "error_l2": err_l2,
        "reference_l2": reference_l2, "estimate_l2": est_l2,
        "cosine_similarity": dot / max(est_l2 * reference_l2, 1e-12),
        "relative_l2_error": err_l2 / max(reference_l2, 1e-12),
    }


if __name__ == "__main__":
    main()
