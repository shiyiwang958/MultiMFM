#!/usr/bin/env python
"""Table 13's (steps, MC) operating-point grid: exact-reward attainment at matched NFE.

Table 13 guides with **one** direct dMFM map ``F(0, 1, eps | t, x_t)`` of the diagonal student
and MC = 8, i.e. a value gradient costing 8 network calls, under the stock clip-down gradient
rule.  ``dmfm.experiments.sample_reward_attainment`` hard-wires that single map: it calls
``one_step_dmfm`` and exposes no posterior-step knob, so the published run could not have spent
its NFE any other way.  This module adds the knob, so the split can actually be measured
instead of assumed:

* ``--steps_values`` composes that many learned flow-map jumps (``flow_map_dmfm``, the same
  function Table 18 and the C0 sweeps use); ``steps = 1, end_time = 1.0`` **is**
  ``one_step_dmfm``, so the 1 x 8 row reproduces the published configuration.
* ``--rules`` runs the published clip-down rule (``clip``) and/or the scale-free rule
  (``norm``, convention 2 of ``docs/REPRODUCIBILITY_PLAN.md`` section 7 item 8).  Under
  clip-down a change in MC moves both the gradient's noise *and*, whenever its norm sits below
  ``grad_clip``, the guided step size; under the scale-free rule only the noise moves.  Both
  are reported so the two effects are not confused.
* ``--student`` follows convention 1 by default: the diagonal student at one composed step
  (Table 13's own artifact) and the 4-step ESD student above that.

Everything else -- reward, calibration, trajectory, guidance window, coefficient, seeds, the
paired-improvement statistic and its bootstrap -- is ``sample_reward_attainment``'s, imported
rather than restated.  Within a batch every configuration starts from the *same* initial noise,
so the unguided arm is shared and only the guidance term differs.

    python -m dmfm.experiments.operating_point_reward_grid --length 50 --reward motif \
        --n_samples 100 --nfe 8 --out_dir results/dna/operating_point/table13_grid/motif
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dmfm import paths
from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import flow_map_dmfm, load_dmfm
from dmfm.experiments.ablate_glass_gradient_mc import (
    DEFAULT_CKPTS,
    calibrate_score,
    load_model,
    motif_tensor,
    reward_score,
    target_reward,
)
from dmfm.experiments.parity_dmfm_glass_c0 import run_env
from dmfm.experiments.sample_reward_attainment import bootstrap_mean, strings
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--reward", choices=["motif", "conjunction"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nfe", type=int, default=8, help="value-gradient NFE to hold fixed (steps x MC)")
    p.add_argument("--steps_values", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--rules", nargs="+", default=["clip", "norm"], choices=["clip", "norm"])
    p.add_argument("--student", default="convention", choices=["convention", "dmfm", "dmfm4"])
    # Published Table 13 settings (scripts/dna/table13_reward_attainment.sbatch).
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--nfe_traj", type=int, default=64)
    p.add_argument("--t_end", type=float, default=0.999)
    p.add_argument("--guide_t_start", type=float, default=0.50)
    p.add_argument("--guide_t_end", type=float, default=0.95)
    p.add_argument("--mc_chunk", type=int, default=8)
    p.add_argument("--guidance_frac", type=float, default=8.0)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--coeff_cap", type=float, default=10.0)
    p.add_argument("--calibration_samples", type=int, default=1024)
    p.add_argument("--nfe_calibration", type=int, default=64)
    p.add_argument("--calibration_batch_size", type=int, default=32)
    p.add_argument("--target_percentile", type=float, default=0.90)
    p.add_argument("--reward_beta", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--motif", default="TTTTTC")
    p.add_argument("--motif2", default="AAAATT")
    p.add_argument("--motif_tau", type=float, default=0.10)
    p.add_argument("--conjunction_tau", type=float, default=0.10)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args(argv)


def student_key(args, steps: int) -> str:
    if args.student != "convention":
        return args.student
    return paths.dmfm_student_kind(int(steps))


def value_gradient(student, x, t, eps_pool, *, steps: int, reward_kwargs, mc_chunk: int):
    """Finite-MC gradient of the dMFM value estimate with ``steps`` composed flow-map jumps."""
    batch, mc = eps_pool.shape[:2]
    chunk = max(1, min(int(mc_chunk), mc))

    def endpoints(x_in, start, end):
        width = end - start
        return flow_map_dmfm(
            student,
            eps_pool[:, start:end].reshape(batch * width, *x_in.shape[1:]),
            x_in.repeat_interleave(width, dim=0),
            t.repeat_interleave(width),
            n_steps=int(steps), end_time=1.0,
        ), width

    with torch.no_grad():
        logs = []
        for start in range(0, mc, chunk):
            end = min(start + chunk, mc)
            ep, width = endpoints(x.detach(), start, end)
            logs.append((reward_kwargs["reward_scale"] * target_reward(ep, **reward_kwargs["tr"])).view(batch, width))
        weights = torch.softmax(torch.cat(logs, dim=1), dim=1)

    x_leaf = x.detach().clone().requires_grad_(True)
    gradient = torch.zeros_like(x_leaf)
    for start in range(0, mc, chunk):
        end = min(start + chunk, mc)
        ep, width = endpoints(x_leaf, start, end)
        rv = target_reward(ep, **reward_kwargs["tr"]).view(batch, width)
        obj = (weights[:, start:end] * reward_kwargs["reward_scale"] * rv).sum()
        gradient = gradient + torch.autograd.grad(obj, x_leaf)[0]
    return gradient.detach()


def generate(teacher, students, cfg, *, args, steps: int, mc: int, rule: str, reward_kwargs,
             length: int, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    student = students[student_key(args, steps)]
    unguided_parts, guided_parts = [], []
    diag = {"raw_mean": [], "frac_above_clip": [], "post_mean": []}
    for start in range(0, args.n_samples, args.batch_size):
        count = min(args.batch_size, args.n_samples - start)
        # Same generator, same order of draws as sample_reward_attainment: the initial noise is
        # drawn first, so it is identical for every configuration; the MC pool follows.
        generator = torch.Generator(device=device).manual_seed(args.seed + 100_000 * length + start)
        x_unguided = torch.randn((count, length, 4), generator=generator, device=device)
        x_guided = x_unguided.clone()
        eps_pool = torch.randn((count, mc, length, 4), generator=generator, device=device)
        grid = torch.linspace(0.0, args.t_end, args.nfe_traj + 1, device=device)
        for s0, s1 in zip(grid[:-1], grid[1:]):
            s = s0.expand(count)
            with torch.no_grad():
                x_unguided, _, _ = gaussian_denoiser_flow_step(cfg, teacher, x_unguided, s, s1.expand(count))
                x_base, _, _ = gaussian_denoiser_flow_step(cfg, teacher, x_guided, s, s1.expand(count))
            if args.guide_t_start <= float(s0) <= args.guide_t_end:
                g = value_gradient(student, x_guided, s, eps_pool, steps=steps,
                                   reward_kwargs=reward_kwargs, mc_chunk=args.mc_chunk)
                n = g.flatten(1).norm(dim=1).clamp_min(1e-8)
                diag["raw_mean"].append(float(n.mean()))
                diag["frac_above_clip"].append(float((n > args.grad_clip).float().mean()))
                if rule == "norm":
                    g = g * (args.grad_clip / n)[:, None, None]
                else:
                    g = g * (args.grad_clip / n).clamp(max=1.0)[:, None, None]
                diag["post_mean"].append(float(g.flatten(1).norm(dim=1).mean()))
                coefficient = (args.guidance_frac * teacher.sde_sigma_sq(s)).clamp(max=args.coeff_cap)
                x_guided = x_base + float(s1 - s0) * coefficient[:, None, None] * g
            else:
                x_guided = x_base
        unguided_parts.append(x_unguided.argmax(dim=-1).cpu())
        guided_parts.append(x_guided.argmax(dim=-1).cpu())
    return torch.cat(unguided_parts), torch.cat(guided_parts), {k: float(np.mean(v)) for k, v in diag.items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    length = int(args.length)
    out = Path(args.out_dir) / f"L{length}"
    out.mkdir(parents=True, exist_ok=True)

    teacher, cfg = load_model(DEFAULT_CKPTS[length], device)
    motif = motif_tensor(args.motif, device)
    motif2 = motif_tensor(args.motif2, device)
    calibration = calibrate_score(
        teacher, cfg, n_samples=args.calibration_samples, batch_size=args.calibration_batch_size,
        nfe=args.nfe_calibration, seed=args.seed + 10_000 * length, device=device,
        reward=args.reward, motif=motif, motif2=motif2, motif_tau=args.motif_tau,
        conjunction_tau=args.conjunction_tau,
    )
    score_center = float(np.quantile(calibration, args.target_percentile))
    score_std = float(calibration.std(ddof=1))
    print(f"L={length} {args.reward}: target={score_center:.6f} scale={score_std:.6f}", flush=True)

    tr = dict(score_center=score_center, score_std=score_std, z_target=0.0, beta=args.reward_beta,
              reward=args.reward, motif=motif, motif2=motif2, motif_tau=args.motif_tau,
              conjunction_tau=args.conjunction_tau)
    reward_kwargs = {"tr": tr, "reward_scale": args.reward_scale}

    combos = []
    for steps in args.steps_values:
        mc = args.nfe // int(steps)
        if mc < 1:
            continue
        for rule in args.rules:
            combos.append((int(steps), int(mc), rule))
    students = {}
    for steps, _, _ in combos:
        key = student_key(args, steps)
        if key not in students:
            ckpt = paths.dmfm_ckpt(length) if key == "dmfm" else paths.dmfm4_ckpt(length)
            students[key] = load_dmfm(str(ckpt), device)

    sample_rows, summary_rows = [], []
    for steps, mc, rule in combos:
        t0 = time.time()
        unguided, guided = None, None
        unguided, guided, diag = generate(teacher, students, cfg, args=args, steps=steps, mc=mc,
                                          rule=rule, reward_kwargs=reward_kwargs, length=length,
                                          device=device)
        with torch.no_grad():
            su = reward_score(F.one_hot(unguided.to(device), 4).float(), reward=args.reward,
                              motif=motif, motif2=motif2, motif_tau=args.motif_tau,
                              conjunction_tau=args.conjunction_tau).cpu().numpy()
            sg = reward_score(F.one_hot(guided.to(device), 4).float(), reward=args.reward,
                              motif=motif, motif2=motif2, motif_tau=args.motif_tau,
                              conjunction_tau=args.conjunction_tau).cpu().numpy()
        label = f"{student_key(args, steps)}_fm{steps}@{rule}@mc={mc}"
        delta = np.abs(su - score_center) - np.abs(sg - score_center)
        lo, hi = bootstrap_mean(delta, args.bootstrap, args.seed + length)
        elo, ehi = bootstrap_mean(np.abs(sg - score_center), args.bootstrap, args.seed + length + 11)
        for i in range(len(su)):
            sample_rows.append({"config": label, "length": length, "sample_id": i,
                                "steps": steps, "mc": mc, "nfe_value": steps * mc, "rule": rule,
                                "student": student_key(args, steps),
                                "score_unguided": float(su[i]), "score_guided": float(sg[i]),
                                "target_score": score_center,
                                "seq_unguided": strings(unguided[i : i + 1])[0],
                                "seq_guided": strings(guided[i : i + 1])[0]})
        summary_rows.append({
            "config": label, "length": length, "steps": steps, "mc": mc, "nfe_value": steps * mc,
            "rule": rule, "student": student_key(args, steps), "n": len(su),
            "target_score": score_center,
            "unguided_target_abs_error": float(np.abs(su - score_center).mean()),
            "guided_target_abs_error": float(np.abs(sg - score_center).mean()),
            "guided_target_abs_error_ci_lo": elo, "guided_target_abs_error_ci_hi": ehi,
            "paired_improvement": float(delta.mean()),
            "paired_improvement_ci_lo": lo, "paired_improvement_ci_hi": hi,
            "frac_improved": float((delta > 0).mean()),
            "exact_unique_frac": float(torch.unique(guided, dim=0).shape[0] / len(su)),
            **diag, "seconds": time.time() - t0,
        })
        print(f"  {label:34s} guided|err|={np.abs(sg - score_center).mean():.4f} "
              f"paired={delta.mean():+.4f} |g|raw={diag['raw_mean']:.3g} "
              f"({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(sample_rows).to_csv(out / "sample_scores.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out / "run_metadata.json").write_text(json.dumps({
        "args": vars(args), "run_env": run_env(device), "score_center": score_center,
        "score_std": score_std, "teacher_ckpt": paths.rel(DEFAULT_CKPTS[length]),
        "students": {k: paths.rel(paths.dmfm_ckpt(length) if k == "dmfm" else paths.dmfm4_ckpt(length))
                     for k in students},
        "_what": "Table 13's exact-reward attainment over the (posterior steps, MC) grid at "
                 "fixed value-gradient NFE, under both the published clip-down rule and the "
                 "scale-free rule. steps=1 @clip @mc=8 is the published configuration.",
    }, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
