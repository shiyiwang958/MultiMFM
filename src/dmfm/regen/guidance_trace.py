#!/usr/bin/env python
"""Fig 8: per-step guidance diagnostics for one guided and one unguided DNA sample.

**New implementation.** The producer of ``figures/steering.png`` is unknown (it was
exported on a Mac and no script in ``dirichlet-flow-matching``, its reconstruction or the
backup writes those axis labels), so this module reproduces the three panels the published
figure shows, along the sampling trajectory ``t in [0, t_max]``:

1. ``||grad_x V_t(x_t)||`` (log scale) - the value gradient norm *before* clipping,
2. ``V_t(x_t)`` - the GLASS value (``log mean_j exp(logr(x_1^j))``),
3. "predicted endpoint hard mean" - the guide's C0 score of ``argmax`` of the base model's
   one-step denoiser prediction ``E[x_1 | x_t]``, with the target as a dotted line.

Both a guided and an unguided trajectory are run from the same initial noise and the same
frozen MC noise pool, exactly as ``dmfm.experiments.sample_c0_guidance`` pairs them; on the
unguided arm the value and its gradient are computed as diagnostics but never applied.
Dashed vertical lines in the published panels are the guidance window
``[guide_t_start, guide_t_end]``, which is written to the metadata.

``--posterior`` selects the sampler inside the value gradient: ``glass`` (default, the
published regeneration) or ``dmfm`` (the 4-step ESD student at ``--nfe_value`` composed
flow-map steps), and ``--grad_normalize`` switches to the scale-free guidance rule. Those two
are the conventions REPRODUCIBILITY_PLAN.md S7 item 8 mandates for the dMFM demo reruns; the
defaults leave the published behaviour unchanged.

Usage::

    python -m dmfm.regen.guidance_trace --sample_ids 0 --target_c0 0.3 \
        --guide_t_start 0.0 --guide_t_end 0.95 --mc 128 --nfe_value 12 --nfe_traj 128

    python -m dmfm.regen.guidance_trace --posterior dmfm --grad_normalize \
        --sample_ids 0 1 2 3 --target_c0 0.3 --mc 8 --nfe_value 4 --nfe_traj 64 \
        --guide_t_start 0.5 --guide_t_end 0.95 --guidance_frac 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from dmfm import api, paths
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _hard_endpoint_score(guide, denoiser: torch.Tensor) -> torch.Tensor:
    """Guide C0 of ``argmax`` of the model's predicted endpoint (the Fig 8 panel-3 quantity)."""
    return guide(torch.nn.functional.one_hot(denoiser.argmax(-1), denoiser.shape[-1]).float())


def trace_guided_vs_unguided(
    base,
    cfg,
    guide,
    sample_ids,
    *,
    target_c0: float = 0.30,
    seed: int = 0,
    mc: int = 128,
    mc_chunk: int = 128,
    nfe_traj: int = 128,
    nfe_value: int = 12,
    t_max: float = 0.95,
    guide_t_start: float = 0.0,
    guide_t_end: float = 0.95,
    guidance_frac: float = 1.0,
    coeff_cap: float | None = 10.0,
    grad_clip: float | None = 10.0,
    reward_sigma: float = 0.15,
    reward_scale: float = 0.5,
    posterior: str = "glass",
    student=None,
    grad_normalize: bool = False,
    device=None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Run the paired sampler and record per-step diagnostics for both arms.

    ``posterior`` selects the sampler *inside* the value gradient, exactly as
    ``dmfm.experiments.sample_c0_guidance_dmfm --posterior`` does:

    * ``"glass"`` (default, unchanged): ``nfe_value`` Euler steps of the GLASS posterior
      velocity on the base DFM -- what the published Fig 8 regeneration ran.
    * ``"dmfm"``: ``nfe_value`` composed flow-map steps of the dMFM ``student``, which must
      be the artifact distilled for that jump (``paths.dmfm_ckpt_for_steps(L, nfe_value)``).

    ``grad_normalize`` switches the guidance step from the stock clip-down rule to the
    scale-free rule (rescale the value gradient to exactly ``grad_clip``), which
    REPRODUCIBILITY_PLAN.md S7 item 8 mandates for every dMFM demo; ``False`` keeps the
    published behaviour byte-for-byte.

    Returns ``(rows, endpoints)``: one row per (step, arm) and the final hard sequences.
    """
    sample_ids = list(sample_ids)
    device = torch.device(device) if device is not None else next(base.parameters()).device
    length = int(cfg.seq_len)
    batch = len(sample_ids)

    x0 = api.paired_initial_noise(sample_ids, length, seed=seed, device=device)
    eps_pool = api.paired_eps_pool(sample_ids, length, mc=mc, seed=seed, device=device)

    logr = api.c0_log_reward_fn(guide, target_c0, reward_sigma=reward_sigma, reward_scale=reward_scale)
    if posterior == "glass":
        post = api.glass_posterior_fn(base, n_steps=nfe_value)
    elif posterior == "dmfm":
        if student is None:
            raise ValueError("posterior='dmfm' needs a loaded dMFM student")
        post = api.dmfm_posterior_fn(student, sampler="flow_map", n_steps=nfe_value, end_time=1.0)
    else:
        raise ValueError(f"unknown posterior {posterior!r}")

    x = {"unguided": x0.clone(), "guided": x0.clone()}
    grid = torch.linspace(0, float(t_max), int(nfe_traj) + 1, device=device)
    rows: list[dict] = []
    t0 = time.time()

    for step, (s0, s1) in enumerate(zip(grid[:-1], grid[1:])):
        t_now = float(s0)
        s = s0.expand(batch)
        dt = float(s1 - s0)
        in_window = guide_t_start <= t_now <= guide_t_end

        step_next = {}
        for arm in ("unguided", "guided"):
            xa = x[arm]
            with torch.no_grad():
                x_next, _, denoiser = gaussian_denoiser_flow_step(cfg, base, xa, s, s1.expand(batch))
                endpoint_score = _hard_endpoint_score(guide, denoiser)
            v, g = api.value_and_grad(post, logr, xa, t_now, eps_pool, mc_chunk=mc_chunk)
            gnorm = g.flatten(1).norm(dim=1)

            coeff = guidance_frac * base.sde_sigma_sq(s)
            if coeff_cap is not None:
                coeff = coeff.clamp(max=coeff_cap)
            if arm == "guided" and in_window:
                g_applied = g
                if grad_clip is not None:
                    # Same two rules as api.guided_sample: clip-down (published) or
                    # rescale-to-grad_clip (scale-free, mandated for the dMFM demos).
                    scale = grad_clip / gnorm.clamp_min(1e-8)
                    g_applied = g * (scale if grad_normalize else scale.clamp(max=1.0))[:, None, None]
                elif grad_normalize:
                    raise ValueError("grad_normalize=True needs a grad_clip to normalise to")
                step_next[arm] = x_next + dt * coeff[:, None, None] * g_applied
            else:
                step_next[arm] = x_next

            for j, sid in enumerate(sample_ids):
                rows.append({
                    "sample_idx": int(sid), "arm": arm, "step": step, "t": t_now,
                    "grad_norm": float(gnorm[j]), "value": float(v[j]),
                    "grad_norm_applied": (
                        float(g_applied.flatten(1).norm(dim=1)[j])
                        if (arm == "guided" and in_window) else 0.0),
                    "endpoint_hard_score": float(endpoint_score[j]),
                    "guidance_coeff": float(coeff[j]) if (arm == "guided" and in_window) else 0.0,
                    "guidance_applied": bool(arm == "guided" and in_window),
                })
        x = step_next
        if verbose and (step % 16 == 0 or step == nfe_traj - 1):
            print(f"  step {step + 1}/{nfe_traj} t={t_now:.3f} ({time.time() - t0:.1f}s)", flush=True)

    endpoints = {}
    with torch.no_grad():
        for arm in ("unguided", "guided"):
            endpoints[arm] = {
                "sequences": api.tokens_to_strings(x[arm].argmax(-1)),
                "final_score": [float(v) for v in _hard_endpoint_score(guide, x[arm])],
            }
    return rows, endpoints


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig 8 guided-vs-unguided trajectory diagnostics (regenerated).")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--sample_ids", type=int, nargs="+", default=[0])
    p.add_argument("--target_c0", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc", type=int, default=128)
    p.add_argument("--mc_chunk", type=int, default=128)
    p.add_argument("--nfe_traj", type=int, default=128)
    p.add_argument("--nfe_value", type=int, default=12)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--guide_t_start", type=float, default=0.0)
    p.add_argument("--guide_t_end", type=float, default=0.95)
    p.add_argument("--guidance_frac", type=float, default=1.0)
    p.add_argument("--coeff_cap", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--guide_ckpt", default=None)
    p.add_argument(
        "--posterior",
        choices=["glass", "dmfm"],
        default="glass",
        help="Posterior sampler inside the value gradient. 'glass' (default) is the published "
        "Fig 8 regeneration; 'dmfm' composes --nfe_value flow-map steps of the dMFM student "
        "distilled for that jump (REPRODUCIBILITY_PLAN.md S7 item 8, convention 1).",
    )
    p.add_argument("--student_ckpt", default=None,
                   help="--posterior dmfm: default = paths.dmfm_ckpt_for_steps(--length, --nfe_value).")
    p.add_argument("--allow_gap_mismatch", action="store_true",
                   help="Permit a student that was not distilled for a 1/--nfe_value jump.")
    p.add_argument("--grad_normalize", action="store_true",
                   help="Scale-free guidance rule: rescale the value gradient to exactly "
                   "--grad_clip instead of only clipping it down (S7 item 8, convention 2).")
    p.add_argument("--out_dir", default=None, help="default results/dna/regen/fig8")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    import pandas as pd

    args = parse_args(argv)
    out_dir = Path(args.out_dir or (paths.RESULTS / "regen" / "fig8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = out_dir / "trace.csv"
    if args.skip_existing and trace_csv.exists():
        print(f"{trace_csv} exists, skipping")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base, cfg = api.load_base(args.length, device, ckpt=args.ckpt)
    guide = api.load_c0("guide", device, ckpt=args.guide_ckpt)

    student = None
    student_ckpt = None
    if args.posterior == "dmfm":
        expected = paths.dmfm_ckpt_for_steps(args.length, args.nfe_value)
        student_ckpt = Path(args.student_ckpt) if args.student_ckpt else expected
        if not args.allow_gap_mismatch and student_ckpt.resolve() != expected.resolve():
            raise SystemExit(
                f"--nfe_value {args.nfe_value} composes jumps of {1.0 / args.nfe_value:.3g}, which the "
                f"student at {paths.rel(student_ckpt)} was not distilled for; the matching artifact is "
                f"{paths.rel(expected)}. Pass --allow_gap_mismatch to override. "
                "See docs/dmfm_glass_parity.md."
            )
        student = api.load_dmfm(args.length, device, ckpt=student_ckpt)

    rows, endpoints = trace_guided_vs_unguided(
        base, cfg, guide, args.sample_ids, target_c0=args.target_c0, seed=args.seed, mc=args.mc,
        mc_chunk=args.mc_chunk, nfe_traj=args.nfe_traj, nfe_value=args.nfe_value, t_max=args.t_max,
        guide_t_start=args.guide_t_start, guide_t_end=args.guide_t_end, guidance_frac=args.guidance_frac,
        coeff_cap=args.coeff_cap, grad_clip=args.grad_clip, reward_sigma=args.reward_sigma,
        reward_scale=args.reward_scale, posterior=args.posterior, student=student,
        grad_normalize=bool(args.grad_normalize), device=device,
    )
    pd.DataFrame(rows).to_csv(trace_csv, index=False)

    meta = {
        "item": "Fig 8 (regenerated)",
        "status": "new implementation (producer unknown)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_rev": _git_rev(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm": {k: os.environ.get(v) for k, v in {
            "partition": "SLURM_JOB_PARTITION", "account": "SLURM_JOB_ACCOUNT",
            "nodelist": "SLURM_JOB_NODELIST", "cpus": "SLURM_CPUS_PER_TASK"}.items()},
        "device": str(device),
        "device_name": __import__("dmfm.regen.base_marginals", fromlist=["device_name"]).device_name(device),
        "torch_version": torch.__version__,
        "device_note": (
            "The initial noise and the MC noise pool are drawn by seeded CPU generators and only "
            "then moved to the device, so a CPU and a GPU run of this trace share identical "
            "randomness. They differ in float arithmetic, and on GPU also in the nondeterministic "
            "backward used for the value gradient; a CPU run is deterministic. Fig 8 illustrates "
            "one trajectory and is not the published trajectory in any case (its producer is "
            "lost), so no published number depends on which device drew it."),
        "settings": {k: getattr(args, k) for k in (
            "length", "sample_ids", "target_c0", "seed", "mc", "nfe_traj", "nfe_value", "t_max",
            "guide_t_start", "guide_t_end", "guidance_frac", "coeff_cap", "grad_clip",
            "reward_sigma", "reward_scale", "posterior", "grad_normalize")},
        "student_ckpt": paths.rel(student_ckpt) if student_ckpt is not None else None,
        "base_ckpt": paths.rel(args.ckpt or paths.base_ckpt(args.length)),
        "guide_ckpt": paths.rel(args.guide_ckpt or paths.c0_guide()),
        "endpoints": endpoints,
        "command": "python -m dmfm.regen.guidance_trace " + " ".join(
            [f"--{k} {getattr(args, k)}"
             for k in ("posterior", "target_c0", "mc", "nfe_traj", "nfe_value", "guidance_frac")]
            + (["--grad_normalize"] if args.grad_normalize else [])),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {trace_csv}", flush=True)


if __name__ == "__main__":
    main()
