#!/usr/bin/env python
"""Is the measured dMFM-vs-GLASS deficit on the DNA C0 experiment *our* error?

Background.  ``docs/dmfm_glass_parity.md`` fixed two wiring defects (wrong student, a
one-sided gradient clip) and then measured, at n = 1000 per target on shared sample ids and
under the identical scale-free guidance rule, dMFM guided oracle MAE 0.193 / 0.172 against
matched GLASS-4's 0.153 / 0.124 -- a paired deficit of +0.041 +- 0.006 and +0.049 +- 0.006.
That comparison fixes ``guidance_frac = 8`` and the posterior step count at 4 for both arms.
This module asks whether the deficit survives when the *comparison* is opened up, always
symmetrically:

1. ``guidance_frac`` swept over the same grid for **both** arms under the scale-free rule
   (the n = 1000 run used gf = 8, which the n = 32 sweep suggests is GLASS's optimum and not
   dMFM's).
2. A guidance rule that is scale-free *across arms* but preserves relative magnitude *within*
   an arm (``@relnorm``), instead of flattening every step to exactly ``grad_clip``.
3. The posterior step count / MC budget at matched value-gradient NFE (``steps x mc``), so
   neither arm is pinned to the other's operating point.

Everything is matched the way ``parity_dmfm_glass_c0`` matches it: same initial noise
(``seed + i``), same frozen MC pool (``seed + 1e6 + i``) at a given ``mc``, same guide, same
oracle, same trajectory settings; only the posterior sampler and the named knobs change.

Config tokens: ``NAME[@norm|@relnorm][@gf=<float>][@clip=<float>][@mc=<int>]``.

* ``@norm``    -- the scale-free rule of the n = 1000 runs: rescale the value gradient to
  *exactly* ``grad_clip`` at every guided step (``api.guided_sample(grad_normalize=True)``).
* ``@relnorm`` -- the *relative* rule: divide the gradient by a running quantile of its own
  norm (EMA over guided steps of the batch median, ``RELNORM_EMA``) and multiply by
  ``grad_clip``, then clamp any sample whose rescaled norm exceeds ``RELNORM_TAIL *
  grad_clip``.  The typical step then has the same magnitude as under ``@norm``, so
  ``guidance_frac`` means the same thing, but a sample with an unusually large gradient still
  takes a larger step.  Nothing here is per-arm: the same recipe runs on GLASS and on dMFM.
* ``@mc=<int>`` -- MC draws for the value gradient for that configuration only.  The value
  gradient costs ``mc x n_steps`` network calls, so ``@mc`` is how a cheaper posterior spends
  its saved NFE.

Examples
--------
    python -m dmfm.experiments.rescue_dmfm_glass_c0 --n_samples 200 \
        --configs glass_euler4@norm dmfm4_fm4@norm glass_euler4@norm@gf=16 dmfm4_fm4@norm@gf=16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm import api, paths
from dmfm.experiments.parity_dmfm_glass_c0 import (
    CONFIGS as PARITY_CONFIGS,
    Bank,
    run_env,
)
from dmfm.experiments.sample_c0_guidance import hard_token_strings, score_hard

#: ``parity_dmfm_glass_c0.CONFIGS`` plus the GLASS step counts it does not define, so the
#: posterior-accuracy axis can be swept on the GLASS arm as well as on the dMFM arm.
CONFIGS: dict[str, dict] = dict(PARITY_CONFIGS)
CONFIGS.setdefault("glass_euler2", dict(kind="glass", student=None, n_steps=2, end_time=1.0,
                                        note="GLASS, 2 Euler steps"))
CONFIGS.setdefault("glass_euler8", dict(kind="glass", student=None, n_steps=8, end_time=1.0,
                                        note="GLASS, 8 Euler steps"))

#: ``@relnorm`` constants.  Fixed, not swept, and identical for both arms.
#: ``RELNORM_EMA = 0`` uses the current step's batch median, so the median sample's step is
#: *exactly* ``grad_clip`` -- the same calibration as ``@norm``, which makes the two rules
#: directly comparable. A non-zero EMA smooths the scale across guided steps, but it lags badly
#: where the gradient norm is unstable (measured: with a lagging scale the dMFM median sample's
#: post-rule norm was 3.4 instead of 10), which would silently change the drive per arm.
RELNORM_EMA = 0.0      # weight of the running estimate when updating with a new batch median
RELNORM_TAIL = 4.0     # per-sample norm ceiling, in units of ``grad_clip``


def parse_config_token(token: str) -> tuple[str, str, dict]:
    """``NAME[@norm|@relnorm][@gf=<f>][@clip=<f>][@mc=<i>]`` -> (label, base name, overrides)."""
    parts = token.split("@")
    name, over = parts[0], {}
    for piece in parts[1:]:
        if piece == "norm":
            over["scale_rule"] = "norm"
        elif piece == "relnorm":
            over["scale_rule"] = "relnorm"
        elif piece.startswith("gf="):
            over["guidance_frac"] = float(piece[3:])
        elif piece.startswith("clip="):
            over["grad_clip"] = float(piece[5:])
        elif piece.startswith("mc="):
            over["mc"] = int(piece[3:])
        else:
            raise ValueError(f"unknown config modifier {piece!r} in {token!r}")
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return token, name, over


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--targets", type=float, nargs="+", default=[1.0, -1.0])
    p.add_argument("--student_ckpt_dmfm4", default=None,
                   help="Replace the 'dmfm4' student artifact with this checkpoint, keeping every "
                        "config name and every other setting. Used to score a re-distilled 4-step "
                        "student against the same GLASS arm.")
    p.add_argument("--student_ckpt_dmfm", default=None,
                   help="Same, for the 'dmfm' (diagonal) student artifact.")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    # Table 1 settings.
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--mc_chunk", type=int, default=8)
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--nfe_traj", type=int, default=64)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--guide_t_start", type=float, default=0.50)
    p.add_argument("--guide_t_end", type=float, default=0.95)
    p.add_argument("--guidance_frac", type=float, default=8.0)
    p.add_argument("--coeff_cap", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=10.0)
    return p.parse_args(argv)


def resolve_device(spec) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RescueBank(Bank):
    """``parity_dmfm_glass_c0.Bank`` reading the extended config table.

    ``overrides`` may point either student key at a different checkpoint; the configuration
    names are unchanged, so an override run is directly comparable with a stock one.
    """

    def __init__(self, length: int, device: torch.device, overrides: dict[str, str] | None = None):
        super().__init__(length, device)
        for key, path in (overrides or {}).items():
            if path:
                self._ckpts[key] = Path(path)

    def posterior(self, name: str):
        spec = CONFIGS[name]
        if spec["kind"] == "glass":
            return api.glass_posterior_fn(self.base, n_steps=spec["n_steps"],
                                          end_time=spec["end_time"], solver="euler")
        return api.dmfm_posterior_fn(self.student(spec["student"]), sampler="flow_map",
                                     n_steps=spec["n_steps"], end_time=spec["end_time"])

    def artifact(self, name: str) -> str:
        spec = CONFIGS[name]
        return "base DFM" if spec["student"] is None else paths.rel(self._ckpts[spec["student"]])


def make_value_grad(post, log_reward, pool, *, mc_chunk: int, rule: str | None, clip: float,
                    diag: dict):
    """Value-gradient callable for ``api.guided_sample`` implementing one scale rule.

    ``diag`` accumulates per-step diagnostics: the raw batch-mean and batch-median gradient
    norm, and the batch-mean norm *after* the rule has been applied (the quantity that sets
    the guided step size, so equal post-rule norms across arms means equal drive).
    """
    state = {"running": None}

    def value_grad(x, t):
        v, g = api.value_and_grad(post, log_reward, x, t, pool, mc_chunk=mc_chunk)
        n = g.flatten(1).norm(dim=1).clamp_min(1e-12)
        diag["raw_mean"].append(float(n.mean()))
        diag["raw_median"].append(float(n.median()))
        diag["frac_above_clip"].append(float((n > clip).float().mean()))
        if rule == "norm":
            g = g * (float(clip) / n)[:, None, None]
        elif rule == "relnorm":
            med = float(n.median())
            state["running"] = med if state["running"] is None else (
                RELNORM_EMA * state["running"] + (1.0 - RELNORM_EMA) * med)
            g = g * (float(clip) / max(state["running"], 1e-12))
            n2 = g.flatten(1).norm(dim=1).clamp_min(1e-12)
            ceiling = RELNORM_TAIL * float(clip)
            g = g * (ceiling / n2).clamp(max=1.0)[:, None, None]
            diag["frac_tail_clamped"].append(float((n2 > ceiling).float().mean()))
        post_n = g.flatten(1).norm(dim=1)
        diag["post_mean"].append(float(post_n.mean()))
        diag["post_median"].append(float(post_n.median()))
        return v, g

    return value_grad


def run(args: argparse.Namespace, out: Path, device: torch.device) -> pd.DataFrame:
    bank = RescueBank(args.length, device,
                      {"dmfm4": args.student_ckpt_dmfm4, "dmfm": args.student_ckpt_dmfm})
    L, K = int(bank.cfg.seq_len), int(bank.cfg.alphabet_size)
    ids = list(range(args.sample_start, args.sample_start + args.n_samples))
    specs = [parse_config_token(tok) for tok in args.configs]
    rows: list[dict] = []
    for target in args.targets:
        log_reward = api.c0_log_reward_fn(bank.guide, target, reward_sigma=args.reward_sigma,
                                          reward_scale=args.reward_scale)
        for start in range(0, len(ids), args.batch_size):
            batch_ids = ids[start : start + args.batch_size]
            x0 = api.paired_initial_noise(batch_ids, L, seed=args.seed, device=device)
            pools: dict[int, torch.Tensor] = {}
            for label, name, over in specs:
                mc = int(over.get("mc", args.mc))
                if mc not in pools:
                    pools[mc] = api.paired_eps_pool(batch_ids, L, mc=mc, seed=args.seed,
                                                    device=device)
                post = bank.posterior(name)
                clip = float(over.get("grad_clip", args.grad_clip))
                gf = float(over.get("guidance_frac", args.guidance_frac))
                rule = over.get("scale_rule")
                diag = {k: [] for k in ("raw_mean", "raw_median", "frac_above_clip",
                                        "post_mean", "post_median", "frac_tail_clamped")}
                value_grad = make_value_grad(post, log_reward, pools[mc],
                                             mc_chunk=min(args.mc_chunk, mc), rule=rule,
                                             clip=clip, diag=diag)
                t0 = time.time()
                # The rule is applied inside ``value_grad``; ``guided_sample`` must not scale
                # again, so it gets ``grad_clip=None`` unless the stock clip-down rule is what
                # this configuration asks for.
                x_g, x_u = api.guided_sample(
                    bank.base, bank.cfg, x0, value_grad,
                    nfe_traj=args.nfe_traj, t_max=args.t_max,
                    guide_t_start=args.guide_t_start, guide_t_end=args.guide_t_end,
                    guidance_frac=gf, coeff_cap=args.coeff_cap,
                    grad_clip=(None if rule else clip), return_unguided=True,
                )
                with torch.no_grad():
                    gui_guide = score_hard(bank.guide, x_g, K).cpu().numpy()
                    ung_guide = score_hard(bank.guide, x_u, K).cpu().numpy()
                seq_g, seq_u = hard_token_strings(x_g, K), hard_token_strings(x_u, K)
                ora_g = api.oracle_score(bank.oracle, x_g.argmax(-1).cpu(), rc_average=True)
                ora_u = api.oracle_score(bank.oracle, x_u.argmax(-1).cpu(), rc_average=True)
                spec = CONFIGS[name]
                for j, sid in enumerate(batch_ids):
                    rows.append({
                        "target": float(target), "config": label, "base_config": name,
                        "sample_idx": int(sid), "artifact": bank.artifact(name),
                        "kind": spec["kind"], "n_steps": spec["n_steps"],
                        "end_time": spec["end_time"], "mc": mc,
                        "nfe_value": mc * spec["n_steps"],
                        "scale_rule": rule or "clip", "guidance_frac": gf, "grad_clip": clip,
                        "grad_norm_raw_mean": float(np.mean(diag["raw_mean"])),
                        "grad_norm_raw_median": float(np.mean(diag["raw_median"])),
                        "frac_steps_grad_above_clip": float(np.mean(diag["frac_above_clip"])),
                        "grad_norm_post_mean": float(np.mean(diag["post_mean"])),
                        "grad_norm_post_median": float(np.mean(diag["post_median"])),
                        "frac_tail_clamped": (float(np.mean(diag["frac_tail_clamped"]))
                                              if diag["frac_tail_clamped"] else 0.0),
                        "guide_unguided": float(ung_guide[j]),
                        "guide_guided": float(gui_guide[j]),
                        "oracle_unguided": float(ora_u[j]),
                        "oracle_guided": float(ora_g[j]),
                        "seq_guided": seq_g[j], "seq_unguided": seq_u[j],
                    })
                print(f"  target={target:+.0f} ids[{batch_ids[0]}:{batch_ids[-1]+1}] "
                      f"{label:26s} oracle_mae={np.abs(ora_g - target).mean():.4f} "
                      f"|g|raw={np.mean(diag['raw_mean']):.3g} "
                      f"|g|post={np.mean(diag['post_mean']):.3g} ({time.time()-t0:.0f}s)",
                      flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out / "sample_scores.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sample_scores.csv", index=False)
    return df


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "run_env": run_env(device),
            "relnorm": {"ema": RELNORM_EMA, "tail_x_grad_clip": RELNORM_TAIL},
            "student_overrides": {"dmfm4": args.student_ckpt_dmfm4,
                                  "dmfm": args.student_ckpt_dmfm},
            "configs": {k: dict(v) for k, v in CONFIGS.items()}}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    print(f"out={out} device={device} configs={args.configs}", flush=True)
    run(args, out, device)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
