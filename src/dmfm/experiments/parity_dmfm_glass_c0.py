#!/usr/bin/env python
"""dMFM-vs-GLASS parity harness for the C0 (cyclizability) guidance experiment of Table 1.

Why this exists
---------------
The Table 1 rerun with a dMFM posterior (``scripts/dna/table1_c0_guidance_dmfm.sbatch``,
job 49277977) integrates **four** composed flow-map steps (``--nfe_value 4``) of the
*diagonal* L=50 student ``checkpoints/dna/dmfm/L50`` and lands far from GLASS, while
Table 18 -- which measures the *gradient* of the same kind of posterior and agrees with
GLASS -- uses the **4-step ESD** student ``checkpoints/dna/dmfm_4step/L50`` at four steps.
This module runs every (student, step-count, end-time) combination through the *same*
sampler with the *same* initial noise and the *same* frozen MC pool, so the configurations
differ in nothing but the posterior sampler.

Two modes, both writing into ``results/dna/parity/`` by default:

``--mode grad``
    No guided sampling. Probe states are taken from the base flow at a few outer times
    ``t``; for each configuration the finite-MC value gradient is compared against a
    high-step GLASS posterior evaluated on the *identical* MC pool, so the only source of
    disagreement is the posterior sampler (not the MC noise). Reports cosine similarity,
    relative L2 error and per-coordinate MAE, plus the endpoint statistics (guide score,
    simplex mass, argmax margin). Cheap: runs on CPU in minutes at ``--n_probes 8``.

``--mode sample``
    The end-to-end Table 1 protocol: paired unguided/guided sampling with
    :func:`dmfm.api.guided_sample` at the exact Table 1 settings, one configuration after
    another on the same ``sample_ids``, scored with the independent park_cnn oracle
    (reverse-complement averaged, as Table 1) and with the guide. Reports guided oracle MAE
    per configuration and the per-sample paired difference against the GLASS configuration.

Examples
--------
    python -m dmfm.experiments.parity_dmfm_glass_c0 --mode grad --n_probes 8
    python -m dmfm.experiments.parity_dmfm_glass_c0 --mode sample --n_samples 32 \
        --targets 1.0 -1.0 --out_dir results/dna/parity/sample_n32
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm import api, paths
from dmfm.experiments.sample_c0_guidance import hard_token_strings, score_hard

# Every configuration is (posterior kind, student artifact, inner steps, inner end time).
# "student" is a key of STUDENT_CKPTS; None for GLASS.
CONFIGS: dict[str, dict] = {
    # Table 1 as printed and its n=1000 rerun: GLASS Euler, 4 steps, to 1.0.
    "glass_euler4": dict(kind="glass", student=None, n_steps=4, end_time=1.0,
                         note="Table 1 / Fig 3 posterior (GLASS, 4 Euler steps)"),
    # What job 49277977 actually ran for the "dMFM" Table 1: the diagonal student, 4 steps.
    "diag_fm4": dict(kind="dmfm", student="dmfm", n_steps=4, end_time=1.0,
                     note="as shipped in table1_c0_guidance_dmfm.sbatch (suspected defect)"),
    # The diagonal student at the jump Tables 13/17 use it at (one step, gap 1.0).
    "diag_fm1": dict(kind="dmfm", student="dmfm", n_steps=1, end_time=1.0,
                     note="diagonal student, one-step map (Tables 13/17 pairing)"),
    # The author's claim: the 4-step ESD student integrated with the 4 steps it was distilled for.
    "dmfm4_fm4": dict(kind="dmfm", student="dmfm4", n_steps=4, end_time=1.0,
                      note="4-step ESD student at 4 steps (Table 18 pairing)"),
    # Controls isolating the step count and the end time.
    "dmfm4_fm1": dict(kind="dmfm", student="dmfm4", n_steps=1, end_time=1.0,
                      note="4-step student forced to one step"),
    "dmfm4_fm4_e999": dict(kind="dmfm", student="dmfm4", n_steps=4, end_time=0.999,
                           note="4-step student at 4 steps, Table 18's end_time"),
    "diag_fm4_e999": dict(kind="dmfm", student="dmfm", n_steps=4, end_time=0.999,
                          note="diagonal student at 4 steps, Table 18's end_time"),
    "dmfm4_fm2": dict(kind="dmfm", student="dmfm4", n_steps=2, end_time=1.0,
                      note="4-step student at 2 steps (gap 0.5, above its 0.25 training gap)"),
    "dmfm4_fm8": dict(kind="dmfm", student="dmfm4", n_steps=8, end_time=1.0,
                      note="4-step student at 8 steps (gap 0.125, inside its training range)"),
    # Reference-quality GLASS posterior; only meaningful in --mode sample as an upper bound.
    "glass_euler32": dict(kind="glass", student=None, n_steps=32, end_time=1.0,
                          note="GLASS with 32 Euler steps (posterior-accuracy ceiling)"),
}

DEFAULT_GRAD_CONFIGS = ["glass_euler4", "diag_fm4", "diag_fm1", "dmfm4_fm4", "dmfm4_fm1",
                        "dmfm4_fm4_e999", "diag_fm4_e999", "dmfm4_fm2", "dmfm4_fm8"]
DEFAULT_SAMPLE_CONFIGS = ["glass_euler4", "diag_fm4", "diag_fm1", "dmfm4_fm4", "dmfm4_fm1",
                          "dmfm4_fm4_e999"]


def parse_config_token(token: str) -> tuple[str, str, dict]:
    """``NAME[@gf=<float>][@clip=<float>][@norm]`` -> (label, base config name, overrides).

    ``@gf`` / ``@clip`` change ``guidance_frac`` / ``grad_clip`` for that configuration only.
    ``@norm`` rescales the value gradient to *exactly* ``grad_clip`` instead of only clipping
    it down, which is the scale-free comparison: GLASS's gradient norm is above the clip at
    every guided step, so for GLASS the stock clip already acts as a normaliser, while for a
    dMFM posterior with a smaller gradient norm it is a no-op.
    """
    parts = token.split("@")
    name, over = parts[0], {}
    for piece in parts[1:]:
        if piece == "norm":
            over["normalize_grad"] = True
        elif piece.startswith("gf="):
            over["guidance_frac"] = float(piece[3:])
        elif piece.startswith("clip="):
            over["grad_clip"] = float(piece[5:])
        else:
            raise ValueError(f"unknown config modifier {piece!r} in {token!r}")
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return token, name, over


def student_ckpts(length: int) -> dict[str, Path]:
    return {"dmfm": paths.dmfm_ckpt(length), "dmfm4": paths.dmfm4_ckpt(length)}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["grad", "sample"], default="grad")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--configs", nargs="+", default=None,
                   help=f"subset of {sorted(CONFIGS)}; default depends on --mode")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    # Table 1 settings (shared by both modes).
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--mc_chunk", type=int, default=8)
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--targets", type=float, nargs="+", default=[1.0, -1.0])
    # --mode grad
    p.add_argument("--n_probes", type=int, default=8)
    p.add_argument("--probe_times", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--reference_steps", type=int, default=100,
                   help="GLASS steps of the reference posterior (--mode grad).")
    p.add_argument("--reference_solver", choices=["euler", "rk4"], default="rk4",
                   help="Reference posterior solver (--mode grad). Table 18's reference is rk4.")
    p.add_argument("--reference_end_time", type=float, default=0.999,
                   help="Reference posterior end time (--mode grad). Below 1 avoids the singular "
                   "linear-DFM data-time velocity; integrating a fine grid all the way to 1.0 "
                   "drives the endpoint onto a hard corner and its value gradient to ~0.")
    # --mode sample
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
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


class Bank:
    """Lazily loaded base DFM, C0 guide/oracle and dMFM students."""

    def __init__(self, length: int, device: torch.device):
        self.length = int(length)
        self.device = device
        self.base, self.cfg = api.load_base(length, device)
        self.guide = api.load_c0("guide", device)
        self._oracle = None
        self._students: dict[str, object] = {}
        self._ckpts = student_ckpts(length)

    @property
    def oracle(self):
        if self._oracle is None:
            self._oracle = api.load_c0("oracle", self.device)
        return self._oracle

    def student(self, key: str):
        if key not in self._students:
            self._students[key] = api.load_dmfm(self.length, self.device, ckpt=self._ckpts[key])
        return self._students[key]

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


def run_env(device: torch.device) -> dict:
    import os

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or "cpu"),
        "torch": torch.__version__,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False,
        "host": platform.node(),
        **{k.lower(): os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT")},
    }


# ------------------------------------------------------------------ --mode grad

def endpoint_stats(endpoint: torch.Tensor, guide) -> dict[str, float]:
    """Cheap shape diagnostics of a posterior endpoint plus its guide score."""
    with torch.no_grad():
        x = endpoint.detach()
        srt = x.sort(dim=-1, descending=True).values
        score = guide(x)
        hard = torch.nn.functional.one_hot(x.argmax(-1), x.shape[-1]).float()
        score_hard_ = guide(hard)
    return {
        "endpoint_sum": float(x.sum(-1).mean()),
        "endpoint_min": float(x.min()),
        "endpoint_max": float(x.max()),
        "argmax_margin": float((srt[..., 0] - srt[..., 1]).mean()),
        "guide_score_soft": float(score.mean()),
        "guide_score_hard": float(score_hard_.mean()),
        "soft_hard_gap": float((score - score_hard_).abs().mean()),
    }


def run_grad(args: argparse.Namespace, out: Path, device: torch.device) -> pd.DataFrame:
    bank = Bank(args.length, device)
    names = [tok.split("@")[0] for tok in (args.configs or DEFAULT_GRAD_CONFIGS)]
    L = int(bank.cfg.seq_len)
    ids = list(range(args.sample_start, args.sample_start + args.n_probes))
    x0 = api.paired_initial_noise(ids, L, seed=args.seed, device=device)
    pool = api.paired_eps_pool(ids, L, mc=args.mc, seed=args.seed, device=device)
    reference = api.glass_posterior_fn(bank.base, n_steps=args.reference_steps,
                                      end_time=args.reference_end_time, solver=args.reference_solver)

    rows: list[dict] = []
    for target in args.targets:
        log_reward = api.c0_log_reward_fn(bank.guide, target, reward_sigma=args.reward_sigma,
                                         reward_scale=args.reward_scale)
        for t in args.probe_times:
            # Probe state: the base flow run from the shared initial noise up to t (the states the
            # Table 1 guided sampler conditions on).
            nfe = max(1, int(round(t / (args.t_max / args.nfe_traj))))
            x_t = api.sample_unguided(bank.base, bank.cfg, x0, nfe=nfe, t_max=float(t))
            _, g_ref = api.value_and_grad(reference, log_reward, x_t, float(t), pool, mc_chunk=args.mc_chunk)
            ref_norm = g_ref.flatten(1).norm(dim=1)
            t_vec = torch.full((x_t.shape[0],), float(t), device=device)
            with torch.no_grad():
                ep_ref = reference(pool[:, 0], x_t, t_vec)
                tok_ref = ep_ref.argmax(-1)
            for name in names:
                post = bank.posterior(name)
                v, g = api.value_and_grad(post, log_reward, x_t, float(t), pool, mc_chunk=args.mc_chunk)
                diff = (g - g_ref).flatten(1)
                gn = g.flatten(1).norm(dim=1)
                cos = torch.nn.functional.cosine_similarity(g.flatten(1), g_ref.flatten(1), dim=1)
                with torch.no_grad():
                    ep = post(pool[:, 0], x_t, t_vec)
                    ep_l1 = float((ep - ep_ref).abs().sum(-1).mean())
                    tok_agree = float((ep.argmax(-1) == tok_ref).float().mean())
                per_sample_rel = (diff.norm(dim=1) / ref_norm.clamp_min(1e-12))
                rows.append({
                    "target": float(target), "t": float(t), "config": name,
                    "artifact": bank.artifact(name), "n_steps": CONFIGS[name]["n_steps"],
                    "end_time": CONFIGS[name]["end_time"],
                    "value_mean": float(v.mean()),
                    "grad_norm": float(gn.mean()), "grad_norm_median": float(gn.median()),
                    "ref_grad_norm": float(ref_norm.mean()),
                    "norm_ratio_median": float((gn / ref_norm.clamp_min(1e-12)).median()),
                    "cosine": float(cos.mean()), "cosine_min": float(cos.min()),
                    # Per-sample relative L2 can divide by a near-zero reference norm (the
                    # posterior gradient collapses at late t), so report a median and a pooled
                    # ratio alongside the mean.
                    "rel_l2": float(per_sample_rel.mean()),
                    "rel_l2_median": float(per_sample_rel.median()),
                    "rel_l2_pooled": float(diff.norm() / ref_norm.norm().clamp_min(1e-12)),
                    "mae": float(diff.abs().mean()),
                    "endpoint_l1_vs_ref": ep_l1, "endpoint_token_agree_vs_ref": tok_agree,
                    **endpoint_stats(ep, bank.guide),
                })
                print(f"  target={target:+.0f} t={t:.2f} {name:16s} cos={rows[-1]['cosine']:+.4f} "
                      f"rel_l2_med={rows[-1]['rel_l2_median']:.4f} |g|={rows[-1]['grad_norm']:.3g} "
                      f"|g|/|g_ref|={rows[-1]['norm_ratio_median']:.3g}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "grad_agreement.csv", index=False)
    summary = (df.groupby("config")[["cosine", "rel_l2_median", "rel_l2_pooled", "mae", "grad_norm",
                                     "norm_ratio_median", "endpoint_l1_vs_ref",
                                     "endpoint_token_agree_vs_ref", "guide_score_soft",
                                     "argmax_margin"]]
                 .mean().reset_index().sort_values("cosine", ascending=False))
    summary.to_csv(out / "grad_agreement_summary.csv", index=False)
    print("\n== gradient agreement vs GLASS-%d %s to %.4g on the identical MC pool ==" %
          (args.reference_steps, args.reference_solver, args.reference_end_time))
    print(summary.to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------- --mode sample

def run_sample(args: argparse.Namespace, out: Path, device: torch.device) -> pd.DataFrame:
    bank = Bank(args.length, device)
    names = args.configs or DEFAULT_SAMPLE_CONFIGS
    L, K = int(bank.cfg.seq_len), int(bank.cfg.alphabet_size)
    ids = list(range(args.sample_start, args.sample_start + args.n_samples))
    specs = [parse_config_token(tok) for tok in names]
    rows: list[dict] = []
    for target in args.targets:
        log_reward = api.c0_log_reward_fn(bank.guide, target, reward_sigma=args.reward_sigma,
                                         reward_scale=args.reward_scale)
        for start in range(0, len(ids), args.batch_size):
            batch_ids = ids[start : start + args.batch_size]
            x0 = api.paired_initial_noise(batch_ids, L, seed=args.seed, device=device)
            pool = api.paired_eps_pool(batch_ids, L, mc=args.mc, seed=args.seed, device=device)
            for label, name, over in specs:
                post = bank.posterior(name)
                clip = over.get("grad_clip", args.grad_clip)
                norms: list[float] = []

                def value_grad(x, t, _post=post, _clip=clip, _over=over, _norms=norms):
                    v, g = api.value_and_grad(post, log_reward, x, t, pool, mc_chunk=args.mc_chunk)
                    _norms.append(float(g.flatten(1).norm(dim=1).mean()))
                    if _over.get("normalize_grad"):
                        n = g.flatten(1).norm(dim=1).clamp_min(1e-12)
                        g = g * (float(_clip) / n)[:, None, None]
                    return v, g

                t0 = time.time()
                x_g, x_u = api.guided_sample(
                    bank.base, bank.cfg, x0, value_grad,
                    nfe_traj=args.nfe_traj, t_max=args.t_max, guide_t_start=args.guide_t_start,
                    guide_t_end=args.guide_t_end,
                    guidance_frac=over.get("guidance_frac", args.guidance_frac),
                    coeff_cap=args.coeff_cap, grad_clip=clip, return_unguided=True,
                )
                with torch.no_grad():
                    gui_guide = score_hard(bank.guide, x_g, K).cpu().numpy()
                    ung_guide = score_hard(bank.guide, x_u, K).cpu().numpy()
                seq_g, seq_u = hard_token_strings(x_g, K), hard_token_strings(x_u, K)
                ora_g = api.oracle_score(bank.oracle, x_g.argmax(-1).cpu(), rc_average=True)
                ora_u = api.oracle_score(bank.oracle, x_u.argmax(-1).cpu(), rc_average=True)
                mean_norm = float(np.mean(norms)) if norms else float("nan")
                frac_clipped = float(np.mean([n > clip for n in norms])) if norms else float("nan")
                for j, sid in enumerate(batch_ids):
                    rows.append({
                        "target": float(target), "config": label, "base_config": name,
                        "sample_idx": int(sid),
                        "artifact": bank.artifact(name), "n_steps": CONFIGS[name]["n_steps"],
                        "end_time": CONFIGS[name]["end_time"],
                        "guidance_frac": over.get("guidance_frac", args.guidance_frac),
                        "grad_clip": clip, "normalize_grad": bool(over.get("normalize_grad", False)),
                        "grad_norm_mean": mean_norm, "frac_steps_grad_above_clip": frac_clipped,
                        "guide_unguided": float(ung_guide[j]), "guide_guided": float(gui_guide[j]),
                        "oracle_unguided": float(ora_u[j]), "oracle_guided": float(ora_g[j]),
                        "seq_guided": seq_g[j], "seq_unguided": seq_u[j],
                    })
                print(f"  target={target:+.0f} ids[{batch_ids[0]}:{batch_ids[-1]+1}] {label:22s} "
                      f"oracle_mae={np.abs(ora_g - target).mean():.4f} |g|={mean_norm:.3g} "
                      f"clipped={frac_clipped:.2f} ({time.time()-t0:.0f}s)", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out / "sample_scores.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sample_scores.csv", index=False)
    summarize_sample(df, out, baseline="glass_euler4" if "glass_euler4" in names else names[0])
    return df


def summarize_sample(df: pd.DataFrame, out: Path, *, baseline: str) -> pd.DataFrame:
    recs = []
    for (target, config), grp in df.groupby(["target", "config"]):
        base = df[(df.target == target) & (df.config == baseline)].set_index("sample_idx")
        g = grp.set_index("sample_idx")
        paired = (g["oracle_guided"] - target).abs() - (base["oracle_guided"] - target).abs()
        recs.append({
            "target": target, "config": config, "n": len(grp),
            "grad_norm_mean": float(grp["grad_norm_mean"].mean()),
            "frac_steps_grad_above_clip": float(grp["frac_steps_grad_above_clip"].mean()),
            "unguided_oracle_mae": float((grp["oracle_unguided"] - target).abs().mean()),
            "guided_oracle_mae": float((grp["oracle_guided"] - target).abs().mean()),
            "guided_oracle_mean": float(grp["oracle_guided"].mean()),
            "guided_guide_mae": float((grp["guide_guided"] - target).abs().mean()),
            "paired_improvement": float(((grp["oracle_unguided"] - target).abs()
                                         - (grp["oracle_guided"] - target).abs()).mean()),
            "frac_improved": float((((grp["oracle_unguided"] - target).abs()
                                     - (grp["oracle_guided"] - target).abs()) > 0).mean()),
            f"paired_mae_minus_{baseline}": float(paired.mean()),
            f"paired_mae_minus_{baseline}_sem": float(paired.std(ddof=1) / max(1, len(paired)) ** 0.5),
        })
    summary = pd.DataFrame(recs).sort_values(["target", "guided_oracle_mae"])
    summary.to_csv(out / "sample_summary.csv", index=False)
    print("\n== guided oracle MAE, matched noise, matched MC pool, matched guidance ==")
    print(summary.to_string(index=False), flush=True)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir or (paths.RESULTS / "parity" / f"{args.mode}_L{args.length}"))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "run_env": run_env(device),
            "configs": {k: {**v, "ckpt": (None if v["student"] is None
                                          else paths.rel(student_ckpts(args.length)[v["student"]]))}
                        for k, v in CONFIGS.items()}}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"mode={args.mode} device={device} out={out}", flush=True)
    if args.mode == "grad":
        run_grad(args, out, device)
    else:
        run_sample(args, out, device)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
