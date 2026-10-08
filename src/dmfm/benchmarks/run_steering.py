#!/usr/bin/env python
"""Figure 4 (right): one (method, budget, repeat) shard of the target-error-vs-NFE sweep.

    reward  r(x) = -(f_cyc(x) - y*)^2   (Sec. 3.1 cyclizability guide, y* = 0.30)
    metric  |f_cyc(x) - y*| of the returned hard sequence, mean over n_outputs
    x-axis  measured generative-model NFE per returned sample

All five methods share the base flow (parent-disjoint L=50 DFM), the guide
(``checkpoints/dna/c0/guide``), the target, the 96-step grid, the seeds and the
scoring code; the counters in :class:`dmfm.benchmarks.core.NFECounter` are
*measured*, not predicted, and nothing assumes parallel execution.

Modes
-----
``run``        execute one job of a plan (``--plan pilot|sweep --index i --repeat r``)
``calibrate``  unguided C0 distribution at several stochasticity levels eta, to
               confirm the SDE sampler used for branching has the same marginals
               as the deterministic sampler
``list``       print the plan as JSON (used to size the Slurm array)

Each shard writes one JSON and skips itself if that JSON already exists, so a
preempted array task reruns alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dmfm import paths
from dmfm.benchmarks import plan as plan_mod
from dmfm.benchmarks.core import (
    NFECounter,
    SamplerConfig,
    ShardResult,
    load_base_flow,
    load_c0,
    load_dmfm_student,
    summarize_errors,
    tokens_to_strings,
)
from dmfm.benchmarks.dmfm_steering import GuidanceConfig, PosteriorSampler, sample_dmfm_guided
from dmfm.benchmarks.search import RewardSpec, beam_search, best_of_n, feynman_kac, mcts


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["run", "calibrate", "list"], default="run")
    p.add_argument("--plan", choices=sorted(plan_mod.PLANS), default="sweep")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n-outputs", type=int, default=100)
    p.add_argument("--n-steps", type=int, default=96, help="base-flow steps (the original's --nfe_sample)")
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--sigma-sq-cap", type=float, default=10.0)
    p.add_argument("--target-c0", type=float, default=0.30)
    p.add_argument("--reward-sigma", type=float, default=0.15)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--guidance-frac", type=float, default=8.0)
    p.add_argument("--coeff-cap", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--grad-normalize", action="store_true",
                   help="scale-free gradient rule: rescale the gradient to exactly "
                        "--grad-clip instead of only clipping it down (the convention every "
                        "other DNA result in this paper uses)")
    p.add_argument("--guide-t-start", type=float, default=0.01)
    p.add_argument("--guide-t-end", type=float, default=0.95)
    p.add_argument("--fk-t-min", type=float, default=0.01)
    p.add_argument("--fk-t-max", type=float, default=0.95)
    p.add_argument("--batch-rows", type=int, default=2048)
    p.add_argument("--dmfm-batch", type=int, default=25)
    p.add_argument("--seed", type=int, default=0, help="base seed; the shard seed is seed + 1000*index + repeat")
    p.add_argument("--save-sequences", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _sampler(cli, eta: float) -> SamplerConfig:
    return SamplerConfig(n_steps=cli.n_steps, t_max=cli.t_max, eta=float(eta), sigma_sq_cap=cli.sigma_sq_cap)


# --------------------------------------------------------------------------- shard identity
# A shard's file name carries its plan job, n_outputs, base seed and repeat, and the JSON
# records every setting that changes its numbers. An existing shard is reused only if
# its recorded settings equal the requested ones; otherwise the run refuses to start,
# so two submissions with different settings can never silently mix in one directory.


def shard_settings(cli, job, repeat: int) -> dict:
    return {
        "plan": cli.plan,
        "method": job.method,
        "config": job.label,
        "config_params": dict(job.config),
        "repeat": int(repeat),
        "n_outputs": int(cli.n_outputs),
        "base_seed": int(cli.seed),
        "length": int(cli.length),
        "n_steps": int(cli.n_steps),
        "t_max": float(cli.t_max),
        "sigma_sq_cap": float(cli.sigma_sq_cap),
        "target_c0": float(cli.target_c0),
        "reward_sigma": float(cli.reward_sigma),
        "reward_scale": float(cli.reward_scale),
        "guidance": [cli.guidance_frac, cli.coeff_cap, cli.grad_clip, cli.guide_t_start,
                     cli.guide_t_end, bool(cli.grad_normalize)],
        "fk_window": [cli.fk_t_min, cli.fk_t_max],
    }


def shard_path(out_dir, job, repeat: int, n_outputs: int, seed: int) -> Path:
    return Path(out_dir) / f"{job.key}__n{int(n_outputs)}_s{int(seed)}__rep{int(repeat)}.json".replace("/", "_")


def calibration_settings(cli) -> dict:
    return {
        "n_outputs": int(cli.n_outputs),
        "seed": int(cli.seed),
        "length": int(cli.length),
        "n_steps": int(cli.n_steps),
        "t_max": float(cli.t_max),
        "sigma_sq_cap": float(cli.sigma_sq_cap),
        "target_c0": float(cli.target_c0),
        "etas": list(CALIBRATION_ETAS),
    }


CALIBRATION_ETAS = (0.0, 0.25, 0.5, 1.0)


class SettingsMismatch(SystemExit):
    pass


def reuse_existing(path: Path, settings: dict) -> bool:
    """True if ``path`` exists with exactly ``settings``; raise if it exists with others."""
    if not path.exists():
        return False
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SettingsMismatch(f"{path} exists but is not valid JSON ({exc}); remove it to rerun") from exc
    recorded = stored.get("settings") or stored.get("meta", {}).get("settings")
    requested = json.loads(json.dumps(settings))  # same JSON normalisation as the stored copy
    if recorded != requested:
        if recorded is None:
            diff = "none recorded (file written by an older version of this code)"
        else:
            diff = {
                k: {"stored": recorded.get(k), "requested": v}
                for k, v in requested.items()
                if recorded.get(k) != v
            }
        raise SettingsMismatch(
            f"refusing to reuse {path}: it was produced with different settings {diff}. "
            "Use another --out-dir, or delete the file if it is stale."
        )
    return True


def _exclusive_lock(path: Path, stale_min: float = 30.0) -> bool:
    """O_EXCL lock file; a lock older than ``stale_min`` minutes is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.uname().nodename}:{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = (time.time() - path.stat().st_mtime) / 60
            except FileNotFoundError:
                continue
            if age <= stale_min:
                return False
            path.unlink(missing_ok=True)
    return False


def run_job(cli, job, repeat: int, out_path: Path) -> ShardResult:
    device = torch.device(cli.device) if cli.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = int(cli.seed) + 1000 * int(cli.index) + int(repeat)
    torch.manual_seed(seed)
    np.random.seed(seed)

    nfe = NFECounter()
    flow = load_base_flow(cli.length, device, nfe)
    guide = load_c0("guide", device, nfe, rc_average=False)
    oracle_nfe = NFECounter()
    oracle = load_c0("oracle", device, oracle_nfe, rc_average=True)
    reward = RewardSpec(target=cli.target_c0, sigma=cli.reward_sigma, scale=cli.reward_scale, hard=True)

    cfg = dict(job.config)
    eta = float(cfg.pop("eta", 0.0))
    sampler = _sampler(cli, eta)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    t0 = time.time()
    if job.method == "dmfm":
        student = load_dmfm_student(cli.length, device)
        post = PosteriorSampler(flow, student, nfe)
        guidance = GuidanceConfig(
            guidance_frac=cli.guidance_frac,
            coeff_cap=cli.coeff_cap,
            grad_clip=cli.grad_clip,
            grad_normalize=cli.grad_normalize,
            t_start=cli.guide_t_start,
            t_end=cli.guide_t_end,
            n_mc=int(cfg["n_mc"]),
            nfe_value=int(cfg.get("nfe_value", 1)),
        )
        out = sample_dmfm_guided(
            flow, post, guide, reward,
            n_outputs=cli.n_outputs, sampler=sampler, guidance=guidance,
            batch_size=cli.dmfm_batch, nfe=nfe,
        )
    elif job.method == "best_of_n":
        out = best_of_n(
            flow, guide, reward, n_outputs=cli.n_outputs, sampler=sampler,
            generator=gen, batch_rows=cli.batch_rows, **cfg
        )
    elif job.method == "fk":
        out = feynman_kac(
            flow, guide, reward, n_outputs=cli.n_outputs, sampler=sampler, generator=gen,
            t_min=cli.fk_t_min, t_max=cli.fk_t_max, **cfg
        )
    elif job.method == "beam":
        out = beam_search(flow, guide, reward, n_outputs=cli.n_outputs, sampler=sampler, generator=gen, **cfg)
    elif job.method == "mcts":
        out = mcts(flow, guide, reward, n_outputs=cli.n_outputs, sampler=sampler, generator=gen, **cfg)
    else:
        raise ValueError(f"unknown method {job.method!r}")

    tokens = out["tokens"]
    scores = out["score"].detach().float().cpu().numpy()
    seqs = tokens_to_strings(tokens.cpu())
    with torch.no_grad():
        oracle_scores = oracle(torch.nn.functional.one_hot(tokens, flow.K).float()).cpu().numpy()

    settings = shard_settings(cli, job, repeat)
    metrics = summarize_errors(scores, cli.target_c0)
    metrics.update({f"oracle_{k}": v for k, v in summarize_errors(oracle_scores, cli.target_c0).items()})
    result = ShardResult(
        method=job.method,
        config=job.label,
        config_params=dict(job.config),
        repeat=repeat,
        seed=seed,
        n_outputs=cli.n_outputs,
        metrics=metrics,
        nfe=nfe.per_output(cli.n_outputs) | {"nfe_gen_total": nfe.gen, "eta": eta},
        per_sample={
            "guide_score": [float(v) for v in scores],
            "oracle_score": [float(v) for v in oracle_scores],
            **({"sequence": seqs} if cli.save_sequences else {}),
        },
        meta={
            "elapsed_s": time.time() - t0,
            "device": str(device),
            "device_type": device.type,
            # gpu_requeue is a mixed pool (A100 / H100 / H200 / RTX Pro), so record which
            # GPU actually ran the shard; kempner_requeue is H100 only.
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            # TF32 is on for CUDA and irrelevant on CPU; recorded so a shard always
            # says what arithmetic produced it (see results/dna/fig4/PROVENANCE.md).
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "settings": settings,
            "plan": cli.plan,
            "plan_index": cli.index,
            "base_seed": cli.seed,
            "n_steps": cli.n_steps,
            "target_c0": cli.target_c0,
            "reward": "-(f_cyc(x) - y*)^2, scaled by 1/(2 sigma^2), sigma=%.3g" % cli.reward_sigma,
            "checkpoints": {
                "base": str(paths.base_ckpt(cli.length)),
                "dmfm": str(paths.dmfm_ckpt(cli.length)),
                "c0_guide": str(paths.c0_guide()),
                "c0_oracle": str(paths.c0_oracle()),
            },
        },
    )
    result.write(out_path)
    print(
        f"{job.key} rep{repeat}: mae={metrics['mae']:.4f} frac10={metrics['frac10']:.2f} "
        f"nfe_gen/out={result.nfe['nfe_gen_per_output']:.0f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return result


def calibrate(cli, out_path: Path) -> None:
    """Does the eta > 0 SDE sampler have the same unguided C0 distribution as eta = 0?"""
    device = torch.device(cli.device) if cli.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfe = NFECounter()
    flow = load_base_flow(cli.length, device, nfe)
    guide = load_c0("guide", device, nfe, rc_average=False)
    settings = calibration_settings(cli)
    if reuse_existing(out_path, settings):
        print(f"calibration exists with the same settings, skipping: {out_path}")
        return
    lock = out_path.with_name(out_path.name + ".lock")
    if not _exclusive_lock(lock):
        print(f"calibration is being computed by another process ({lock}); skipping")
        return
    rows = []
    for eta in CALIBRATION_ETAS:
        sampler = _sampler(cli, eta)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cli.seed) + 991)
        scores = []
        left = int(cli.n_outputs)
        while left > 0:
            b = min(512, left)
            x = flow.integrate(flow.prior(b, generator=gen), sampler, generator=gen)
            with torch.no_grad():
                scores.append(guide(torch.nn.functional.one_hot(x.argmax(-1), flow.K).float()).cpu().numpy())
            left -= b
        s = np.concatenate(scores)
        rows.append(
            {
                "eta": eta,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "q05": float(np.quantile(s, 0.05)),
                "q50": float(np.quantile(s, 0.50)),
                "q95": float(np.quantile(s, 0.95)),
                "frac_within_0.10_of_target": float((np.abs(s - cli.target_c0) <= 0.10).mean()),
                "scores": [float(v) for v in s],
            }
        )
        print(f"eta={eta}: mean={s.mean():.4f} std={s.std(ddof=1):.4f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "settings": settings,
                "_what": "Unguided C0 distribution of the base flow at several stochasticity levels eta. "
                "eta=0 is the deterministic interpolant sampler used by every other DNA experiment; "
                "eta>0 is the SDE sampler (Eq. 2 with V=0) that the branching baselines need. Equal "
                "distributions mean eta only adds trajectory-level randomness, not a distribution shift.",
                "n_steps": cli.n_steps,
                "sigma_sq_cap": cli.sigma_sq_cap,
                "seed": cli.seed,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(tmp, out_path)  # atomic: readers never see a partial file
    lock.unlink(missing_ok=True)
    print(f"wrote {out_path}")


def main(argv=None) -> None:
    cli = parse_args(argv)
    jobs = plan_mod.PLANS[cli.plan]
    out_dir = Path(cli.out_dir or (paths.OUTPUTS / "fig4" / "panelB" / cli.plan))

    if cli.mode == "list":
        print(json.dumps([{"index": i, **j.__dict__} for i, j in enumerate(jobs)], indent=2, default=str))
        return
    if cli.mode == "calibrate":
        calibrate(cli, out_dir.parent / "calibration.json")
        return

    if not 0 <= cli.index < len(jobs):
        raise SystemExit(f"--index {cli.index} out of range for plan {cli.plan!r} ({len(jobs)} jobs)")
    job = jobs[cli.index]
    out_path = shard_path(out_dir, job, cli.repeat, cli.n_outputs, cli.seed)
    if reuse_existing(out_path, shard_settings(cli, job, cli.repeat)):
        print(f"shard exists with the same settings, skipping: {out_path}")
        return
    run_job(cli, job, cli.repeat, out_path)


if __name__ == "__main__":
    main()
