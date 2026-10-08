#!/usr/bin/env python
"""Work-stealing driver for all Figure 4 shards (one identical worker per GPU).

Every shard -- the sampler calibration, the 49 pilot shards, the 8 + 8 panel-A blocks
and the 32 x 3 sweep shards -- is a unit with its own output file. A worker walks
the stages in order and, for each unit, skips it if its output exists, otherwise
claims it with an atomic ``mkdir <output>.lock`` and runs it. So

* one worker alone does everything (about 1-2 GPU-hours in total), and any number of
  extra workers just split the work -- no matter which array tasks the queue starts;
* a preempted/requeued worker releases its own stale locks on restart, and a lock
  older than ``--stale-min`` minutes without an output is taken over;
* the sweep starts only when every pilot shard exists, because the sweep's baseline
  hyperparameters are frozen from the pilot (``plan.frozen_from_pilot``); a worker
  waiting for other workers' pilot shards meanwhile does panel-A blocks.

    python -m dmfm.benchmarks.worker                 # everything
    python -m dmfm.benchmarks.worker --stages sweep  # one stage
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from dmfm import paths
from dmfm.benchmarks import plan as plan_mod
from dmfm.benchmarks.run_steering import shard_path

OUT = paths.OUTPUTS / "fig4"
STAGES = ("calibrate", "pilot", "panelA", "panelA_c0", "sweep")

PILOT_ENV = {"n_outputs": 32, "seed": 12345, "repeats": 1}
SWEEP_ENV = {"n_outputs": 100, "seed": 0, "repeats": 3}
PANEL_A = {"n_conditions": 32, "block": 4}  # 8 = the original protocol; the rest add error bars


@dataclass
class Unit:
    stage: str
    out: Path
    argv: list[str]
    module: str  # "steer" | "value"


def _owner() -> str:
    return f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('SLURM_ARRAY_TASK_ID', '-')}:{os.getpid()}"


def _job_tag() -> str:
    return f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('SLURM_ARRAY_TASK_ID', '-')}"


def units(stage: str) -> list[Unit]:
    if stage == "calibrate":
        d = OUT / "panelB" / "sweep"
        return [Unit(stage, OUT / "panelB" / "calibration.json",
                     ["--mode", "calibrate", "--n-outputs", "1024", "--seed", "0", "--out-dir", str(d)], "steer")]
    if stage in ("pilot", "sweep"):
        env = PILOT_ENV if stage == "pilot" else SWEEP_ENV
        jobs = plan_mod.PLANS[stage]
        d = OUT / "panelB" / stage
        out = []
        for rep in range(env["repeats"]):
            for i, job in enumerate(jobs):
                path = shard_path(d, job, rep, env["n_outputs"], env["seed"])
                out.append(Unit(stage, path, [
                    "--plan", stage, "--index", str(i), "--repeat", str(rep), "--out-dir", str(d),
                    "--n-outputs", str(env["n_outputs"]), "--seed", str(env["seed"])], "steer"))
        return out
    if stage in ("panelA", "panelA_c0"):
        # panelA: the recovered original protocol (synthetic SmoothDNAReward, teacher-SDE
        # reference). panelA_c0: the same, with the cyclizability reward of panel B and a
        # GLASS reference -- a robustness variant, not the submitted experiment.
        extra = [] if stage == "panelA" else ["--reward", "c0", "--reference", "glass"]
        d = OUT / stage
        out = []
        n, b = PANEL_A["n_conditions"], PANEL_A["block"]
        for start in range(0, n, b):
            end = min(n, start + b)
            path = d / f"conds_{start:02d}_{end:02d}.json"
            out.append(Unit(stage, path, [
                "--out", str(path), "--n-conditions", str(n),
                "--cond-start", str(start), "--cond-end", str(end),
                "--chunk", "512"] + extra, "value"))
        return out
    raise ValueError(stage)


def _lock(u: Unit) -> Path:
    return u.out.with_name(u.out.name + ".lock")


def claim(u: Unit, stale_min: float) -> bool:
    lock = _lock(u)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            owner = (lock / "owner").read_text()
        except OSError:
            owner = ""
        age_min = (time.time() - lock.stat().st_mtime) / 60 if lock.exists() else 0
        # our own lock from before a requeue, or someone else's that went stale
        mine = owner.split(":")[1:3] == _job_tag().split(":") and owner.split(":")[-1] != str(os.getpid())
        if not (mine or age_min > stale_min):
            return False
        shutil.rmtree(lock, ignore_errors=True)
        try:
            lock.mkdir()
        except FileExistsError:
            return False
        print(f"took over lock of {u.out.name} (owner {owner or '?'}, {age_min:.0f} min old)", flush=True)
    (lock / "owner").write_text(_owner())
    return True


def run_unit(u: Unit, device: str | None = None) -> None:
    t0 = time.time()
    print(f"[{u.stage}] {u.out.name} ...", flush=True)
    argv = list(u.argv) + (["--device", device] if device else [])
    if u.module == "steer":
        from dmfm.benchmarks import run_steering

        run_steering.main(argv)
    else:
        from dmfm.benchmarks import value_error

        value_error.main(argv)
    print(f"[{u.stage}] {u.out.name} done in {time.time() - t0:.0f}s", flush=True)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _pilot_ready() -> bool:
    """Is a complete pilot available anywhere (this tree, another tree, or already pinned)?

    The pilot only ranks baseline hyperparameter families and no pilot shard is ever
    plotted, so it is the one part that may come from a different tree or device.
    """
    if plan_mod.frozen_choice_file().exists():
        return True
    n_expected = len(plan_mod.PLANS["pilot"])
    return any(
        d.is_dir() and len(list(d.glob("*.json"))) >= n_expected for d in plan_mod.pilot_dirs()
    )


def sweep_stage(stage: str, stale_min: float, device: str | None = None) -> tuple[int, int]:
    """One pass over a stage. Returns (#missing outputs, #claimed by others)."""
    todo = [u for u in units(stage) if not u.out.exists()]
    busy = 0
    for u in todo:
        if u.out.exists():
            continue
        if not claim(u, stale_min):
            busy += 1
            continue
        try:
            if not u.out.exists():
                run_unit(u, device)
        except (Exception, SystemExit) as exc:  # keep the worker alive; release the lock
            traceback.print_exc()
            print(f"[{u.stage}] {u.out.name} FAILED: {exc}", flush=True)
        finally:
            shutil.rmtree(_lock(u), ignore_errors=True)
    missing = sum(1 for u in units(stage) if not u.out.exists())
    return missing, busy


def main(argv=None) -> None:
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--out-root", default=str(OUT), help="root of all shard outputs (default outputs/dna/fig4)")
    p.add_argument(
        "--device",
        default=None,
        help="Force a device for every unit (e.g. cpu). Default: CUDA when available. "
             "A plotted panel must be produced entirely on ONE device type -- CUDA uses "
             "TF32 matmuls and a CUDA RNG stream, CPU true fp32 and a CPU RNG -- so pair "
             "--device cpu with a separate --out-root; every shard records its device, "
             "Slurm account and partition.",
    )
    p.add_argument("--stale-min", type=float, default=45.0)
    p.add_argument("--poll-s", type=float, default=30.0)
    p.add_argument("--max-wait-min", type=float, default=240.0)
    args = p.parse_args(argv)
    OUT = Path(args.out_root)
    stages = [s for s in args.stages.split(",") if s]
    print(f"worker {_owner()} stages={stages}", flush=True)

    side = [s for s in ("panelA", "panelA_c0") if s in stages]
    for stage in stages:
        if stage == "sweep":
            # The sweep needs a complete pilot, because its baseline hyperparameters are
            # frozen from one. The pilot may live in ANOTHER tree (it is the one part that
            # may run on a different device -- it only ranks families and is never plotted),
            # so accept a complete pilot wherever it is rather than insisting on this
            # worker's own tree. A worker that was not asked to run the pilot waits for
            # somebody else's instead of silently running it.
            t0 = time.time()
            while True:
                if _pilot_ready():
                    break
                missing = len(units("pilot"))
                if "pilot" in stages:
                    missing, _ = sweep_stage("pilot", args.stale_min, args.device)
                    if missing == 0:
                        break
                for s in side:
                    sweep_stage(s, args.stale_min, args.device)
                if (time.time() - t0) / 60 > args.max_wait_min:
                    print(f"no complete pilot after {args.max_wait_min} min ({missing} missing "
                          f"here); giving up rather than sweeping on fallback hyperparameters", flush=True)
                    return
                time.sleep(args.poll_s)
            print("pilot complete; frozen:", plan_mod.frozen_from_pilot(), flush=True)
        while True:
            missing, busy = sweep_stage(stage, args.stale_min, args.device)
            if missing == 0 or busy == 0:
                break
            time.sleep(args.poll_s)
        if missing:
            print(f"stage {stage}: {missing} outputs missing after this worker's pass", flush=True)
    print("worker done", flush=True)


if __name__ == "__main__":
    main()
