#!/usr/bin/env python
"""One shard (length x reward x student) of the dMFM scale-free MC gradient analysis.

Reruns the Fig 5 / Tables 14-15 protocol with the **dMFM** posterior sampler on the
finite-N side and the published GLASS-2048 estimator as the reference. Every other
ingredient is taken from :mod:`dmfm.experiments.ablate_glass_gradient_mc` (the script
that produced the published numbers) so the cells stay comparable one to one:

* reward calibration on the base DFM (1,024 samples, 64 NFE, ``t=0.999``);
* the 32 conditioning states = base flow from Gaussian noise to ``--t_eval`` (32 NFE);
* identical seed arithmetic for the states, the reference pool and every candidate pool;
* the GLASS-2048 reference gradient (``--reference_glass_steps`` Euler steps to
  ``--glass_end_time``, exact autodiff through the sampler);
* the same ``gradient_pairs.pt`` layout, so
  :mod:`dmfm.experiments.rescore_gradient_pairs_scale_free` (Tables 14-15) and
  :mod:`dmfm.experiments.summarize_gradient_accuracy_metrics` run on it unchanged.

Outputs (``<output_root>/<student tag>/<reward>/``)::

    L{L}/gradient_pairs.pt              reference + estimate gradients (rescore input)
    L{L}/gradient_errors.csv            per (probe, repeat, N) row, published schema
    L{L}/gradient_error_summary.csv     per-N E_RMS summary, published schema
    L{L}/score_calibration_samples.npy  reward calibration draws
    L{L}/metadata.json                  written last; doubles as the "shard done" marker
    per_length/L{L}.json                copy of the metadata, for `collect` to assemble

Usage::

    python -m dmfm.scalefree.run_shard --length 50 --reward motif --student dmfm
    python -m dmfm.scalefree.run_shard --length 50 --reward motif --student dmfm4 \
        --reference_from outputs/dna/scale_free_mc_dmfm/diag1step/motif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm import paths
from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import (
    estimate_value_gradient as estimate_dmfm_value_gradient,
)
from dmfm.experiments.ablate_glass_gradient_mc import (
    bootstrap_ci,
    calibrate_score,
    estimate_value_gradient as estimate_glass_value_gradient,
    make_probe_states,
    motif_tensor,
    seeded_randn,
)
from dmfm.scalefree import LENGTHS, MC_VALUES, REWARDS, STUDENT_TAGS

# Published protocol (results/dna/scale_free_mc/*/run_metadata.json -> "args").
PUBLISHED_SEED = 20260726


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, required=True, choices=list(LENGTHS))
    p.add_argument("--reward", choices=list(REWARDS), required=True)
    p.add_argument(
        "--student",
        choices=sorted(STUDENT_TAGS),
        default="dmfm",
        help="dmfm: the diagonal+ESD students of Tables 13/17 (the paper's posterior sampler); "
        "dmfm4: the 4-step ESD students of Table 18.",
    )
    p.add_argument("--dmfm_ckpt", default=None, help="Override the student checkpoint (args.json must sit next to it).")
    p.add_argument("--base_ckpt", default=None, help="Override the base DFM checkpoint (calibration + reference).")
    p.add_argument("--output_root", default=str(paths.OUTPUTS / "scale_free_mc_dmfm"))
    p.add_argument("--student_tag", default=None, help="Override the <student tag> output subdirectory.")
    p.add_argument(
        "--reference_from",
        default=None,
        help="Reuse the GLASS-2048 reference gradients from another run directory of this same "
        "protocol (they do not depend on the student), instead of recomputing them.",
    )
    p.add_argument("--skip_existing", action="store_true", help="Exit 0 if this shard's output already exists.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch_threads", type=int, default=None, help="Pin torch CPU threads (keeps fp32 reduction order fixed across probe-range parts).")

    # Probe-range sharding: needed on CPU, where a whole 32-state shard is hours long and
    # serial_requeue is preemptible. A strict subrange writes L{L}/parts/probes_A_B.* and
    # `dmfm.scalefree.collect` merges a complete set of parts into the usual shard files.
    p.add_argument("--probe_start", type=int, default=0)
    p.add_argument("--probe_end", type=int, default=None, help="Exclusive; defaults to --n_probes.")
    p.add_argument(
        "--calibration_from",
        default=None,
        help="Take score_mean/std/center and the reward z-target from an existing run's "
        "L{L}/metadata.json instead of recomputing the 1,024-sample calibration. Use the "
        "published run when reusing its reference, so the reward is identical by construction "
        "(recomputing it costs ~18 min at L=50 and ~106 min at L=400 on CPU).",
    )

    # dMFM posterior sampler: one composed flow-map step (App F.2).
    p.add_argument("--dmfm_sampler", choices=["flow_map", "one_step", "diagonal_rk4"], default="flow_map")
    p.add_argument("--dmfm_steps", type=int, default=1)
    p.add_argument("--dmfm_end_time", type=float, default=1.0)

    # Published protocol knobs; the defaults reproduce the Fig 5 / Tables 14-15 runs.
    p.add_argument("--seed", type=int, default=PUBLISHED_SEED)
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--nfe_probe", type=int, default=32)
    p.add_argument("--nfe_calibration", type=int, default=64)
    p.add_argument("--calibration_samples", type=int, default=1024)
    p.add_argument("--sample_batch_size", type=int, default=32)
    p.add_argument("--n_probes", type=int, default=32)
    p.add_argument("--n_repeats", type=int, default=8)
    p.add_argument("--reference_mc", type=int, default=2048)
    p.add_argument("--reference_glass_steps", type=int, default=8, help="--nfe_value of the published run.")
    p.add_argument("--reference_glass_solver", choices=["euler", "rk4"], default="euler")
    p.add_argument("--glass_end_time", type=float, default=1.0)
    p.add_argument("--mc_values", type=int, nargs="+", default=list(MC_VALUES))
    p.add_argument("--mc_chunk", type=int, default=16)
    p.add_argument("--reference_mc_chunk", type=int, default=None, help="Defaults to --mc_chunk.")
    p.add_argument("--z_target", type=float, default=1.0)
    p.add_argument("--reward_beta", type=float, default=1.0)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--reward_objective", choices=["target", "maximize"], default="target")
    p.add_argument("--motif", default="TTTTTC")
    p.add_argument("--motif2", default="AAAATT")
    p.add_argument("--motif_tau", type=float, default=0.10)
    p.add_argument("--conjunction_tau", type=float, default=0.10)
    p.add_argument("--target_percentile", type=float, default=0.90)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args(argv)


def run_dir_for(output_root: str | Path, student_tag: str, reward: str) -> Path:
    return Path(output_root) / student_tag / reward


def part_stem(probe_start: int, probe_end: int) -> str:
    return f"probes_{int(probe_start):03d}_{int(probe_end):03d}"


def _atomic(path: Path, write) -> None:
    """Write via a unique temp file plus os.replace.

    The same shard may be queued on several partitions at once (the arrays are
    skip-if-exists, but the check happens at task start, so two routes can still overlap).
    Without this, two writers could interleave and leave a truncated ``gradient_pairs.pt``.
    Same seeds mean same content, so whichever lands last is equally valid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_published_calibration(run_dir: str | Path, length: int, reward: str) -> dict[str, float]:
    """score_mean / score_std / score_center / z_target from an existing run's metadata."""
    path = Path(run_dir) / f"L{length}" / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"--calibration_from has no {path}")
    with open(path) as handle:
        meta = json.load(handle)
    if str(meta.get("reward", {}).get("name")) != reward:
        raise ValueError(f"{path} is reward {meta.get('reward', {}).get('name')!r}, not {reward!r}.")
    if int(meta["length"]) != int(length):
        raise ValueError(f"{path} is L={meta['length']}, not L={length}.")
    return {
        "score_mean": float(meta["score_mean"]),
        "score_std": float(meta["score_std"]),
        "score_center": float(meta["score_center"]),
        "z_target": float(meta["reward"]["z_target"]),
        "source": str(path),
    }


def _load_reference_gradients(run_dir: Path, length: int, n_probes: int) -> torch.Tensor:
    pairs_path = Path(run_dir) / f"L{length}" / "gradient_pairs.pt"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"--reference_from has no {pairs_path}")
    reference = torch.load(pairs_path, map_location="cpu", weights_only=False)["reference_gradients"]
    # Indexed by absolute probe index, so a run with more states than we need is fine.
    if reference.shape[0] < n_probes:
        raise ValueError(
            f"{pairs_path} holds only {reference.shape[0]} reference gradients, need {n_probes}."
        )
    return reference


def run_shard(args: argparse.Namespace) -> Path:
    length = int(args.length)
    tag = args.student_tag or STUDENT_TAGS[args.student]
    run_dir = run_dir_for(args.output_root, tag, args.reward)
    out_dir = run_dir / f"L{length}"
    probe_start = max(0, int(args.probe_start))
    probe_end = int(args.probe_end) if args.probe_end is not None else int(args.n_probes)
    probe_end = min(probe_end, int(args.n_probes))
    if probe_end <= probe_start:
        raise ValueError(f"empty probe range [{probe_start}, {probe_end}).")
    is_part = (probe_start, probe_end) != (0, int(args.n_probes))
    marker = (
        out_dir / "parts" / f"{part_stem(probe_start, probe_end)}.json" if is_part else out_dir / "metadata.json"
    )
    if args.skip_existing and marker.is_file():
        print(f"shard already complete: {marker}", flush=True)
        return out_dir
    (out_dir / "parts" if is_part else out_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "per_length").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device} but CUDA is unavailable.")
    if args.torch_threads:
        torch.set_num_threads(int(args.torch_threads))
    # Same numerical setup as the published run.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from dmfm import api

    base_path = Path(args.base_ckpt) if args.base_ckpt else paths.base_ckpt(length)
    base, cfg = api.load_base(length, device, ckpt=base_path)
    if int(cfg.seq_len) != length or int(cfg.alphabet_size) != 4:
        raise ValueError(f"Base checkpoint {base_path} has seq_len={cfg.seq_len}, alphabet={cfg.alphabet_size}.")
    dmfm_path = Path(args.dmfm_ckpt) if args.dmfm_ckpt else None
    student = api.load_dmfm(length, device, kind=args.student, ckpt=dmfm_path)
    dmfm_path = dmfm_path or (paths.dmfm_ckpt(length) if args.student == "dmfm" else paths.dmfm4_ckpt(length))
    print(f"L={length} {args.reward} {tag}: base={paths.rel(base_path)} student={paths.rel(dmfm_path)}", flush=True)

    motif = motif_tensor(args.motif, device)
    motif2 = motif_tensor(args.motif2, device)
    started = time.time()

    # --- reward calibration: identical to the published run (base DFM samples), or taken
    # straight from an existing run's metadata (exact, and free on CPU).
    calibration_source = None
    if args.calibration_from:
        stored = _load_published_calibration(args.calibration_from, length, args.reward)
        score_mean, score_std = stored["score_mean"], stored["score_std"]
        score_center, reward_z_target = stored["score_center"], stored["z_target"]
        calibration_source = stored["source"]
        print(f"  calibration reused from {calibration_source}", flush=True)
    else:
        calibration = calibrate_score(
            base,
            cfg,
            n_samples=args.calibration_samples,
            batch_size=args.sample_batch_size,
            nfe=args.nfe_calibration,
            seed=args.seed + 100_000 * length,
            device=device,
            reward=args.reward,
            motif=motif,
            motif2=motif2,
            motif_tau=args.motif_tau,
            conjunction_tau=args.conjunction_tau,
        )
        score_mean = float(calibration.mean())
        score_std = float(calibration.std(ddof=1))
        if not np.isfinite(score_std) or score_std <= 0.0:
            raise RuntimeError(f"Invalid {args.reward} calibration std at L={length}: {score_std}")
        if args.reward_objective == "maximize":
            score_center, reward_z_target = score_mean, 0.0
        elif args.reward == "gc":
            score_center, reward_z_target = score_mean, float(args.z_target)
        else:
            score_center, reward_z_target = float(np.quantile(calibration, args.target_percentile)), 0.0
        np.save(out_dir / "score_calibration_samples.npy", calibration)
    print(f"  calibration mean={score_mean:.6f} std={score_std:.6f} center={score_center:.6f}", flush=True)

    # --- conditioning states: the published probe construction and seed.
    states = make_probe_states(
        base,
        cfg,
        n_probes=args.n_probes,
        t_eval=args.t_eval,
        nfe=args.nfe_probe,
        seed=args.seed + 1_000_000 + length,
        device=device,
    )
    # All 32 states are built together from one seed, so every probe-range part must see the
    # same tensor; the digest lets `collect` assert that before it merges parts.
    states_sha256 = hashlib.sha256(states.detach().cpu().numpy().tobytes()).hexdigest()

    reward_common = dict(
        score_center=score_center,
        score_std=score_std,
        z_target=reward_z_target,
        reward_beta=args.reward_beta,
        reward_scale=args.reward_scale,
        reward=args.reward,
        motif=motif,
        motif2=motif2,
        motif_tau=args.motif_tau,
        conjunction_tau=args.conjunction_tau,
        reward_objective=args.reward_objective,
    )
    dmfm_common = dict(
        **reward_common,
        mc_chunk=args.mc_chunk,
        dmfm_sampler=args.dmfm_sampler,
        dmfm_steps=args.dmfm_steps,
        dmfm_end_time=args.dmfm_end_time,
    )
    reference_chunk = int(args.reference_mc_chunk or args.mc_chunk)

    cached_reference = (
        _load_reference_gradients(Path(args.reference_from), length, args.n_probes)
        if args.reference_from
        else None
    )

    rows: list[dict[str, float | int]] = []
    reference_gradients: list[torch.Tensor] = []
    estimate_gradients = torch.empty(
        (probe_end - probe_start, len(args.mc_values), args.n_repeats, length, 4), dtype=torch.float32
    )
    for probe_idx in range(probe_start, probe_end):
        x = states[probe_idx : probe_idx + 1]
        if cached_reference is not None:
            reference_grad = cached_reference[probe_idx : probe_idx + 1].to(device)
        else:
            reference_eps = seeded_randn(
                (1, args.reference_mc, length, 4),
                args.seed + 10_000_000 + 10_000 * length + probe_idx,
                device,
            )
            _, reference_grad = estimate_glass_value_gradient(
                base,
                x,
                t_eval=args.t_eval,
                eps_pool=reference_eps,
                nfe_value=args.reference_glass_steps,
                glass_end_time=args.glass_end_time,
                glass_solver=args.reference_glass_solver,
                mc_chunk=reference_chunk,
                **reward_common,
            )
            del reference_eps
        reference_l2 = reference_grad.flatten().norm().item()
        reference_gradients.append(reference_grad.cpu())

        for mc_idx, mc in enumerate(args.mc_values):
            for repeat in range(args.n_repeats):
                eps_pool = seeded_randn(
                    (1, mc, length, 4),
                    args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe_idx + 100 * mc + repeat,
                    device,
                )
                _, estimate_grad = estimate_dmfm_value_gradient(
                    student, x, t_eval=args.t_eval, eps_pool=eps_pool, **dmfm_common
                )
                error_l2 = (estimate_grad - reference_grad).flatten().norm().item()
                estimate_l2 = estimate_grad.flatten().norm().item()
                dot_product = (estimate_grad * reference_grad).sum().item()
                estimate_gradients[probe_idx - probe_start, mc_idx, repeat] = estimate_grad.cpu()
                rows.append(
                    {
                        "length": length,
                        "probe": probe_idx,
                        "repeat": repeat,
                        "mc": mc,
                        "e_rms": error_l2 / math.sqrt(4 * length),
                        "error_l2": error_l2,
                        "reference_l2": reference_l2,
                        "reference_rms": reference_l2 / math.sqrt(4 * length),
                        "estimate_l2": estimate_l2,
                        "gradient_dot_product": dot_product,
                        "cosine_similarity": dot_product / max(estimate_l2 * reference_l2, 1e-12),
                        "relative_l2_error": error_l2 / max(reference_l2, 1e-12),
                    }
                )
        print(
            f"  probe {probe_idx + 1}/{args.n_probes} "
            f"(range {probe_start}:{probe_end}, {time.time() - started:.0f}s)",
            flush=True,
        )
        del reference_grad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    pairs = {
        "reference_gradients": torch.cat(reference_gradients, dim=0),
        "estimate_gradients": estimate_gradients,
        "mc_values": torch.tensor(args.mc_values, dtype=torch.long),
        "layout": "estimate_gradients[probe, mc_index, repeat, position, alphabet]",
    }
    if is_part:
        stem = part_stem(probe_start, probe_end)
        _atomic(out_dir / "parts" / f"{stem}.csv", lambda p: raw.to_csv(p, index=False))
        _atomic(out_dir / "parts" / f"{stem}.pt", lambda p: torch.save(pairs, p))
    else:
        _atomic(out_dir / "gradient_errors.csv", lambda p: raw.to_csv(p, index=False))
        _atomic(out_dir / "gradient_pairs.pt", lambda p: torch.save(pairs, p))

    summary_rows = []
    for mc, group in raw.groupby("mc", sort=True):
        rms = group.groupby("probe")["e_rms"].mean().to_numpy()
        l2 = group.groupby("probe")["error_l2"].mean().to_numpy()
        relative = group.groupby("probe")["relative_l2_error"].mean().to_numpy()
        rms_low, rms_high = bootstrap_ci(rms, args.bootstrap, args.seed + length + int(mc))
        l2_low, l2_high = bootstrap_ci(l2, args.bootstrap, args.seed + 10_000 + length + int(mc))
        rel_low, rel_high = bootstrap_ci(relative, args.bootstrap, args.seed + 20_000 + length + int(mc))
        summary_rows.append(
            {
                "length": length,
                "mc": int(mc),
                "mean_e_rms": float(rms.mean()),
                "std_across_probes": float(rms.std(ddof=1)) if len(rms) > 1 else 0.0,
                "ci95_low": rms_low,
                "ci95_high": rms_high,
                "mean_error_l2": float(l2.mean()),
                "error_l2_ci95_low": l2_low,
                "error_l2_ci95_high": l2_high,
                "mean_relative_l2_error": float(relative.mean()),
                "relative_l2_error_ci95_low": rel_low,
                "relative_l2_error_ci95_high": rel_high,
                "n_probes": int(args.n_probes),
                "n_repeats": int(args.n_repeats),
            }
        )
    if not is_part:
        _atomic(
            out_dir / "gradient_error_summary.csv",
            lambda p: pd.DataFrame(summary_rows).to_csv(p, index=False),
        )

    metadata = {
        "length": length,
        "student_kind": args.student,
        "student_tag": tag,
        "checkpoint": paths.rel(base_path),
        "base_checkpoint": paths.rel(base_path),
        "dmfm_checkpoint": paths.rel(dmfm_path),
        "posterior_sampler": args.dmfm_sampler,
        "dmfm_steps": args.dmfm_steps,
        "dmfm_end_time": args.dmfm_end_time,
        "reference_sampler": "glass",
        "reference_glass_steps": args.reference_glass_steps,
        "reference_glass_solver": args.reference_glass_solver,
        "glass_end_time": args.glass_end_time,
        "reference_reused_from": args.reference_from,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_center": score_center,
        "dimensions": 4 * length,
        "reference_mc": args.reference_mc,
        "normalization": "E_RMS = ||gradient_estimate - gradient_reference||_2 / sqrt(4L)",
        "reference": "independent finite-MC GLASS estimator with reference_mc samples",
        "estimator": "finite-N dMFM posterior sampler (this is what changes vs the published run)",
        "reward": {
            "name": args.reward,
            "objective": args.reward_objective,
            "z_target": reward_z_target,
            "beta": args.reward_beta,
            "lambda": args.reward_scale,
            "motif": args.motif,
            "motif2": args.motif2 if args.reward == "conjunction" else None,
            "motif_tau": args.motif_tau if args.reward != "gc" else None,
            "conjunction_tau": args.conjunction_tau if args.reward == "conjunction" else None,
            "target_percentile": args.target_percentile if args.reward != "gc" else None,
        },
        "t_eval": args.t_eval,
        "nfe_probe": args.nfe_probe,
        "nfe_value": args.reference_glass_steps,
        "n_probes": args.n_probes,
        "n_repeats": args.n_repeats,
        "mc_values": list(args.mc_values),
        "mc_chunk": args.mc_chunk,
        "reference_mc_chunk": reference_chunk,
        "seed": args.seed,
        "probe_states": "base flow from Gaussian noise to t_eval (the published Fig 5 construction)",
        "probe_states_sha256": states_sha256,
        "probe_start": probe_start,
        "probe_end": probe_end,
        "calibration_reused_from": calibration_source,
        # Which machine produced these numbers (the provenance must say so per cell).
        "compute": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "matmul_precision": "tf32" if device.type == "cuda" else "fp32",
            "torch_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        },
        "elapsed_seconds": time.time() - started,
    }
    def _dump(path: Path) -> None:
        with open(path, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    # The metadata is the done-marker, so it is written last and atomically.
    if is_part:
        _atomic(out_dir / "parts" / f"{part_stem(probe_start, probe_end)}.json", _dump)
    else:
        _atomic(out_dir / "metadata.json", _dump)
        _atomic(run_dir / "per_length" / f"L{length}.json", _dump)
    print(
        f"{'part' if is_part else 'shard'} done in {metadata['elapsed_seconds']:.0f}s "
        f"({probe_end - probe_start} of {args.n_probes} states) -> {out_dir}",
        flush=True,
    )
    return out_dir


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.t_eval < 1.0:
        raise ValueError("--t_eval must be in (0, 1).")
    if max(args.mc_values) >= args.reference_mc:
        raise ValueError("--reference_mc must exceed every --mc_values entry.")
    run_shard(args)


if __name__ == "__main__":
    main()
