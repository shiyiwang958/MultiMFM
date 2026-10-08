#!/usr/bin/env python
"""CPU check that the dMFM rerun really matches the published Fig 5 protocol.

Everything except the finite-N estimator is shared with the published run, so it can be
checked directly against the published outputs, without a GPU and without touching the
dMFM student:

1. the per-length reward calibration (mean, std, center) vs the published
   ``L{L}/metadata.json``;
2. a handful of finite-N **GLASS** estimate gradients, recomputed through exactly the
   states/calibration/seeds/estimator that :mod:`dmfm.scalefree.run_shard` uses, vs
   ``estimate_l2`` in the published ``L{L}/gradient_errors.csv``.

If both agree, the conditioning states, the seed arithmetic, the reward and the GLASS
integrator all match, so the GLASS-2048 reference of the rerun is the published reference
and only the finite-N estimator differs. On CPU the comparison is fp32 against the
published TF32 run, so a few tenths of a percent (more for the tiny small-N gradients) is
expected; see ``tests/dna/check_gradient_mc_reference.py`` for the TF32 noise floor.

The published outputs live outside this repo::

    python -m dmfm.scalefree.check_published_protocol \
        --published-root /n/holylabs/.../paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm.experiments.ablate_glass_gradient_mc import (
    calibrate_score,
    estimate_value_gradient,
    make_probe_states,
    motif_tensor,
    seeded_randn,
)
from dmfm.scalefree.run_shard import PUBLISHED_SEED
from dmfm.scalefree.verify_reference import DEFAULT_PUBLISHED_RUNS

# (probe, N, repeat) cells to recompute; cheap and spread over states, N and repeats.
DEFAULT_CELLS = ((0, 1, 0), (0, 4, 3), (1, 2, 0), (3, 1, 1))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--published-root", required=True, help="Directory holding the published run directories.")
    p.add_argument("--published-run", action="append", default=[], metavar="REWARD=DIRNAME")
    p.add_argument("--reward", choices=sorted(DEFAULT_PUBLISHED_RUNS), default="motif")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=PUBLISHED_SEED)
    p.add_argument("--t_eval", type=float, default=0.50)
    p.add_argument("--out", default=None, help="Write the report as JSON here.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    runs = dict(DEFAULT_PUBLISHED_RUNS)
    for spec in args.published_run:
        reward, name = spec.split("=", 1)
        runs[reward] = name
    published = Path(args.published_root) / runs[args.reward] / f"L{args.length}"
    with open(published / "metadata.json") as handle:
        published_meta = json.load(handle)

    from dmfm import api

    device = torch.device(args.device)
    length = int(args.length)
    base, cfg = api.load_base(length, device)
    motif = motif_tensor("TTTTTC", device)
    motif2 = motif_tensor("AAAATT", device)

    calibration = calibrate_score(
        base, cfg, n_samples=1024, batch_size=32, nfe=64, seed=args.seed + 100_000 * length,
        device=device, reward=args.reward, motif=motif, motif2=motif2, motif_tau=0.10,
        conjunction_tau=0.10,
    )
    score_std = float(calibration.std(ddof=1))
    if args.reward == "gc":
        score_center, z_target = float(calibration.mean()), 1.0
    else:
        score_center, z_target = float(np.quantile(calibration, 0.90)), 0.0
    calibration_report = {
        "score_mean": [float(calibration.mean()), published_meta["score_mean"]],
        "score_std": [score_std, published_meta["score_std"]],
        "score_center": [score_center, published_meta["score_center"]],
    }
    print("calibration (ours, published):")
    for key, (ours, theirs) in calibration_report.items():
        print(f"  {key:13s} {ours:.9f}  {theirs:.9f}  rel dev {abs(ours - theirs) / abs(theirs):.2e}")

    states = make_probe_states(
        base, cfg, n_probes=32, t_eval=args.t_eval, nfe=32,
        seed=args.seed + 1_000_000 + length, device=device,
    )
    rows = pd.read_csv(published / "gradient_errors.csv")
    common = dict(
        score_center=score_center, score_std=score_std, z_target=z_target, reward_beta=1.0,
        reward_scale=1.0, reward=args.reward, motif=motif, motif2=motif2, motif_tau=0.10,
        conjunction_tau=0.10, nfe_value=8, glass_end_time=1.0, glass_solver="euler",
        mc_chunk=16, reward_objective="target",
    )
    cells = []
    for probe, mc, repeat in DEFAULT_CELLS:
        eps = seeded_randn(
            (1, mc, length, 4),
            args.seed + 20_000_000 + 1_000_000 * length + 10_000 * probe + 100 * mc + repeat,
            device,
        )
        _, gradient = estimate_value_gradient(
            base, states[probe : probe + 1], t_eval=args.t_eval, eps_pool=eps, **common
        )
        ours = gradient.flatten().norm().item()
        theirs = float(rows[(rows.probe == probe) & (rows.mc == mc) & (rows.repeat == repeat)].estimate_l2.iloc[0])
        cells.append({"probe": probe, "N": mc, "repeat": repeat, "ours": ours,
                      "published": theirs, "rel_dev": abs(ours - theirs) / max(theirs, 1e-30)})
    frame = pd.DataFrame(cells)
    print("\nfinite-N GLASS estimate gradient norms:")
    print(frame.to_string(index=False))
    print(f"\nworst relative deviation: {frame.rel_dev.max():.2%} (fp32 CPU vs the published TF32 run)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(
                {"reward": args.reward, "length": length, "device": args.device,
                 "published_run": str(published), "calibration": calibration_report,
                 "cells": cells, "worst_rel_dev": float(frame.rel_dev.max())},
                handle, indent=2, sort_keys=True,
            )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
