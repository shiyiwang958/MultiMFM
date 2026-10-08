#!/usr/bin/env python
"""Rescore saved gradient-pair tensors as coordinate-wise MAE (Table 18).

Ported from ``dirichlet-flow-matching/scripts/rescore_gradient_pairs_mae.py`` (2026-07-27),
which produced ``gradient_mae_per_condition.csv`` / ``gradient_mae_summary.csv`` /
``gradient_mae_summary_all_lengths.csv`` in
``workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/{gc,motif,conjunction}`` (Table 18),
from the ``gradient_pairs.pt`` written by ``dmfm.experiments.ablate_dmfm_one_step_gradient_mc``.

Per length directory ``L*``: MAE over (L, 4) coordinates between each MC estimate and the
reference gradient, per (probe, mc, repeat); the per-state mean over repeats; and a
bootstrap over states (seed ``seed + L + mc``) of the mean.

Port changes (defaults reproduce the original exactly):
- ``main(argv)`` instead of reading ``sys.argv``.
- ``--lengths`` restricts which ``L*`` directories are (re)scored, so each Table 18 array task
  can rescore its own length. ``gradient_mae_summary_all_lengths.csv`` is then assembled from
  every ``L*/gradient_mae_summary.csv`` present, which equals the original concatenation when
  all lengths are scored in one call.

    python -m dmfm.experiments.rescore_gradient_pairs_mae --input-dir outputs/dna/dmfm4_glass128_mae_precision/motif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--lengths", nargs="+", type=int, default=None,
                        help="Only rescore these lengths (default: every L* directory).")
    return parser.parse_args(argv)


def _artifact_dir(length_dir: Path) -> Path:
    if (length_dir / "gradient_pairs.pt").exists():
        return length_dir
    return length_dir / length_dir.name


def rescore_length(length_dir: Path, *, bootstrap: int, seed: int) -> pd.DataFrame:
    artifact_dir = _artifact_dir(length_dir)
    length = int(length_dir.name[1:])
    pairs = torch.load(artifact_dir / "gradient_pairs.pt", map_location="cpu", weights_only=False)
    reference = pairs["reference_gradients"][:, None, None]
    estimate = pairs["estimate_gradients"]
    mae = (estimate - reference).abs().mean(dim=(-1, -2))

    rows = []
    for mc_index, mc in enumerate(pairs["mc_values"].tolist()):
        for probe in range(mae.shape[0]):
            for repeat in range(mae.shape[2]):
                rows.append({"length": length, "probe": probe, "repeat": repeat,
                             "mc": mc, "mae": mae[probe, mc_index, repeat].item()})
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(artifact_dir / "gradient_mae_per_condition.csv", index=False)
    state_mean = per_sample.groupby(["mc", "probe"], as_index=False)["mae"].mean()
    summary_rows = []
    for mc, group in state_mean.groupby("mc", sort=True):
        values = group["mae"].to_numpy()
        rng = np.random.default_rng(seed + length + int(mc))
        bootstrap_means = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
        summary_rows.append({
            "mc": mc,
            "mean_mae": values.mean(),
            "median_state_mae": np.median(values),
            "std_across_states": values.std(ddof=1),
            "ci95_low": np.quantile(bootstrap_means, 0.025),
            "ci95_high": np.quantile(bootstrap_means, 0.975),
            "n_conditions": len(values),
            "n_repeats": mae.shape[2],
        })
    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "length", length)
    summary.to_csv(artifact_dir / "gradient_mae_summary.csv", index=False)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    root = Path(args.input_dir)
    length_dirs = sorted((p for p in root.glob("L*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    if args.lengths is not None:
        wanted = set(args.lengths)
        length_dirs = [p for p in length_dirs if int(p.name[1:]) in wanted]
    for length_dir in length_dirs:
        if not (_artifact_dir(length_dir) / "gradient_pairs.pt").exists():
            raise FileNotFoundError(f"no gradient_pairs.pt under {length_dir}")
        rescore_length(length_dir, bootstrap=args.bootstrap, seed=args.seed)

    summaries = []
    for length_dir in sorted((p for p in root.glob("L*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:])):
        f = _artifact_dir(length_dir) / "gradient_mae_summary.csv"
        if f.exists():
            summaries.append(pd.read_csv(f))
    if summaries:
        out = root / "gradient_mae_summary_all_lengths.csv"
        pd.concat(summaries, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
