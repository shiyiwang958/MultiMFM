#!/usr/bin/env python
"""Summarize scale-invariant and directional GLASS gradient-accuracy metrics.

Ported from dirichlet-flow-matching/scripts/summarize_gradient_accuracy_metrics.py (Tables 14-15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=None)
    return parser.parse_args(argv)


def bootstrap_ratio_ci(numerator: np.ndarray, denominator: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(numerator), size=(bootstrap, len(numerator)))
    samples = np.sqrt(numerator[ids].sum(axis=1) / denominator[ids].sum(axis=1).clip(min=1e-24))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bootstrap_mean_ci(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(values), size=(bootstrap, len(values)))
    samples = values[ids].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    with open(run_dir / "run_metadata.json") as handle:
        metadata = json.load(handle)
    bootstrap = int(args.bootstrap or metadata["args"].get("bootstrap", 2000))
    seed = int(metadata["args"]["seed"])
    rows: list[dict[str, float | int]] = []
    for length_metadata in metadata["per_length"]:
        length = int(length_metadata["length"])
        path = run_dir / f"L{length}" / "gradient_errors.csv"
        raw = pd.read_csv(path)
        required = {"error_l2", "reference_l2"}
        if not required.issubset(raw.columns):
            raise ValueError(f"{path} lacks {sorted(required - set(raw.columns))}; replay pair metrics first.")
        for mc, group in raw.groupby("mc", sort=True):
            # Bootstrap at the independent probe level. Repeats are retained
            # inside each probe's MC-noise average.
            per_probe = group.groupby("probe").agg(
                squared_error=("error_l2", lambda x: float(np.mean(np.square(x)))),
                squared_reference=("reference_l2", lambda x: float(np.mean(np.square(x)))),
            )
            numerator = per_probe["squared_error"].to_numpy()
            denominator = per_probe["squared_reference"].to_numpy()
            relative_rmse = float(np.sqrt(numerator.sum() / denominator.sum().clip(min=1e-24)))
            rel_low, rel_high = bootstrap_ratio_ci(numerator, denominator, bootstrap, seed + length + int(mc))
            row: dict[str, float | int] = {
                "length": length,
                "mc": int(mc),
                "global_relative_rmse": relative_rmse,
                "global_relative_rmse_ci95_low": rel_low,
                "global_relative_rmse_ci95_high": rel_high,
                "n_probes": int(len(per_probe)),
                "n_repeats": int(group["repeat"].nunique()),
            }
            if "cosine_similarity" in group:
                cosine_by_probe = group.groupby("probe")["cosine_similarity"].mean().to_numpy()
                cos_low, cos_high = bootstrap_mean_ci(cosine_by_probe, bootstrap, seed + 10_000 + length + int(mc))
                row.update(
                    {
                        "mean_cosine_similarity": float(cosine_by_probe.mean()),
                        "cosine_similarity_ci95_low": cos_low,
                        "cosine_similarity_ci95_high": cos_high,
                        "median_cosine_similarity": float(np.median(cosine_by_probe)),
                    }
                )
            rows.append(row)
    summary = pd.DataFrame(rows)
    out_path = run_dir / "gradient_accuracy_relative_directional_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
