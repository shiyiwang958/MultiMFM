#!/usr/bin/env python
"""Compute scale-free gradient diagnostics from saved gradient-pair tensors.

Ported from dirichlet-flow-matching/scripts/rescore_gradient_pairs_scale_free.py (Tables 14-15).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def bootstrap_mean(values: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args(argv)

    root = Path(args.input_dir)
    summaries = []
    for length_dir in sorted(root.glob("L*"), key=lambda path: int(path.name[1:])):
        pairs = torch.load(length_dir / "gradient_pairs.pt", map_location="cpu", weights_only=False)
        length = int(length_dir.name[1:])
        reference = pairs["reference_gradients"][:, None, None]
        estimate = pairs["estimate_gradients"]
        abs_error = (estimate - reference).abs()
        reference_l1 = reference.abs().sum(dim=(-1, -2))
        relative_l1 = abs_error.sum(dim=(-1, -2)) / reference_l1.clamp_min(1e-12)
        cosine = (estimate * reference).sum(dim=(-1, -2)) / (
            estimate.square().sum(dim=(-1, -2)).sqrt()
            * reference.square().sum(dim=(-1, -2)).sqrt()
        ).clamp_min(1e-12)

        rows, summary_rows = [], []
        for mc_index, mc in enumerate(pairs["mc_values"].tolist()):
            state_relative_l1 = relative_l1[:, mc_index].mean(dim=1).numpy()
            state_cosine = cosine[:, mc_index].mean(dim=1).numpy()
            for state, (rel, cos) in enumerate(zip(state_relative_l1, state_cosine)):
                rows.append({"length": length, "state": state, "mc": mc,
                             "relative_l1": rel, "cosine": cos})
            rel_low, rel_high = bootstrap_mean(state_relative_l1, args.bootstrap, args.seed + length + int(mc))
            cos_low, cos_high = bootstrap_mean(state_cosine, args.bootstrap, args.seed + 10_000 + length + int(mc))
            global_relative_l1 = abs_error[:, mc_index].sum().item() / (
                reference.abs().sum().item() * estimate.shape[2]
            )
            summary_rows.append({
                "length": length,
                "mc": mc,
                "global_relative_l1": global_relative_l1,
                "mean_state_relative_l1": state_relative_l1.mean(),
                "relative_l1_ci95_low": rel_low,
                "relative_l1_ci95_high": rel_high,
                "mean_cosine": state_cosine.mean(),
                "cosine_ci95_low": cos_low,
                "cosine_ci95_high": cos_high,
                "n_states": len(state_relative_l1),
                "n_repeats": estimate.shape[2],
            })

        pd.DataFrame(rows).to_csv(length_dir / "gradient_scale_free_per_state.csv", index=False)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(length_dir / "gradient_scale_free_summary.csv", index=False)
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv(root / "gradient_scale_free_summary_all_lengths.csv", index=False)


if __name__ == "__main__":
    main()
