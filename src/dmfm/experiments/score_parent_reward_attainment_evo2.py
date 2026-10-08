#!/usr/bin/env python
"""Score paired reward-attainment samples and held-out references with Evo2.

Ported from dirichlet-flow-matching/scripts/score_parent_reward_attainment_evo2.py (2026-07-26; Evo2
column of Table 13 / App. F.1, workdir/rebuttal_reward_attainment_{motif,conjunction}_20260731/L*/evo2).
Only change: the flat ``score_yeast_c0_evo2`` import now points at
``dmfm.experiments.score_yeast_c0_evo2``; ``main`` accepts an optional argv list.
Run in the Evo2 environment (scripts/dna/evo2_environment.yml).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dmfm.experiments.score_yeast_c0_evo2 import load_evo2, reverse_complement, score_mean_logprobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="evo2_7b_base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def score_sequences(sequences: list[str], model, cli: argparse.Namespace) -> np.ndarray:
    forward, _ = score_mean_logprobs(sequences, model, batch_size=cli.batch_size, prepend_bos=False, device=cli.device)
    reverse, _ = score_mean_logprobs([reverse_complement(sequence) for sequence in sequences], model, batch_size=cli.batch_size, prepend_bos=False, device=cli.device)
    return -0.5 * (forward + reverse)


def bootstrap_mean(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    sample_dir = Path(cli.sample_dir)
    samples = pd.read_csv(sample_dir / "reward_attainment_samples.csv")
    heldout = pd.read_csv(sample_dir / "heldout_test_reference.csv")
    length = int(samples["length"].iloc[0])
    if not (samples["length"] == length).all() or not (heldout["length"] == length).all():
        raise ValueError("Expected one sequence length per sample directory")
    sets = {
        "unguided": samples[["sample_id", "sequence_unguided"]].rename(columns={"sequence_unguided": "sequence"}),
        "guided": samples[["sample_id", "sequence_guided"]].rename(columns={"sequence_guided": "sequence"}),
        "heldout_test": heldout[["test_id", "sequence"]].rename(columns={"test_id": "sample_id"}),
    }
    print(f"Loading {cli.model_name}", flush=True)
    model = load_evo2(cli.model_name)
    rows, summaries = [], []
    for offset, (name, frame) in enumerate(sets.items()):
        frame = frame.copy()
        print(f"L={length} set={name}: n={len(frame)}", flush=True)
        nll = score_sequences(frame["sequence"].tolist(), model, cli)
        frame["length"] = length
        frame["set"] = name
        frame["evo2_nll"] = nll
        frame["orientation"] = "forward_rc_mean"
        rows.append(frame)
        low, high = bootstrap_mean(nll, cli.bootstrap, cli.seed + offset)
        summaries.append({
            "length": length, "set": name, "n_sequences": len(frame), "scored_positions": length - 1,
            "orientation": "forward_rc_mean", "evo2_nll_mean": float(nll.mean()),
            "evo2_nll_ci95_low": low, "evo2_nll_ci95_high": high,
            "evo2_perplexity_from_mean_nll": float(np.exp(nll.mean())),
        })
    summary = pd.DataFrame(summaries)
    heldout_nll = float(summary.loc[summary["set"] == "heldout_test", "evo2_nll_mean"].iloc[0])
    summary["evo2_nll_delta_vs_heldout_test"] = summary["evo2_nll_mean"] - heldout_nll
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "evo2_per_sequence.csv", index=False)
    summary.to_csv(out_dir / "evo2_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
