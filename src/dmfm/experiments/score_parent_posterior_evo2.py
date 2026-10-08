#!/usr/bin/env python
"""Score reward-free posterior futures with the established Evo2 RC-NLL metric.

Ported from dirichlet-flow-matching/scripts/score_parent_posterior_evo2.py (2026-07-26; used for the
Evo2 column of the posterior-diversity rebuttal, workdir/parent_posterior_diversity_*_evo2_20260726).
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--futures_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", default="evo2_7b_base")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sources_per_cell", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def bootstrap_ci_by_source(frame: pd.DataFrame, bootstrap: int, seed: int) -> tuple[float, float]:
    source_means = frame.groupby("source_id", sort=False)["evo2_nll"].mean().to_numpy(dtype=float)
    if len(source_means) <= 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap, dtype=float)
    for i in range(bootstrap):
        draws[i] = source_means[rng.integers(0, len(source_means), len(source_means))].mean()
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def score_sequences(sequences: list[str], model, cli: argparse.Namespace) -> np.ndarray:
    forward, _ = score_mean_logprobs(sequences, model, batch_size=cli.batch_size, prepend_bos=False, device=cli.device)
    reverse, _ = score_mean_logprobs([reverse_complement(x) for x in sequences], model, batch_size=cli.batch_size, prepend_bos=False, device=cli.device)
    return -0.5 * (forward + reverse)


def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    futures = pd.read_csv(cli.futures_csv)
    required = {"length", "t", "method", "source_id", "future_id", "source_sequence", "sequence"}
    missing = required.difference(futures.columns)
    if missing:
        raise ValueError(f"Missing required future columns: {sorted(missing)}")

    selected = []
    for (length, t, method), group in futures.groupby(["length", "t", "method"], sort=True):
        source_ids = np.sort(group["source_id"].unique())[: cli.sources_per_cell]
        cell = group[group["source_id"].isin(source_ids)].copy()
        # At t=1 all 16 posterior futures are analytically identical. Score
        # each source once, while retaining the same hierarchical unit.
        if np.isclose(float(t), 1.0):
            cell = cell.sort_values(["source_id", "future_id"]).groupby("source_id", as_index=False).head(1)
        selected.append(cell)
    eval_futures = pd.concat(selected, ignore_index=True)

    # Use precisely the held-out source conditions represented in the scored
    # posterior futures. Comparing 16 selected source conditions to all 32
    # available sources would bias the reported delta by source composition.
    selected_source_ids = eval_futures[["length", "source_id"]].drop_duplicates()
    source_refs = (
        futures[["length", "source_id", "source_sequence"]]
        .drop_duplicates(["length", "source_id"])
        .merge(selected_source_ids, on=["length", "source_id"], how="inner")
        .rename(columns={"source_sequence": "sequence"})
        .assign(t=np.nan, method="heldout_source", future_id=0)
    )
    source_refs = source_refs[["length", "t", "method", "source_id", "future_id", "sequence"]]
    eval_futures = eval_futures[["length", "t", "method", "source_id", "future_id", "sequence"]]
    to_score = pd.concat([eval_futures, source_refs], ignore_index=True)

    print(f"Loading {cli.model_name}", flush=True)
    model = load_evo2(cli.model_name)
    per_sequence = []
    for length, group in to_score.groupby("length", sort=True):
        group = group.reset_index(drop=True)
        print(f"Scoring L={length}: n={len(group)}", flush=True)
        scores = score_sequences(group["sequence"].tolist(), model, cli)
        scored = group.copy()
        scored["evo2_nll"] = scores
        scored["scored_positions"] = int(length) - 1
        scored["orientation"] = "forward_rc_mean"
        per_sequence.append(scored)
    per_sequence = pd.concat(per_sequence, ignore_index=True)

    rows = []
    for (length, t, method), group in per_sequence.groupby(["length", "t", "method"], dropna=False, sort=True):
        lo, hi = bootstrap_ci_by_source(group, cli.bootstrap, cli.seed + int(length) * 1000 + (0 if pd.isna(t) else int(round(float(t) * 100))) )
        rows.append({
            "length": int(length), "t": float(t) if not pd.isna(t) else np.nan, "method": method,
            "n_sequences": int(len(group)), "n_sources": int(group["source_id"].nunique()),
            "scored_positions": int(length) - 1, "orientation": "forward_rc_mean",
            "evo2_nll_mean": float(group["evo2_nll"].mean()),
            "evo2_nll_std": float(group["evo2_nll"].std(ddof=0)),
            "evo2_nll_ci95_low": lo, "evo2_nll_ci95_high": hi,
            "evo2_perplexity_from_mean_nll": float(np.exp(group["evo2_nll"].mean())),
        })
    summary = pd.DataFrame(rows).sort_values(["length", "method", "t"], na_position="last", ignore_index=True)
    refs = summary[summary["method"] == "heldout_source"][["length", "evo2_nll_mean"]].rename(columns={"evo2_nll_mean": "heldout_source_nll"})
    summary = summary.merge(refs, on="length", how="left")
    summary["evo2_nll_delta_vs_heldout_source"] = summary["evo2_nll_mean"] - summary["heldout_source_nll"]
    per_sequence.to_csv(out_dir / "posterior_evo2_per_sequence.csv", index=False)
    summary.to_csv(out_dir / "posterior_evo2_summary.csv", index=False)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
