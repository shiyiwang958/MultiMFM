#!/usr/bin/env python
"""Evo2 plausibility scoring of C0-steered yeast samples and real-data baselines (Table 6).

DERIVED FROM THE ORIGINAL SOURCE, not from a reconstruction. The upstream file is

    github.com/tullebulle/DNA-MFM@187fe7b  scripts/score_yeast_c0_evo2.py
    13,706 bytes, sha256 059498df13c607cee8736fabb41f64d1b6ddfecab3f1d758c761f76787fafbe9

read from the durable clone on holylabs
(``/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/DNA-MFM_187fe7b``,
a read-only source) and kept verbatim at ``results/dna/regen/original/score_yeast_c0_evo2.py``
with its hash in that directory's ``_sources.json``. This module is that file with the edits
listed below and nothing else; the derivation is scripted and re-checkable.

History: in ``dirichlet-flow-matching`` only the bytecode
(``scripts/__pycache__/score_yeast_c0_evo2.cpython-39.pyc``) survived the purge, so this module
was first rebuilt from the Codex session that wrote it and checked against that bytecode (all 22
code objects agreed; disassembly in ``docs/provenance/dna/score_yeast_c0_evo2.dis.txt``). The
upstream source was found later. Re-deriving from it changed **no** executable statement: the
reconstruction and this file are identical under ``ast.dump`` with docstrings stripped.

THE SCORING PATH IS UNTOUCHED. Ignoring docstrings and comments, these functions are
AST-identical to the original -- tokenisation, padding, the optional BOS, windowing, strand
handling, the reduction and everything that is averaged:

    prepare_batch, score_mean_logprobs, score_rows, bootstrap_ci, summarize,
    reverse_complement, tokens_to_seqs, model_forward, load_evo2, generated_rows,
    cap_df, run_metadata, load_args_payload

In particular, per sequence Evo2 scores ``mean_j log p(x_j | x_<j)`` over the L-1 predictable
positions (L-1+1 with ``--prepend_bos``); with ``--reverse_complement`` (the default) the forward
and reverse-complement means are averaged with weight 0.5 each; NLL = -mean_logprob; the CI is a
2,000-draw nonparametric bootstrap of the per-sequence NLL; ``evo2_perplexity_from_mean_nll`` is
``exp(mean NLL)``; and ``evo2_nll_delta_vs_split_test`` is measured against ``split_test_sample``.

EXHAUSTIVE LIST OF DEVIATIONS (imports, paths and CLI wiring only; five functions touched):

1. ``parse_args`` gains an ``argv`` parameter and passes it to ``parse_args(argv)``, so the
   module is callable in-process (notebooks, tests) as well as from the command line.
2. ``parse_args`` defaults become repo-relative via :mod:`dmfm.paths`. The originals were
   cwd-relative paths into the purged split65k tree and are kept in comments beside each:
   ``--train_pt data/yeast_mid50.pt``,
   ``--split_pt data/yeast_mid50_split_seed0_train65000_val8000.pt``,
   ``--out_dir workdir/yeast_split_dmfm_c0_evo2``.
3. ``parse_args`` gains ``--train_split`` / ``--val_split`` / ``--test_split``. The original
   hard-coded the split keys ``train`` / ``val`` / ``test`` inside ``baseline_rows``; those keys
   exist only in the purged split65k split file. The defaults here are the parent-disjoint
   equivalents ``a_train`` / ``a_val`` / ``test``. **Passing ``--train_split train --val_split val
   --test_split test`` restores the original behaviour exactly.**
4. ``baseline_rows`` gains a ``split_names`` parameter carrying those keys. Its default is the
   ORIGINAL triple ``("train", "val", "test")``, so calling it positionally is unchanged.
5. ``discover_csvs``: its default glob pointed at the purged split65k sample directories
   (``workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv``) and now points at the
   shipped parent-disjoint n=1,000 C0 sets (``DEFAULT_RUN_GLOB``). Its local variable ``paths``
   is renamed ``paths_found`` because ``dmfm.paths`` is now imported under that name.
6. ``load_train_tokens``: ``torch.load`` -> ``dmfm.utils.torch_io.torch_load``
   (``weights_only=False``). The original ran under torch 2.7, where that was the default; the
   payloads are plain tensors or a dict of tensors, so nothing about the data changes.
7. ``main`` gains the same ``argv`` parameter, passes ``split_names`` through, and names the
   default glob in its "no files matched" error.
8. Imports: ``from utils.yeast_splits import ...`` -> ``dmfm.utils.yeast_splits``; plus
   ``dmfm.paths`` and ``dmfm.utils.torch_io``.

None of the above touches a number Table 6 reports. What DOES change Table 6 is the input:
the paper's split65k sample sets, split file and models are all purged, so the shipped defaults
score the parent-disjoint n=1,000 C0 sets instead. See ``results/dna/regen/PROVENANCE.md``.

Provenance of the published Table 6: run ``workdir/yeast_split_dmfm_c0_evo2_2026-06-10``
(outputs purged; its summary CSV survives in ``results/dna/june_split65k_recovered/``) with

    PYTHONPATH=. .envs/evo2/bin/python scripts/score_yeast_c0_evo2.py \
      --run_glob 'workdir/yeast_split_dmfm_c0_samples_target_p1_2026-06-02/sample_scores.csv' \
      --run_glob 'workdir/yeast_split_dmfm_c0_samples_target_m1_2026-06-02/sample_scores.csv' \
      --train_pt data/yeast_mid50.pt --split_pt data/yeast_mid50_split_seed0_train65000_val8000.pt \
      --out_dir workdir/yeast_split_dmfm_c0_evo2_2026-06-10 --targets -1 1 --baseline_n 1000 \
      --batch_size 1 --bootstrap 2000 --model_name evo2_7b_base --device cuda:0

Run it in the separate Evo2 environment (``scripts/dna/regen/evo2_environment.yml`` plus
``evo2_postinstall.py``); ``evo2`` is imported lazily in :func:`load_evo2`, so this module
imports fine without it.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from dmfm import paths
from dmfm.utils.torch_io import torch_load
from dmfm.utils.yeast_splits import load_yeast_split_indices


INT_TO_DNA = np.array(list("ACGT"))
RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")

# Original default: "workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv"
# (split65k sample dirs, purged). Port default: the shipped parent-disjoint n=1,000 C0 sets.
DEFAULT_RUN_GLOB = str(paths.RESULTS / "c0_guidance" / "n1000" / "target_*" / "sample_scores.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score clean-split yeast dMFM C0 samples and real-data baselines with Evo2. "
            "Outputs per-sequence mean log-likelihood, per-base NLL, and perplexity."
        )
    )
    p.add_argument(
        "--run_glob",
        action="append",
        default=None,
        help="Glob(s) for sample_scores.csv. Can be passed multiple times. "
        f"Default: {paths.rel(DEFAULT_RUN_GLOB)}",
    )
    # Original default: data/yeast_mid50.pt (split65k, purged).
    p.add_argument("--train_pt", default=str(paths.data_pt(50)))
    # Original default: data/yeast_mid50_split_seed0_train65000_val8000.pt (split65k, purged).
    p.add_argument("--split_pt", default=str(paths.split_pt(50)))
    # Added in the port; the original hard-coded train/val/test inside baseline_rows.
    p.add_argument("--train_split", default="a_train", help="Split key for split_train_sample (original: train).")
    p.add_argument("--val_split", default="a_val", help="Split key for split_val_sample (original: val).")
    p.add_argument("--test_split", default="test", help="Split key for split_test_sample (original: test).")
    # Original default: workdir/yeast_split_dmfm_c0_evo2
    p.add_argument("--out_dir", default=str(paths.OUTPUTS / "yeast_parent_c0_evo2"))
    p.add_argument("--model_name", default="evo2_7b_base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--baseline_n",
        type=int,
        default=1000,
        help="Number of real sequences sampled from each reference split.",
    )
    p.add_argument(
        "--max_per_set",
        type=int,
        default=None,
        help="Optional cap per generated/baseline set for smoke tests.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend Evo2 EOD token before scoring so every base has a previous token.",
    )
    p.add_argument(
        "--reverse_complement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average forward and reverse-complement mean log-likelihoods.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--targets",
        type=float,
        nargs="*",
        default=None,
        help="Optional target_c0 values to keep, e.g. --targets -1 1.",
    )
    return p.parse_args(argv)


def load_train_tokens(path: str | Path) -> np.ndarray:
    payload = torch_load(path, map_location="cpu")
    seqs = payload["seqs"] if isinstance(payload, dict) and "seqs" in payload else payload
    if not isinstance(seqs, torch.Tensor):
        raise TypeError(f"Expected tensor or dict with 'seqs' in {path}, got {type(seqs)}")
    arr = seqs.detach().cpu().numpy().astype(np.int16, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected train tokens [N,L], got {arr.shape}")
    if arr.min() < 0 or arr.max() > 3:
        raise ValueError(f"Expected DNA integer tokens in {{0,1,2,3}}, got min={arr.min()} max={arr.max()}")
    return arr


def tokens_to_seqs(tokens: np.ndarray) -> list[str]:
    return ["".join(INT_TO_DNA[row]) for row in tokens]


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TRANS)[::-1].upper()


def discover_csvs(patterns: list[str] | None) -> list[str]:
    # Local renamed from `paths` because `dmfm.paths` is imported under that name.
    paths_found: list[str] = []
    for pat in patterns or [DEFAULT_RUN_GLOB]:
        paths_found.extend(glob.glob(pat))
    return sorted(set(paths_found))


def load_args_payload(path: str | Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    return payload.get("args", payload)


def run_metadata(csv_path: str | Path) -> dict:
    args_path = Path(csv_path).with_name("args.json")
    if not args_path.exists():
        return {}
    return load_args_payload(args_path)


def cap_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=int(n), random_state=seed).sort_index().reset_index(drop=True)


def generated_rows(csvs: list[str], targets: set[float] | None, max_per_set: int | None, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for csv_path in csvs:
        meta = run_metadata(csv_path)
        target = float(meta.get("target_c0", np.nan))
        if targets is not None and target not in targets:
            continue
        df = pd.read_csv(csv_path)
        for seq_col, set_name, score_col in [
            ("seq_unguided", "unguided", "unguided"),
            ("seq_guided", "guided", "guided"),
        ]:
            if seq_col not in df.columns:
                continue
            part = pd.DataFrame(
                {
                    "source": "generated",
                    "set": set_name,
                    "target_c0": target,
                    "run_dir": str(Path(csv_path).parent),
                    "sample_idx": df["sample_idx"].to_numpy() if "sample_idx" in df else np.arange(len(df)),
                    "score": df[score_col].to_numpy() if score_col in df else np.nan,
                    "seq": df[seq_col].astype(str).str.upper().to_numpy(),
                }
            )
            parts.append(cap_df(part, max_per_set, seed))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def baseline_rows(
    train_pt: str | Path,
    split_pt: str | Path,
    baseline_n: int,
    max_per_set: int | None,
    seed: int,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Real-data reference sets. ``split_names`` (added in the port) defaults to the original keys."""
    train_split, val_split, test_split = split_names
    full = load_train_tokens(train_pt)
    references = {
        "full_sample": full,
        "split_train_sample": full[load_yeast_split_indices(split_pt, train_split).numpy()],
        "split_val_sample": full[load_yeast_split_indices(split_pt, val_split).numpy()],
        "split_test_sample": full[load_yeast_split_indices(split_pt, test_split).numpy()],
    }
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for set_name, tokens in references.items():
        n = min(int(baseline_n), tokens.shape[0])
        idx = np.sort(rng.choice(tokens.shape[0], size=n, replace=False))
        seqs = tokens_to_seqs(tokens[idx])
        part = pd.DataFrame(
            {
                "source": "reference",
                "set": set_name,
                "target_c0": np.nan,
                "run_dir": "reference_data",
                "sample_idx": idx,
                "score": np.nan,
                "seq": seqs,
            }
        )
        rows.append(cap_df(part, max_per_set, seed))
    return pd.concat(rows, ignore_index=True)


def load_evo2(model_name: str):
    try:
        from evo2 import Evo2
    except ImportError as exc:
        raise ImportError(
            "Could not import Evo2. Install a Python 3.11/3.12 Evo2 environment first; "
            "the official package requires Python >=3.11,<3.13."
        ) from exc
    model = Evo2(model_name)
    return model


def model_forward(evo2_model, input_ids: torch.Tensor) -> torch.Tensor:
    model = getattr(evo2_model, "model", evo2_model)
    outputs = model(input_ids)
    first = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if isinstance(first, (tuple, list)):
        first = first[0]
    return first


def prepare_batch(
    seqs: list[str],
    tokenizer,
    *,
    prepend_bos: bool,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    lengths = np.array([len(seq) for seq in seqs], dtype=np.int64)
    max_len = int(lengths.max())
    pad_id = int(tokenizer.pad_id)
    bos = [int(tokenizer.eod_id)] if prepend_bos else []
    encoded = []
    for seq in seqs:
        token_ids = tokenizer.tokenize(seq)
        padding = [pad_id] * (max_len - len(seq))
        encoded.append(torch.tensor(bos + token_ids + padding, dtype=torch.long))
    return torch.stack(encoded, dim=0).to(device), lengths


def score_mean_logprobs(
    seqs: list[str],
    evo2_model,
    *,
    batch_size: int,
    prepend_bos: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = evo2_model.tokenizer
    scores: list[float] = []
    scored_positions: list[int] = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        input_ids, lengths = prepare_batch(batch, tokenizer, prepend_bos=prepend_bos, device=device)
        with torch.inference_mode():
            logits = model_forward(evo2_model, input_ids)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs[:, :-1, :],
                2,
                input_ids[:, 1:].unsqueeze(-1),
            ).squeeze(-1)
        arr = token_log_probs.float().detach().cpu().numpy()
        offset = 1 if prepend_bos else 0
        for row_idx, length in enumerate(lengths):
            n_scored = int(length - 1 + offset)
            if n_scored <= 0:
                raise ValueError("Evo2 likelihood scoring requires sequences of length at least 2.")
            scores.append(float(arr[row_idx, :n_scored].mean()))
            scored_positions.append(n_scored)
    return np.asarray(scores, dtype=np.float64), np.asarray(scored_positions, dtype=np.int64)


def score_rows(rows: pd.DataFrame, evo2_model, args: argparse.Namespace) -> pd.DataFrame:
    rows = rows.copy()
    seqs = rows["seq"].astype(str).str.upper().tolist()
    fwd_scores, n_pos = score_mean_logprobs(
        seqs,
        evo2_model,
        batch_size=args.batch_size,
        prepend_bos=args.prepend_bos,
        device=args.device,
    )
    rows["evo2_mean_logprob_fwd"] = fwd_scores
    rows["evo2_scored_positions"] = n_pos
    if args.reverse_complement:
        rc_seqs = [reverse_complement(seq) for seq in seqs]
        rc_scores, rc_n_pos = score_mean_logprobs(
            rc_seqs,
            evo2_model,
            batch_size=args.batch_size,
            prepend_bos=args.prepend_bos,
            device=args.device,
        )
        if not np.array_equal(n_pos, rc_n_pos):
            raise RuntimeError("Forward and reverse-complement scored-position counts differ.")
        rows["evo2_mean_logprob_rc"] = rc_scores
        rows["evo2_mean_logprob"] = (fwd_scores + rc_scores) * 0.5
        rows["evo2_orientation"] = "forward_rc_mean"
    else:
        rows["evo2_mean_logprob_rc"] = np.nan
        rows["evo2_mean_logprob"] = fwd_scores
        rows["evo2_orientation"] = "forward"
    rows["evo2_nll"] = -rows["evo2_mean_logprob"]
    rows["evo2_perplexity"] = np.exp(rows["evo2_nll"])
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def summarize(scored: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    split_test = scored[scored["set"] == "split_test_sample"]
    heldout_nll = float(split_test["evo2_nll"].mean()) if not split_test.empty else np.nan
    rows: list[dict] = []
    group_cols = ["source", "set", "target_c0"]
    for key, part in scored.groupby(group_cols, dropna=False, sort=True):
        source, set_name, target = key
        nll = part["evo2_nll"].to_numpy(dtype=np.float64)
        ppl = part["evo2_perplexity"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(nll, n_boot, seed)
        rows.append(
            {
                "source": source,
                "set": set_name,
                "target_c0": target,
                "n_samples": len(part),
                "seq_len": int(part["seq"].str.len().iloc[0]),
                "scored_positions": float(part["evo2_scored_positions"].mean()),
                "orientation": part["evo2_orientation"].iloc[0],
                "evo2_mean_logprob_mean": float(part["evo2_mean_logprob"].mean()),
                "evo2_nll_mean": float(nll.mean()),
                "evo2_nll_std": float(nll.std(ddof=0)),
                "evo2_nll_median": float(np.median(nll)),
                "evo2_nll_mean_ci95_low": ci_low,
                "evo2_nll_mean_ci95_high": ci_high,
                "evo2_perplexity_mean": float(ppl.mean()),
                "evo2_perplexity_from_mean_nll": float(np.exp(nll.mean())),
                "evo2_perplexity_median": float(np.median(ppl)),
                "evo2_nll_delta_vs_split_test": float(nll.mean() - heldout_nll)
                if np.isfinite(heldout_nll)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = discover_csvs(args.run_glob)
    if not csvs:
        raise FileNotFoundError(
            f"No sample_scores.csv files matched: {args.run_glob or [DEFAULT_RUN_GLOB]}"
        )
    targets = set(args.targets) if args.targets is not None else None

    rows = pd.concat(
        [
            baseline_rows(
                args.train_pt, args.split_pt, args.baseline_n, args.max_per_set, args.seed,
                split_names=(args.train_split, args.val_split, args.test_split),
            ),
            generated_rows(csvs, targets, args.max_per_set, args.seed),
        ],
        ignore_index=True,
    )
    if rows.empty:
        raise ValueError("No sequences to score.")

    evo2_model = load_evo2(args.model_name)
    scored = score_rows(rows, evo2_model, args)
    summary = summarize(scored, args.bootstrap, args.seed)

    scored_path = out_dir / "evo2_per_sequence.csv"
    summary_path = out_dir / "evo2_summary.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nwrote {scored_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
