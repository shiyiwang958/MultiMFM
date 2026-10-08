"""Sequence-diversity metrics for the yeast C0 sets (vendored metric functions).

These functions are copied verbatim from DNA-MFM@187fe7b
``scripts/analyze_yeast_c0_diversity.py`` (only the ``torch.load`` call goes
through ``dmfm.utils.torch_io``). The Tables 4-5 driver
``dmfm.experiments.analyze_c0_diversity`` calls them (``nearest_reference_distances``,
``pairwise_generated_distances``, ``nearest_generated_distances``,
``summarize_one``, ``summarize_reference_baseline``, ``seqs_to_tokens``,
``load_train_tokens``, ``run_metadata``) exactly as the original 2026-09-25 run
did through its ``vendor/`` directory; rerunning the driver on the n=1000 sets
reproduces that run's CSVs exactly (max |diff| = 0).

Not ported: the original file's own command-line driver (``parse_args``,
``main``, ``plot_summary``, ``discover_csvs``, ``summarize_train_baseline``,
``teacher_split_indices``, ``dmfm_split_indices``, ``load_args_payload``). It was
the older pre-parent-split analysis; its defaults pointed at purged split65k data
and models, and no paper number was produced by it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from dmfm.utils.torch_io import torch_load


DNA_TO_INT = np.full(256, -1, dtype=np.int16)
for i, ch in enumerate("ACGT"):
    DNA_TO_INT[ord(ch)] = i


def load_train_tokens(path: str | Path) -> np.ndarray:
    payload = torch_load(path, map_location="cpu")
    seqs = payload["seqs"] if isinstance(payload, dict) and "seqs" in payload else payload
    if not isinstance(seqs, torch.Tensor):
        raise TypeError(f"Expected tensor or dict with 'seqs' in {path}, got {type(seqs)}")
    arr = seqs.detach().cpu().numpy().astype(np.int16, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected train tokens [N,L], got {arr.shape}")
    return arr


def seqs_to_tokens(seqs: Iterable[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    length: int | None = None
    for seq in seqs:
        s = str(seq).strip().upper()
        if length is None:
            length = len(s)
        elif len(s) != length:
            raise ValueError(f"Mixed sequence lengths: expected {length}, got {len(s)} for {s[:20]}...")
        encoded = DNA_TO_INT[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        if (encoded < 0).any():
            bad = sorted(set(ch for ch in s if ch not in "ACGT"))
            raise ValueError(f"Non-ACGT characters {bad} in sequence {s[:40]}...")
        rows.append(encoded)
    if not rows:
        return np.zeros((0, 0), dtype=np.int16)
    return np.stack(rows).astype(np.int16, copy=False)


def hamming_dist_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Returns absolute Hamming distances with shape [len(a), len(b)].
    return (a[:, None, :] != b[None, :, :]).sum(axis=2).astype(np.int16, copy=False)


def nearest_reference_distances(
    gen: np.ndarray,
    reference: np.ndarray,
    *,
    k_nearest: int = 5,
    gen_chunk: int = 64,
    ref_chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape[0] == 0:
        return (
            np.full(gen.shape[0], np.nan, dtype=np.float64),
            np.full(gen.shape[0], np.nan, dtype=np.float64),
        )
    if gen.shape[1] != reference.shape[1]:
        raise ValueError(f"Length mismatch: generated L={gen.shape[1]}, reference L={reference.shape[1]}")
    k = max(1, min(int(k_nearest), reference.shape[0]))
    mins = np.empty(gen.shape[0], dtype=np.int16)
    mean_topk = np.empty(gen.shape[0], dtype=np.float64)

    for i0 in range(0, gen.shape[0], gen_chunk):
        i1 = min(gen.shape[0], i0 + gen_chunk)
        best = np.full((i1 - i0, k), gen.shape[1] + 1, dtype=np.int16)
        for j0 in range(0, reference.shape[0], max(1, ref_chunk)):
            j1 = min(reference.shape[0], j0 + max(1, ref_chunk))
            d = hamming_dist_chunk(gen[i0:i1], reference[j0:j1])
            cand = np.concatenate([best, d], axis=1)
            best = np.partition(cand, kth=k - 1, axis=1)[:, :k]
        best.sort(axis=1)
        mins[i0:i1] = best[:, 0]
        mean_topk[i0:i1] = best.mean(axis=1)
    return mins, mean_topk


def pairwise_generated_distances(
    gen: np.ndarray,
    *,
    chunk: int = 128,
    max_n: int = 3000,
    seed: int = 0,
) -> np.ndarray:
    n = gen.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.int16)
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
        gen = gen[idx]
        n = gen.shape[0]

    parts: list[np.ndarray] = []
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        for j0 in range(i0, n, chunk):
            j1 = min(n, j0 + chunk)
            d = hamming_dist_chunk(gen[i0:i1], gen[j0:j1])
            if i0 == j0:
                iu = np.triu_indices(i1 - i0, k=1)
                parts.append(d[iu])
            else:
                parts.append(d.reshape(-1))
    return np.concatenate(parts).astype(np.int16, copy=False)


def nearest_generated_distances(gen: np.ndarray, *, chunk: int = 128) -> np.ndarray:
    n = gen.shape[0]
    if n < 2:
        return np.zeros((n,), dtype=np.int16)
    out = np.full(n, gen.shape[1] + 1, dtype=np.int16)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        for j0 in range(0, n, chunk):
            j1 = min(n, j0 + chunk)
            d = hamming_dist_chunk(gen[i0:i1], gen[j0:j1])
            if i0 == j0:
                np.fill_diagonal(d, gen.shape[1] + 1)
            out[i0:i1] = np.minimum(out[i0:i1], d.min(axis=1))
    return out


def run_metadata(csv_path: str | Path) -> dict:
    args_path = Path(csv_path).with_name("args.json")
    if not args_path.exists():
        return {}
    with open(args_path) as f:
        return json.load(f)


def summarize_one(
    df: pd.DataFrame,
    seq_col: str,
    references: dict[str, np.ndarray],
    meta: dict,
    csv_path: str | Path,
    *,
    k_nearest_train: int,
    gen_chunk: int,
    train_ref_chunk: int,
    pair_chunk: int,
    max_pairwise_n: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    gen = seqs_to_tokens(df[seq_col].tolist())
    length = gen.shape[1]

    nearest_by_ref: dict[str, np.ndarray] = {}
    mean_k_by_ref: dict[str, np.ndarray] = {}
    for ref_name, ref in references.items():
        nearest, mean_k = nearest_reference_distances(
            gen,
            ref,
            k_nearest=k_nearest_train,
            gen_chunk=gen_chunk,
            ref_chunk=train_ref_chunk,
        )
        nearest_by_ref[ref_name] = nearest
        mean_k_by_ref[ref_name] = mean_k

    nearest_full = nearest_by_ref["full"]
    mean_k_full = mean_k_by_ref["full"]
    pairwise = pairwise_generated_distances(
        gen,
        chunk=pair_chunk,
        max_n=max_pairwise_n,
        seed=seed,
    )
    nearest_gen = nearest_generated_distances(gen, chunk=pair_chunk)

    seqs = df[seq_col].astype(str)
    exact_unique = int(seqs.nunique())

    target = meta.get("target_c0")
    target_second = meta.get("target_c0_second")
    score_col = "guided" if seq_col == "seq_guided" and "guided" in df else "unguided"

    summary = {
        "run_dir": str(Path(csv_path).parent),
        "set": "guided" if seq_col == "seq_guided" else "unguided",
        "target_c0": target,
        "target_c0_second": target_second,
        "guidance_frac": meta.get("guidance_frac"),
        "reward_sigma": meta.get("reward_sigma"),
        "reward_scale": meta.get("reward_scale"),
        "guide_t_start": meta.get("guide_t_start"),
        "guide_t_end": meta.get("guide_t_end"),
        "n_samples": int(len(df)),
        "seq_len": int(length),
        "score_mean": float(df[score_col].mean()) if score_col in df else np.nan,
        "score_std": float(df[score_col].std(ddof=0)) if score_col in df else np.nan,
        "exact_unique": exact_unique,
        "exact_unique_frac": float(exact_unique / max(len(df), 1)),
        "duplicate_frac": float(1.0 - exact_unique / max(len(df), 1)),
        "novel_exact_frac": float((nearest_full > 0).mean()),
        "nearest_train_hamming_mean": float(nearest_full.mean()),
        "nearest_train_hamming_median": float(np.median(nearest_full)),
        "nearest_train_hamming_p05": float(np.quantile(nearest_full, 0.05)),
        "nearest_train_hamming_p95": float(np.quantile(nearest_full, 0.95)),
        "nearest_train_frac_mean": float(nearest_full.mean() / length),
        "nearest_train_frac_median": float(np.median(nearest_full) / length),
        f"mean_{k_nearest_train}nn_train_frac_mean": float(mean_k_full.mean() / length),
        "nearest_gen_hamming_mean": float(nearest_gen.mean()),
        "nearest_gen_hamming_median": float(np.median(nearest_gen)),
        "nearest_gen_frac_mean": float(nearest_gen.mean() / length),
        "pairwise_gen_hamming_mean": float(pairwise.mean()) if len(pairwise) else np.nan,
        "pairwise_gen_hamming_median": float(np.median(pairwise)) if len(pairwise) else np.nan,
        "pairwise_gen_hamming_p05": float(np.quantile(pairwise, 0.05)) if len(pairwise) else np.nan,
        "pairwise_gen_frac_mean": float(pairwise.mean() / length) if len(pairwise) else np.nan,
        "pairwise_gen_frac_median": float(np.median(pairwise) / length) if len(pairwise) else np.nan,
    }

    for ref_name, nearest in nearest_by_ref.items():
        mean_k = mean_k_by_ref[ref_name]
        summary[f"nearest_{ref_name}_hamming_mean"] = float(np.nanmean(nearest))
        summary[f"nearest_{ref_name}_frac_mean"] = float(np.nanmean(nearest) / length)
        summary[f"nearest_{ref_name}_frac_median"] = float(np.nanmedian(nearest) / length)
        summary[f"novel_exact_vs_{ref_name}_frac"] = float(np.nanmean(nearest > 0))
        summary[f"mean_{k_nearest_train}nn_{ref_name}_frac_mean"] = float(np.nanmean(mean_k) / length)

    per_sample = pd.DataFrame(
        {
            "run_dir": str(Path(csv_path).parent),
            "set": summary["set"],
            "target_c0": target,
            "target_c0_second": target_second,
            "guidance_frac": meta.get("guidance_frac"),
            "sample_idx": df["sample_idx"].to_numpy() if "sample_idx" in df else np.arange(len(df)),
            "score": df[score_col].to_numpy() if score_col in df else np.nan,
            "seq": seqs.to_numpy(),
            "nearest_train_hamming": nearest_full,
            "nearest_train_frac": nearest_full / length,
            f"mean_{k_nearest_train}nn_train_frac": mean_k_full / length,
            "nearest_gen_hamming": nearest_gen,
            "nearest_gen_frac": nearest_gen / length,
        }
    )
    for ref_name, nearest in nearest_by_ref.items():
        per_sample[f"nearest_{ref_name}_hamming"] = nearest
        per_sample[f"nearest_{ref_name}_frac"] = nearest / length
    return summary, per_sample


def summarize_reference_baseline(
    reference: np.ndarray,
    *,
    set_name: str,
    n: int,
    k_nearest_train: int,
    pair_chunk: int,
    max_pairwise_n: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = min(int(n), reference.shape[0])
    idx = np.sort(rng.choice(reference.shape[0], size=n, replace=False))
    sample = reference[idx]
    length = sample.shape[1]
    pairwise = pairwise_generated_distances(
        sample,
        chunk=pair_chunk,
        max_n=max_pairwise_n,
        seed=seed,
    )
    nearest = nearest_generated_distances(sample, chunk=pair_chunk)
    exact_unique = np.unique(sample, axis=0).shape[0]
    return {
        "run_dir": "reference_data",
        "set": set_name,
        "target_c0": np.nan,
        "target_c0_second": None,
        "guidance_frac": np.nan,
        "reward_sigma": np.nan,
        "reward_scale": np.nan,
        "guide_t_start": np.nan,
        "guide_t_end": np.nan,
        "n_samples": int(n),
        "seq_len": int(length),
        "score_mean": np.nan,
        "score_std": np.nan,
        "exact_unique": int(exact_unique),
        "exact_unique_frac": float(exact_unique / max(n, 1)),
        "duplicate_frac": float(1.0 - exact_unique / max(n, 1)),
        "novel_exact_frac": np.nan,
        "nearest_train_hamming_mean": np.nan,
        "nearest_train_hamming_median": np.nan,
        "nearest_train_hamming_p05": np.nan,
        "nearest_train_hamming_p95": np.nan,
        "nearest_train_frac_mean": np.nan,
        "nearest_train_frac_median": np.nan,
        f"mean_{k_nearest_train}nn_train_frac_mean": np.nan,
        "nearest_gen_hamming_mean": float(nearest.mean()),
        "nearest_gen_hamming_median": float(np.median(nearest)),
        "nearest_gen_frac_mean": float(nearest.mean() / length),
        "pairwise_gen_hamming_mean": float(pairwise.mean()) if len(pairwise) else np.nan,
        "pairwise_gen_hamming_median": float(np.median(pairwise)) if len(pairwise) else np.nan,
        "pairwise_gen_hamming_p05": float(np.quantile(pairwise, 0.05)) if len(pairwise) else np.nan,
        "pairwise_gen_frac_mean": float(pairwise.mean() / length) if len(pairwise) else np.nan,
        "pairwise_gen_frac_median": float(np.median(pairwise) / length) if len(pairwise) else np.nan,
    }


