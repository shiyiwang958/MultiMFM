#!/usr/bin/env python3
"""Build parent-disjoint yeast datasets for diagonal DFM experiments.

The raw assay rows contain fixed 25-bp flanks around a central 50-bp yeast
window. Consecutive central windows within a parent region overlap by 43 bp,
which allows reconstruction of 576 contiguous 1,044-bp parent sequences.
Splitting is performed on those parents before extracting windows at any length.

Ported from dirichlet-flow-matching/data/prepare_yeast_parent_splits.py (Table 10 split).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
WINDOW_LENGTH = 50
STRIDE = 7
PARENT_LENGTH = 1044


from dmfm import paths


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The raw loop-seq table (Basu et al. 2021) is not redistributed here; see
    # docs/provenance/dna/README.md for how to re-download it and check it
    # against the sha256 stored in parent_split_seed0.json.
    parser.add_argument("--input", default=str(paths.DATA / "raw" / "yeast_sequences.txt"))
    parser.add_argument("--out_dir", default=str(paths.DATA / "yeast_parent_disjoint"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 400])
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_central_windows(path: Path) -> list[tuple[str, float]]:
    windows: list[tuple[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        for row_number, row in enumerate(reader, start=2):
            row = {(key or "").strip(): value for key, value in row.items()}
            sequence = (row.get("Sequence") or "").strip().upper()
            try:
                c0 = float(row["C0"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing finite C0 value at TSV row {row_number}") from exc
            if len(sequence) != 100 or any(base not in DNA_TO_INT for base in sequence):
                raise ValueError(f"Invalid 100-bp DNA sequence at TSV row {row_number}")
            if not math.isfinite(c0):
                raise ValueError(f"Missing finite C0 value at TSV row {row_number}")
            windows.append((sequence[25:75], c0))
    if not windows:
        raise ValueError(f"No valid rows in {path}")
    return windows


def reconstruct_parents(windows: list[tuple[str, float]]) -> tuple[list[str], list[list[float]]]:
    parents_as_windows: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    for window in windows:
        if not current or current[-1][0][STRIDE:] == window[0][:-STRIDE]:
            current.append(window)
        else:
            parents_as_windows.append(current)
            current = [window]
    if current:
        parents_as_windows.append(current)

    parents: list[str] = []
    parent_c0: list[list[float]] = []
    for parent_id, group in enumerate(parents_as_windows):
        if len(group) != 143:
            raise ValueError(f"Parent {parent_id} has {len(group)} windows; expected 143")
        parent = group[0][0] + "".join(window[0][-STRIDE:] for window in group[1:])
        if len(parent) != PARENT_LENGTH:
            raise ValueError(f"Parent {parent_id} has length {len(parent)}; expected {PARENT_LENGTH}")
        for start, (window, _) in zip(range(0, PARENT_LENGTH - WINDOW_LENGTH + 1, STRIDE), group):
            if parent[start : start + WINDOW_LENGTH] != window:
                raise ValueError(f"Parent {parent_id} does not reproduce its source windows")
        parents.append(parent)
        parent_c0.append([c0 for _, c0 in group])

    if len(parents) != 576:
        raise ValueError(f"Recovered {len(parents)} parents; expected 576")
    return parents, parent_c0


def parent_split(n_parents: int, seed: int) -> dict[str, list[int]]:
    if n_parents != 576:
        raise ValueError(f"Expected 576 parents, got {n_parents}")
    perm = torch.randperm(n_parents, generator=torch.Generator().manual_seed(seed)).tolist()
    train, val, test = perm[:460], perm[460:518], perm[518:]
    split = {
        "a_train": train[:230],
        "b_train": train[230:],
        "a_val": val[:29],
        "b_val": val[29:],
        "test": test,
    }
    expected = {"a_train": 230, "b_train": 230, "a_val": 29, "b_val": 29, "test": 58}
    if {name: len(ids) for name, ids in split.items()} != expected:
        raise AssertionError("Unexpected parent split sizes")
    if len(set().union(*(set(ids) for ids in split.values()))) != n_parents:
        raise AssertionError("Parent split does not form a disjoint partition")
    return split


def encode(sequence: str) -> list[int]:
    return [DNA_TO_INT[base] for base in sequence]


def build_length_dataset(
    parents: list[str],
    parent_c0: list[list[float]],
    split: dict[str, list[int]],
    length: int,
    source: Path,
    seed: int,
    out_dir: Path,
) -> dict[str, int]:
    if length < WINDOW_LENGTH or length > PARENT_LENGTH:
        raise ValueError(f"Length must lie in [{WINDOW_LENGTH}, {PARENT_LENGTH}], got {length}")
    starts = list(range(0, PARENT_LENGTH - length + 1, STRIDE))
    sequences: list[list[int]] = []
    parent_ids: list[int] = []
    window_starts: list[int] = []
    c0_values: list[float] = []
    parent_to_indices: list[list[int]] = [[] for _ in parents]
    for parent_id, parent in enumerate(parents):
        for window_number, start in enumerate(starts):
            index = len(sequences)
            sequences.append(encode(parent[start : start + length]))
            parent_ids.append(parent_id)
            window_starts.append(start)
            parent_to_indices[parent_id].append(index)
            if length == WINDOW_LENGTH:
                c0_values.append(parent_c0[parent_id][window_number])

    seqs = torch.tensor(sequences, dtype=torch.long)
    parent_tensor = torch.tensor(parent_ids, dtype=torch.long)
    start_tensor = torch.tensor(window_starts, dtype=torch.long)
    split_idx = {
        f"{name}_idx": torch.tensor(
            [index for parent_id in ids for index in parent_to_indices[parent_id]], dtype=torch.long
        )
        for name, ids in split.items()
    }
    expected_windows = (PARENT_LENGTH - length) // STRIDE + 1
    if seqs.shape != (576 * expected_windows, length):
        raise AssertionError(f"Unexpected tensor shape {tuple(seqs.shape)} for length {length}")
    for name, ids in split.items():
        if torch.unique(parent_tensor[split_idx[f"{name}_idx"]]).numel() != len(ids):
            raise AssertionError(f"Split {name} does not contain the expected number of distinct parents")

    data_path = out_dir / f"yeast_parent_L{length}.pt"
    split_path = out_dir / f"yeast_parent_L{length}_split_seed{seed}.pt"
    payload = {
        "seqs": seqs,
        "parent_id": parent_tensor,
        "window_start": start_tensor,
        "seq_len": length,
        "stride": STRIDE,
        "parent_length": PARENT_LENGTH,
        "source": str(source),
    }
    if length == WINDOW_LENGTH:
        payload["c0"] = torch.tensor(c0_values, dtype=torch.float32)
    torch.save(payload, data_path)
    torch.save(
        {
            **split_idx,
            "seed": seed,
            "n_total": len(seqs),
            "seq_len": length,
            "windows_per_parent": expected_windows,
            "source": str(data_path),
            "parent_ids": {name: torch.tensor(ids, dtype=torch.long) for name, ids in split.items()},
        },
        split_path,
    )
    return {
        "n_total": int(len(seqs)),
        "windows_per_parent": expected_windows,
        **{name: int(len(index)) for name, index in split_idx.items()},
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    source = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted(set(args.lengths))
    windows = load_central_windows(source)
    parents, parent_c0 = reconstruct_parents(windows)
    split = parent_split(len(parents), args.seed)
    summary = {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "seed": args.seed,
        "n_source_windows": len(windows),
        "n_parents": len(parents),
        "parent_length": PARENT_LENGTH,
        "stride": STRIDE,
        "parent_split": split,
        "lengths": {},
    }
    for length in lengths:
        summary["lengths"][str(length)] = build_length_dataset(
            parents, parent_c0, split, length, source, args.seed, out_dir
        )
    manifest_path = out_dir / f"parent_split_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
