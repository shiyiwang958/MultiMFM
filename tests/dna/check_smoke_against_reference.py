#!/usr/bin/env python
"""Check a small C0-guidance rerun against a stored reference run.

The unguided branch of ``dmfm.experiments.sample_c0_guidance`` is deterministic:
each sample's initial noise comes from a CPU generator seeded with
``seed + sample_idx``, and no cross-sample interaction happens, so the unguided
sequences and guide scores must match the reference exactly (this is how the
2026-09-25 reconstruction was validated against the surviving July run). The
guided branch uses a nondeterministic CUDA backward and only agrees
statistically.

    python tests/dna/check_smoke_against_reference.py \
        --run_dir outputs/dna/smoke_.../c0_target1p0 \
        --reference results/dna/c0_guidance/n1000/target_1p0

Exits non-zero if the unguided sequences or scores disagree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Directory with sample_scores.csv from the rerun.")
    p.add_argument("--reference", required=True, help="Reference run directory (same seed and target).")
    p.add_argument("--atol", type=float, default=1e-4, help="Tolerance on the unguided guide score.")
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    ref = Path(args.reference)
    new = pd.read_csv(run / "sample_scores.csv").set_index("sample_idx").sort_index()
    ref_csv = ref / "sample_scores.csv"
    if not ref_csv.exists():
        print(f"SKIP: no reference at {ref_csv}")
        return 0
    old = pd.read_csv(ref_csv).set_index("sample_idx").sort_index()
    shared = new.index.intersection(old.index)
    if len(shared) == 0:
        print("SKIP: reference and rerun share no sample_idx")
        return 0
    a, b = new.loc[shared], old.loc[shared]

    bad = 0
    seq_match = int((a["seq_unguided"].values == b["seq_unguided"].values).sum())
    print(f"unguided sequences identical: {seq_match}/{len(shared)}")
    if seq_match != len(shared):
        bad += 1
    d = np.abs(a["unguided"].to_numpy() - b["unguided"].to_numpy())
    print(f"unguided guide score  max|diff| = {d.max():.3e} (tol {args.atol})")
    if d.max() > args.atol:
        bad += 1

    for name, oracle_col in (("unguided", "oracle_unguided"), ("guided", "oracle_guided")):
        f_new = run / "oracle" / "oracle_per_sample.csv"
        f_old = ref / "oracle" / "oracle_per_sample.csv"
        if not (f_new.exists() and f_old.exists()):
            continue
        on = pd.read_csv(f_new).set_index("sample_idx").sort_index().loc[shared]
        oo = pd.read_csv(f_old).set_index("sample_idx").sort_index().loc[shared]
        dd = np.abs(on[oracle_col].to_numpy() - oo[oracle_col].to_numpy())
        tag = "must match" if name == "unguided" else "nondeterministic, informational"
        print(f"oracle {name:8s} max|diff| = {dd.max():.3e} ({tag})")
        if name == "unguided" and dd.max() > 1e-3:
            bad += 1

    gd = a["guided"].to_numpy() - b["guided"].to_numpy()
    print(
        f"guided guide score: rerun mean {a['guided'].mean():.4f} vs reference {b['guided'].mean():.4f} "
        f"(n={len(shared)}, mean diff {gd.mean():+.4f}) -- guided sampling is not bit-reproducible"
    )
    print("UNGUIDED_CHECK_OK" if not bad else f"UNGUIDED_CHECK_FAILED ({bad} problem(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
