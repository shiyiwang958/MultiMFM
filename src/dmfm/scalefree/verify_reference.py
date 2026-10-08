#!/usr/bin/env python
"""Check the recomputed GLASS-2048 reference against the published Fig 5 tensors.

The dMFM rerun recomputes the reference from scratch so the repo is self-contained. This
script proves the protocol was matched: it compares ``reference_gradients`` in our
``gradient_pairs.pt`` with the published run's, per conditioning state. Agreement to a
fraction of a percent (TF32 matmul noise alone is 0.2-0.4 %, see
``tests/dna/check_gradient_mc_reference.py``) means the states, seeds, calibration and
GLASS integrator all match.

CPU only. The published tensors live outside this repo, so pass their directory::

    python -m dmfm.scalefree.verify_reference \
        --published-root /n/holylabs/.../dirichlet-flow-matching/workdir \
        --published-run motif=rebuttal_gradient_accuracy_motif \
        --published-run gc=rebuttal_gradient_accuracy_gc_pairmetrics \
        --published-run conjunction=rebuttal_gradient_accuracy_conjunction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dmfm import paths
from dmfm.scalefree import LENGTHS, REWARDS, STUDENT_TAGS

DEFAULT_PUBLISHED_RUNS = {
    "gc": "rebuttal_gradient_accuracy_gc_pairmetrics",
    "motif": "rebuttal_gradient_accuracy_motif",
    "conjunction": "rebuttal_gradient_accuracy_conjunction",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", default=str(paths.OUTPUTS / "scale_free_mc_dmfm"))
    p.add_argument("--student_tag", default=STUDENT_TAGS["dmfm"])
    p.add_argument("--rewards", nargs="+", default=list(REWARDS))
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--published-root", required=True, help="Directory holding the published run directories.")
    p.add_argument(
        "--published-run",
        action="append",
        default=[],
        metavar="REWARD=DIRNAME",
        help=f"Override a run directory name (defaults {DEFAULT_PUBLISHED_RUNS}).",
    )
    p.add_argument("--out", default=None, help="Write the report as JSON here.")
    return p.parse_args(argv)


def compare(ours: torch.Tensor, theirs: torch.Tensor) -> dict[str, float]:
    if ours.shape != theirs.shape:
        raise ValueError(f"shape mismatch {tuple(ours.shape)} vs {tuple(theirs.shape)}")
    flat_ours = ours.reshape(ours.shape[0], -1).double()
    flat_theirs = theirs.reshape(theirs.shape[0], -1).double()
    rel = (flat_ours - flat_theirs).norm(dim=1) / flat_theirs.norm(dim=1).clamp_min(1e-30)
    cos = (flat_ours * flat_theirs).sum(dim=1) / (
        flat_ours.norm(dim=1) * flat_theirs.norm(dim=1)
    ).clamp_min(1e-30)
    return {
        "n_states": int(ours.shape[0]),
        "max_relative_l2_deviation": float(rel.max()),
        "mean_relative_l2_deviation": float(rel.mean()),
        "min_cosine": float(cos.min()),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    runs = dict(DEFAULT_PUBLISHED_RUNS)
    for spec in args.published_run:
        reward, name = spec.split("=", 1)
        runs[reward] = name
    root = Path(args.output_root) / args.student_tag
    published_root = Path(args.published_root)

    report: dict[str, dict] = {}
    worst = 0.0
    for reward in args.rewards:
        for length in args.lengths:
            ours_path = root / reward / f"L{length}" / "gradient_pairs.pt"
            theirs_path = published_root / runs[reward] / f"L{length}" / "gradient_pairs.pt"
            key = f"{reward}/L{length}"
            if not ours_path.is_file() or not theirs_path.is_file():
                report[key] = {"skipped": f"missing {'ours' if not ours_path.is_file() else 'published'}"}
                continue
            ours = torch.load(ours_path, map_location="cpu", weights_only=False)["reference_gradients"]
            theirs = torch.load(theirs_path, map_location="cpu", weights_only=False)["reference_gradients"]
            report[key] = compare(ours, theirs)
            worst = max(worst, report[key]["max_relative_l2_deviation"])
            print(
                f"{key}: max rel dev {report[key]['max_relative_l2_deviation']:.2%}, "
                f"mean {report[key]['mean_relative_l2_deviation']:.2%}, "
                f"min cosine {report[key]['min_cosine']:.6f}",
                flush=True,
            )
    report["worst_max_relative_l2_deviation"] = worst
    print(f"\nworst max relative deviation across compared cells: {worst:.2%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
