#!/usr/bin/env python
"""Aggregate the dMFM scale-free shards into ``results/dna/scale_free_mc/dmfm/``.

For every ``<output_root>/<student tag>/<reward>`` run directory whose four length
shards are present this

1. assembles ``run_metadata.json`` from the per-shard ``per_length/L*.json``;
2. runs :mod:`dmfm.experiments.rescore_gradient_pairs_scale_free` (the published
   Tables 14-15 rescore: ``--bootstrap 10000 --seed 20260802``) and
   :mod:`dmfm.experiments.summarize_gradient_accuracy_metrics`, both unchanged;
3. copies the small CSV/JSON outputs into
   ``results/dna/scale_free_mc/dmfm/<student tag>/<reward>/`` in the same schema as the
   published GLASS results next to it (``results/dna/scale_free_mc/<reward>/``);
4. writes ``results/dna/scale_free_mc/dmfm/summary.json`` and
   ``tables14_15_cells.csv`` with every Table 14 and Table 15 cell, new next to the
   published GLASS value.

Re-runnable: incomplete run directories are reported and skipped, complete ones are
rescored again from their stored gradient pairs (cheap, CPU only).

Usage::

    python -m dmfm.scalefree.collect                     # default output_root + results
    bash scripts/dna/scalefree/collect.sh                # thin wrapper
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from dmfm import paths
from dmfm.experiments import (
    rescore_gradient_pairs_scale_free,
    summarize_gradient_accuracy_metrics,
)
from dmfm.scalefree import LENGTHS, MC_VALUES, REWARDS, STUDENT_TAGS, TABLE14_MC

# The 2026-08-02 rescore that produced the published Tables 14-15.
RESCORE_BOOTSTRAP = 10_000
RESCORE_SEED = 20260802

PUBLISHED_GLASS_DIR = paths.RESULTS / "scale_free_mc"
RESULTS_DIR = PUBLISHED_GLASS_DIR / "dmfm"

# The metric columns of rescore_gradient_pairs_scale_free that Tables 14-15 print
# (verified cell for cell against the published tables).
REL_COL = "global_relative_l1"
COS_COL = "mean_cosine"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", default=str(paths.OUTPUTS / "scale_free_mc_dmfm"))
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument("--student_tags", nargs="+", default=[STUDENT_TAGS["dmfm"], STUDENT_TAGS["dmfm4"]])
    p.add_argument("--rewards", nargs="+", default=list(REWARDS))
    p.add_argument("--bootstrap", type=int, default=RESCORE_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=RESCORE_SEED)
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--n_probes", type=int, default=32, help="Used to check that probe-range parts tile each shard.")
    return p.parse_args(argv)


def merge_parts(run_dir: Path, length: int, n_probes: int = 32) -> bool:
    """Merge a complete set of probe-range parts into the usual shard files.

    Probe-range parts exist because a whole 32-state shard is hours long on CPU and
    serial_requeue is preemptible. A set is complete when the parts tile [0, n_probes)
    exactly. All parts must agree on the conditioning states (checked via the stored
    sha256), otherwise merging them would mix gradients taken at different states.
    """
    out_dir = run_dir / f"L{length}"
    parts_dir = out_dir / "parts"
    if (out_dir / "metadata.json").is_file() or not parts_dir.is_dir():
        return (out_dir / "metadata.json").is_file()
    metas = []
    for path in sorted(parts_dir.glob("probes_*.json")):
        with open(path) as handle:
            metas.append((path, json.load(handle)))
    if not metas:
        return False
    metas.sort(key=lambda item: item[1]["probe_start"])
    covered, digests = 0, set()
    for path, meta in metas:
        if int(meta["probe_start"]) != covered:
            print(f"  L{length}: parts do not tile [0, {n_probes}) (gap or overlap at {path.name}); not merging")
            return False
        covered = int(meta["probe_end"])
        digests.add(meta.get("probe_states_sha256"))
    if covered != n_probes:
        print(f"  L{length}: parts cover {covered}/{n_probes} states; not merging yet")
        return False
    if len(digests) != 1:
        raise RuntimeError(
            f"L{length}: probe-range parts disagree on the conditioning states "
            f"({len(digests)} distinct sha256). Refusing to merge; rerun them on one device "
            f"with a pinned --torch_threads."
        )

    frames, references, estimates = [], [], []
    for path, meta in metas:
        stem = path.with_suffix("").name
        frames.append(pd.read_csv(parts_dir / f"{stem}.csv"))
        pairs = torch.load(parts_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
        references.append(pairs["reference_gradients"])
        estimates.append(pairs["estimate_gradients"])
        mc_values = pairs["mc_values"]
    raw = pd.concat(frames, ignore_index=True).sort_values(["probe", "mc", "repeat"])
    raw.to_csv(out_dir / "gradient_errors.csv", index=False)
    torch.save(
        {
            "reference_gradients": torch.cat(references, dim=0),
            "estimate_gradients": torch.cat(estimates, dim=0),
            "mc_values": mc_values,
            "layout": "estimate_gradients[probe, mc_index, repeat, position, alphabet]",
        },
        out_dir / "gradient_pairs.pt",
    )
    merged = dict(metas[0][1])
    merged.update(
        {
            "probe_start": 0,
            "probe_end": n_probes,
            "merged_from_parts": [path.name for path, _ in metas],
            "elapsed_seconds": sum(float(meta.get("elapsed_seconds", 0.0)) for _, meta in metas),
            "compute_per_part": [meta.get("compute") for _, meta in metas],
        }
    )
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
    (run_dir / "per_length").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "metadata.json", run_dir / "per_length" / f"L{length}.json")
    print(f"  L{length}: merged {len(metas)} probe-range parts -> {out_dir}")
    return True


def assemble_run_metadata(run_dir: Path, lengths: list[int]) -> dict | None:
    """Build the ``run_metadata.json`` the published rescore/summarize scripts expect."""
    per_length = []
    for length in lengths:
        shard = run_dir / "per_length" / f"L{length}.json"
        if not shard.is_file():
            return None
        with open(shard) as handle:
            per_length.append(json.load(handle))
    first = per_length[0]
    metadata = {
        "args": {
            "seed": first["seed"],
            "bootstrap": 2000,
            "lengths": lengths,
            "reward": first["reward"]["name"],
            "mc_values": first["mc_values"],
            "n_probes": first["n_probes"],
            "n_repeats": first["n_repeats"],
            "reference_mc": first["reference_mc"],
            "nfe_value": first["nfe_value"],
            "glass_end_time": first["glass_end_time"],
            "t_eval": first["t_eval"],
            "student_kind": first["student_kind"],
            "dmfm_sampler": first["posterior_sampler"],
            "dmfm_steps": first["dmfm_steps"],
            "dmfm_end_time": first["dmfm_end_time"],
        },
        "per_length": per_length,
        "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in per_length),
        "note": "dMFM finite-N estimates vs the published GLASS-2048 reference "
        "(dmfm.scalefree.run_shard); every other setting matches the published Fig 5 run.",
    }
    with open(run_dir / "run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def rescore(run_dir: Path, bootstrap: int, seed: int) -> pd.DataFrame:
    rescore_gradient_pairs_scale_free.main(
        ["--input-dir", str(run_dir), "--bootstrap", str(bootstrap), "--seed", str(seed)]
    )
    summarize_gradient_accuracy_metrics.main(["--run_dir", str(run_dir)])
    return pd.read_csv(run_dir / "gradient_scale_free_summary_all_lengths.csv")


def publish(run_dir: Path, dest: Path, lengths: list[int]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "gradient_scale_free_summary_all_lengths.csv",
        "gradient_accuracy_relative_directional_summary.csv",
        "run_metadata.json",
    ):
        if (run_dir / name).is_file():
            shutil.copyfile(run_dir / name, dest / name)
    for length in lengths:
        length_dest = dest / f"L{length}"
        length_dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "gradient_scale_free_per_state.csv",
            "gradient_scale_free_summary.csv",
            "gradient_error_summary.csv",
            "metadata.json",
        ):
            src = run_dir / f"L{length}" / name
            if src.is_file():
                shutil.copyfile(src, length_dest / name)


def cells_from_summary(summary: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in summary.iterrows():
        length = str(int(row["length"]))
        mc = str(int(row["mc"]))
        out.setdefault(length, {})[mc] = {
            "relative_l1": float(row[REL_COL]),
            "cosine": float(row[COS_COL]),
            "relative_l1_ci95": [float(row["relative_l1_ci95_low"]), float(row["relative_l1_ci95_high"])],
            "cosine_ci95": [float(row["cosine_ci95_low"]), float(row["cosine_ci95_high"])],
            "mean_state_relative_l1": float(row["mean_state_relative_l1"]),
            "n_states": int(row["n_states"]),
            "n_repeats": int(row["n_repeats"]),
        }
    return out


def published_glass_cells(rewards: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for reward in rewards:
        path = PUBLISHED_GLASS_DIR / reward / "gradient_scale_free_summary_all_lengths.csv"
        if path.is_file():
            out[reward] = cells_from_summary(pd.read_csv(path))
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(length) for length in args.lengths]

    published = published_glass_cells(list(args.rewards))
    collected: dict[str, dict] = {}
    students: dict[str, dict] = {}
    missing: list[str] = []
    flat_rows: list[dict] = []

    for tag in args.student_tags:
        for reward in args.rewards:
            run_dir = output_root / tag / reward
            if run_dir.is_dir():
                for length in lengths:
                    merge_parts(run_dir, length, n_probes=args.n_probes)
            metadata = assemble_run_metadata(run_dir, lengths) if run_dir.is_dir() else None
            if metadata is None:
                have = sorted(int(p.stem[1:]) for p in (run_dir / "per_length").glob("L*.json")) if run_dir.is_dir() else []
                missing.append(f"{tag}/{reward} (have L{have})")
                continue
            print(f"== rescore {run_dir}", flush=True)
            summary = rescore(run_dir, args.bootstrap, args.seed)
            publish(run_dir, results_dir / tag / reward, lengths)
            collected.setdefault(tag, {})[reward] = cells_from_summary(summary)
            students.setdefault(tag, {})
            for item in metadata["per_length"]:
                students[tag].setdefault("checkpoints", {})[f"L{item['length']}"] = item["dmfm_checkpoint"]
            students[tag].update(
                {
                    "student_kind": metadata["args"]["student_kind"],
                    "posterior_sampler": metadata["args"]["dmfm_sampler"],
                    "steps": metadata["args"]["dmfm_steps"],
                    "end_time": metadata["args"]["dmfm_end_time"],
                }
            )

    for tag, per_reward in collected.items():
        for reward, per_length in per_reward.items():
            for length in lengths:
                for mc in MC_VALUES:
                    cell = per_length.get(str(length), {}).get(str(mc))
                    if cell is None:
                        continue
                    old = published.get(reward, {}).get(str(length), {}).get(str(mc), {})
                    flat_rows.append(
                        {
                            "student": tag,
                            "reward": reward,
                            "mc": mc,
                            "length": length,
                            "in_table14": mc in TABLE14_MC,
                            "dmfm_relative_l1": round(cell["relative_l1"], 4),
                            "dmfm_cosine": round(cell["cosine"], 4),
                            "published_glass_relative_l1": round(old["relative_l1"], 4) if old else None,
                            "published_glass_cosine": round(old["cosine"], 4) if old else None,
                            # Both sides pool the same 32 conditioning states (identical seed
                            # arithmetic and make_probe_states; see protocol_check.json), so the
                            # comparison is state-matched. These columns make that auditable
                            # instead of assumed: if they are not both 32, the row mixes state sets.
                            "dmfm_n_states": cell["n_states"],
                            "dmfm_n_repeats": cell["n_repeats"],
                            "published_glass_n_states": old.get("n_states") if old else None,
                            "state_matched": bool(old) and cell["n_states"] == old.get("n_states"),
                        }
                    )
    if flat_rows:
        frame = pd.DataFrame(flat_rows).sort_values(["student", "reward", "mc", "length"])
        frame.to_csv(results_dir / "tables14_15_cells.csv", index=False)

    summary_json = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "what_this_is": (
            "Fig 5 / Tables 14-15 rerun with finite-N dMFM value-gradient estimates against the "
            "published independent GLASS-2048 reference. The published runs used the GLASS "
            "posterior on the base DFM on both sides, so they measured GLASS Monte Carlo error "
            "only; the caption and App F.2 describe this rerun instead."
        ),
        "metric_columns": {"relative_l1": REL_COL, "cosine": COS_COL},
        "protocol": {
            "rewards": list(args.rewards),
            "lengths": lengths,
            "mc_values": list(MC_VALUES),
            "table14_mc": list(TABLE14_MC),
            "n_states": 32,
            "n_noise_pools_per_state": 8,
            "t_eval": 0.5,
            "conditioning_states": "base flow from Gaussian noise to t=0.5, 32 NFE (published construction)",
            "reference": "GLASS posterior on the base DFM, 8 Euler steps to t=1.0, 2048 terminal-noise samples",
            "seed": 20260726,
            "rescore": {"script": "dmfm.experiments.rescore_gradient_pairs_scale_free",
                        "bootstrap": args.bootstrap, "seed": args.seed},
        },
        "students": students,
        "dmfm": collected,
        "published_glass": published,
        "incomplete": missing,
    }
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary_json, handle, indent=2, sort_keys=True)

    print(f"\nWrote {results_dir / 'summary.json'}")
    if flat_rows:
        print(f"Wrote {results_dir / 'tables14_15_cells.csv'} ({len(flat_rows)} cells)")
        table14 = pd.DataFrame(flat_rows)
        table14 = table14[table14["in_table14"]]
        for tag, group in table14.groupby("student"):
            print(f"\n-- Table 14, {tag}: published GLASS  ->  new dMFM  (relative L1 / cosine)")
            print(f"{'reward':12s} {'N':>4s} " + "".join(f"{f'{L} bp':>26s}" for L in lengths))
            for (reward, mc), rows_ in group.groupby(["reward", "mc"]):
                line = f"{reward:12s} {mc:4d} "
                for length in lengths:
                    cell = rows_[rows_.length == length]
                    if cell.empty:
                        line += f"{'-':>26s}"
                        continue
                    row = cell.iloc[0]
                    old = (
                        f"{row.published_glass_relative_l1:.2f}/{row.published_glass_cosine:.2f}"
                        if pd.notna(row.published_glass_relative_l1)
                        else "n/a"
                    )
                    line += f"{old + ' -> ' + f'{row.dmfm_relative_l1:.2f}/{row.dmfm_cosine:.2f}':>26s}"
                print(line)
            n8 = pd.DataFrame(flat_rows)
            n8 = n8[(n8.student == tag) & (n8.mc == 8)]
            if not n8.empty:
                print(
                    f"   N=8 across all rewards and lengths: relative L1 "
                    f"{n8.dmfm_relative_l1.min():.2f}-{n8.dmfm_relative_l1.max():.2f}, cosine "
                    f"{n8.dmfm_cosine.min():.2f}-{n8.dmfm_cosine.max():.2f} "
                    f"(paper's GLASS-only sentence: 1.40-2.55 and 0.18-0.32)"
                )
            trend = pd.DataFrame(flat_rows)
            trend = trend[(trend.student == tag) & (trend.mc.isin([8, 128]))]
            print("   length trend (does the error grow from 50 to 400 bp?):")
            for (reward, mc), rows_ in trend.groupby(["reward", "mc"]):
                by_length = rows_.sort_values("length")
                series = " ".join(f"{int(r.length)}:{r.dmfm_relative_l1:.2f}" for r in by_length.itertuples())
                first, last = by_length.dmfm_relative_l1.iloc[0], by_length.dmfm_relative_l1.iloc[-1]
                ratio = f"{int(by_length.length.iloc[-1])}/{int(by_length.length.iloc[0])}"
                print(f"     {reward:12s} N={mc:3d}  {series}   {ratio} = {last / max(first, 1e-12):.2f}")
    if missing:
        print("\nIncomplete (skipped):")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
