"""Length-conditioned QM9 property-target guidance smoke experiment.

For each generated sample, this script first samples a molecule length from the
TABASCO generator's learned length prior. It then samples an alpha target from a
cached QM9 alpha histogram conditioned on that heavy-atom length, runs unguided
and GLASS-guided sampling from the same initial noise, and reports per-sample
absolute errors plus aggregate MAE and median absolute error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "qm9"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "qm9" / "datasets")
)
os.environ.setdefault(
    "HF_HUB_CACHE", str(REPO_ROOT / "data" / "qm9" / "hub")
)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "cache" / "xdg"))

from datasets import load_dataset  # noqa: E402

from multimfm.glass_guidance import (  # noqa: E402
    TFG_PROPERTIES,
    build_posebusters,
    guided_sample,
    load_mfm_student,
    load_flow_model,
    pb_summary_for_state,
    predicted_property,
    seed_all,
    write_csv,
)
from multimfm.qm9_data import TFG_SPLITS, load_tfg_regressor, tfg_partition_indices  # noqa: E402
from tabasco.sample.glass import require_linear_dfm_atoms  # noqa: E402
from multimfm.train_base_flow import choose_device  # noqa: E402


DEFAULT_HIST_CACHE = Path("cache/qm9_property_histograms/alpha_train_flow_bins100.json")


def default_hist_cache(property_name: str, partition: str, bins: int) -> Path:
    return Path("cache/qm9_property_histograms") / (
        f"{property_name}_{partition}_bins{bins}.json"
    )


def resolve_hist_cache(args: argparse.Namespace) -> None:
    if Path(args.hist_cache) != DEFAULT_HIST_CACHE:
        return
    if (
        args.property_name == "alpha"
        and args.partition == "train_flow"
        and args.hist_bins == 100
    ):
        return
    args.hist_cache = default_hist_cache(
        args.property_name,
        args.partition,
        args.hist_bins,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/"
            "checkpoints/model_step_100000.pt"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-seed", type=int, default=101)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.3)
    parser.add_argument("--value-samples", type=int, default=4)
    parser.add_argument(
        "--value-batch-size",
        type=int,
        default=0,
        help=(
            "If positive, split differentiable value-gradient evaluation into "
            "chunks of this many conditioning samples to reduce peak GPU memory."
        ),
    )
    parser.add_argument("--value-glass-steps", type=int, default=4)
    parser.add_argument(
        "--value-sampler",
        choices=["glass", "mfm"],
        default="glass",
        help="Posterior sampler differentiated through to estimate V_t.",
    )
    parser.add_argument(
        "--mfm-checkpoint",
        type=Path,
        default=None,
        help="TABASCO MFM student checkpoint used when --value-sampler=mfm.",
    )
    parser.add_argument(
        "--mfm-key",
        default="student",
        help="State-dict key in the MFM checkpoint.",
    )
    parser.add_argument(
        "--mfm-diagonal",
        action="store_true",
        help="Use diagonal MFM Euler steps for the value sampler.",
    )
    parser.add_argument("--guide-every", type=int, default=4)
    parser.add_argument("--guide-min-t", type=float, default=0.05)
    parser.add_argument("--guide-max-t", type=float, default=0.85)
    parser.add_argument("--guide-atomics", action="store_true")
    parser.add_argument(
        "--guidance-max-coord-rms",
        type=float,
        default=0.0,
        help="If positive, clip coordinate guidance gradients per sample.",
    )
    parser.add_argument(
        "--guidance-max-atom-rms",
        type=float,
        default=0.0,
        help="If positive, clip atom-type guidance gradients per sample.",
    )
    parser.add_argument("--soft-atom-temperature", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--property-name", choices=list(TFG_PROPERTIES), default="alpha"
    )
    parser.add_argument("--partition", choices=sorted(TFG_SPLITS), default="train_flow")
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument(
        "--hist-cache",
        type=Path,
        default=DEFAULT_HIST_CACHE,
    )
    parser.add_argument("--force-rebuild-hist", action="store_true")
    parser.add_argument("--compute-pb", action="store_true")
    parser.add_argument(
        "--posebusters-config",
        type=Path,
        default=Path("src/tabasco/utils/posebusters_no_strain.yaml"),
    )
    parser.add_argument("--no-sanitize", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qm9_length_conditioned_guidance"),
    )
    return parser.parse_args()


def heavy_atom_length(symbols: list[str]) -> int:
    return sum(1 for symbol in symbols if symbol != "H")


def histogram_entry(values: list[float], bins: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    value_min = float(array.min())
    value_max = float(array.max())
    if value_min == value_max:
        value_min -= 0.5
        value_max += 0.5
    counts, edges = np.histogram(array, bins=bins, range=(value_min, value_max))
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
    }


def build_or_load_histograms(args: argparse.Namespace, max_len: int) -> dict:
    resolve_hist_cache(args)
    if args.hist_cache.exists() and not args.force_rebuild_hist:
        with args.hist_cache.open(encoding="utf-8") as handle:
            hist = json.load(handle)
        metadata = hist.get("metadata", {})
        if (
            metadata.get("property_name") == args.property_name
            and metadata.get("partition") == args.partition
            and int(metadata.get("bins", -1)) == args.hist_bins
            and int(metadata.get("max_len", -1)) == max_len
        ):
            return hist

    dataset = load_dataset(
        "yairschiff/qm9",
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    selected = tfg_partition_indices(len(dataset), args.partition, limit=0)
    values_by_length: dict[int, list[float]] = defaultdict(list)
    global_values: list[float] = []
    for idx in selected:
        datapoint = dataset[int(idx)]
        length = heavy_atom_length(list(datapoint["atomic_symbols"]))
        if length < 1 or length > max_len:
            continue
        value = float(datapoint[args.property_name])
        values_by_length[length].append(value)
        global_values.append(value)

    hist = {
        "metadata": {
            "property_name": args.property_name,
            "partition": args.partition,
            "bins": args.hist_bins,
            "max_len": max_len,
            "source": "yairschiff/qm9",
            "split_seed": 42,
        },
        "global": histogram_entry(global_values, args.hist_bins),
        "lengths": {
            str(length): histogram_entry(values, args.hist_bins)
            for length, values in sorted(values_by_length.items())
            if values
        },
    }
    args.hist_cache.parent.mkdir(parents=True, exist_ok=True)
    args.hist_cache.write_text(json.dumps(hist, indent=2) + "\n")
    return hist


def sample_property_target(hist: dict, length: int, rng: np.random.Generator) -> float:
    entry = hist["lengths"].get(str(length), hist["global"])
    counts = np.asarray(entry["counts"], dtype=np.float64)
    probs = counts / counts.sum()
    bin_idx = int(rng.choice(np.arange(len(probs)), p=probs))
    edges = np.asarray(entry["edges"], dtype=np.float64)
    low = edges[bin_idx]
    high = edges[bin_idx + 1]
    if high <= low:
        return float(low)
    return float(rng.uniform(low, high))


def summary_from_rows(rows: list[dict], run: str) -> dict:
    run_rows = [row for row in rows if row["run"] == run]
    guide_errors = np.asarray([row["guide_abs_error"] for row in run_rows], dtype=float)
    oracle_errors = np.asarray([row["oracle_abs_error"] for row in run_rows], dtype=float)
    guide_values = np.asarray([row["guide_prediction"] for row in run_rows], dtype=float)
    oracle_values = np.asarray([row["oracle_prediction"] for row in run_rows], dtype=float)
    targets = np.asarray([row["target"] for row in run_rows], dtype=float)
    return {
        "run": run,
        "num_samples": len(run_rows),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "guide_prediction_mean": float(guide_values.mean()),
        "oracle_prediction_mean": float(oracle_values.mean()),
        "guide_mae": float(guide_errors.mean()),
        "guide_median_abs_error": float(np.median(guide_errors)),
        "oracle_mae": float(oracle_errors.mean()),
        "oracle_median_abs_error": float(np.median(oracle_errors)),
    }


def per_sample_rows(
    state,
    *,
    run: str,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    property_name: str,
    guide_regressor,
    oracle_regressor,
) -> list[dict]:
    guide_pred = predicted_property(
        state,
        property_name=property_name,
        property_regressor=guide_regressor,
    ).detach().cpu()
    oracle_pred = predicted_property(
        state,
        property_name=property_name,
        property_regressor=oracle_regressor,
    ).detach().cpu()
    targets_cpu = targets.detach().cpu()
    rows = []
    for sample_idx in range(targets_cpu.shape[0]):
        target = float(targets_cpu[sample_idx])
        guide_value = float(guide_pred[sample_idx])
        oracle_value = float(oracle_pred[sample_idx])
        rows.append(
            {
                "run": run,
                "sample_idx": sample_idx,
                "length": int(lengths[sample_idx]),
                "target": target,
                "guide_prediction": guide_value,
                "guide_abs_error": abs(guide_value - target),
                "oracle_prediction": oracle_value,
                "oracle_abs_error": abs(oracle_value - target),
            }
        )
    return rows


def append_pb_summary(summary_rows: list[dict], states: dict, args, data_stats: dict) -> None:
    if not args.compute_pb:
        return
    posebusters = build_posebusters(args.posebusters_config)
    by_run = {row["run"]: row for row in summary_rows}
    for run, state in states.items():
        pb = pb_summary_for_state(
            state,
            data_stats=data_stats,
            posebusters=posebusters,
            sanitize=not args.no_sanitize,
        )
        by_run[run].update({f"pb_{key}": value for key, value in pb.items()})


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, data_stats, model_args = load_flow_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    mfm_student = None
    if args.value_sampler == "mfm":
        if args.mfm_checkpoint is None:
            raise ValueError("--value-sampler=mfm requires --mfm-checkpoint")
        mfm_student = load_mfm_student(
            args.mfm_checkpoint,
            data_stats=data_stats,
            teacher_args=model_args,
            device=device,
            key=args.mfm_key,
        )

    guide_regressor = load_tfg_regressor("guide", args.property_name, device)
    oracle_regressor = load_tfg_regressor("oracle", args.property_name, device)
    for regressor in (guide_regressor, oracle_regressor):
        regressor.soft_atom_temperature = args.soft_atom_temperature
        regressor.eval()
        for param in regressor.parameters():
            param.requires_grad_(False)

    max_len = int(data_stats["max_num_atoms"])
    hist = build_or_load_histograms(args, max_len=max_len)

    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_samples).to(device)
    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
    target_rng = np.random.default_rng(args.target_seed)
    target_values = [
        sample_property_target(hist, int(length), target_rng) for length in lengths
    ]
    targets = torch.tensor(target_values, dtype=torch.float32, device=device)

    unguided, _ = guided_sample(
        model,
        init_state,
        value_sampler=args.value_sampler,
        mfm_student=mfm_student,
        mfm_diagonal=args.mfm_diagonal,
        num_steps=args.sample_steps,
        mu=0.0,
        reward_scale=args.reward_scale,
        value_samples=0,
        value_glass_steps=args.value_glass_steps,
        reward_name="target_property",
        target_x=0.0,
        property_name=args.property_name,
        property_target=targets,
        property_regressor=guide_regressor,
        guide_atomics=False,
        guide_min_t=args.guide_min_t,
        guide_max_t=args.guide_max_t,
        guide_every=args.guide_every,
        guidance_max_coord_rms=args.guidance_max_coord_rms,
        guidance_max_atom_rms=args.guidance_max_atom_rms,
        eps=args.eps,
        value_batch_size=args.value_batch_size,
    )

    seed_all(args.seed)
    guided, guided_logs = guided_sample(
        model,
        init_state,
        value_sampler=args.value_sampler,
        mfm_student=mfm_student,
        mfm_diagonal=args.mfm_diagonal,
        num_steps=args.sample_steps,
        mu=args.mu,
        reward_scale=args.reward_scale,
        value_samples=args.value_samples,
        value_glass_steps=args.value_glass_steps,
        reward_name="target_property",
        target_x=0.0,
        property_name=args.property_name,
        property_target=targets,
        property_regressor=guide_regressor,
        guide_atomics=args.guide_atomics,
        guide_min_t=args.guide_min_t,
        guide_max_t=args.guide_max_t,
        guide_every=args.guide_every,
        guidance_max_coord_rms=args.guidance_max_coord_rms,
        guidance_max_atom_rms=args.guidance_max_atom_rms,
        eps=args.eps,
        value_batch_size=args.value_batch_size,
    )

    rows = []
    rows.extend(
        per_sample_rows(
            unguided,
            run="unguided",
            lengths=lengths,
            targets=targets,
            property_name=args.property_name,
            guide_regressor=guide_regressor,
            oracle_regressor=oracle_regressor,
        )
    )
    rows.extend(
        per_sample_rows(
            guided,
            run="guided",
            lengths=lengths,
            targets=targets,
            property_name=args.property_name,
            guide_regressor=guide_regressor,
            oracle_regressor=oracle_regressor,
        )
    )
    summary_rows = [
        summary_from_rows(rows, "unguided"),
        summary_from_rows(rows, "guided"),
    ]
    delta = {"run": "guided_minus_unguided"}
    for key in (
        "guide_prediction_mean",
        "oracle_prediction_mean",
        "guide_mae",
        "guide_median_abs_error",
        "oracle_mae",
        "oracle_median_abs_error",
    ):
        delta[key] = summary_rows[1][key] - summary_rows[0][key]
    delta["num_samples"] = args.num_samples
    summary_rows.append(delta)
    append_pb_summary(
        summary_rows[:2],
        {"unguided": unguided, "guided": guided},
        args,
        data_stats,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "guided_step_logs.csv", guided_logs)
    metadata = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hist_cache": str(args.hist_cache),
        "lengths": [int(length) for length in lengths],
        "targets": [float(target) for target in target_values],
        "summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for row in summary_rows:
        print(row)
    print(f"per_sample: {args.output_dir / 'per_sample.csv'}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
