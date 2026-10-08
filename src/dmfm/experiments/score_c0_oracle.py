#!/usr/bin/env python
"""Score guided and unguided C0 samples with a frozen independent oracle.

Ported from dirichlet-flow-matching/scripts/score_yeast_c0_oracle.py (Table 1 oracle column).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dmfm.regressors.c0 import build_c0_regressor
from dmfm.utils.torch_io import torch_load


DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle_ckpt", required=True)
    p.add_argument("--oracle_model_type", default="independent_oracle", choices=["park_cnn", "independent_oracle"])
    p.add_argument("--run_glob", action="append", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--reverse_complement_average",
        action="store_true",
        help="Average oracle predictions over each sequence and its reverse complement.",
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def run_target(path: str | Path) -> float:
    with open(Path(path).with_name("args.json")) as f:
        return float(json.load(f)["target_c0"])


def encode(seqs: list[str]) -> torch.Tensor:
    rows = []
    for seq in seqs:
        seq = seq.upper()
        if len(seq) != 50 or any(base not in DNA_TO_INT for base in seq):
            raise ValueError(f"Expected 50 bp A/C/G/T sequence, got {seq!r}")
        rows.append([DNA_TO_INT[base] for base in seq])
    return F.one_hot(torch.tensor(rows), num_classes=4).float()


@torch.inference_mode()
def reverse_complement_one_hot(x_acgt: torch.Tensor) -> torch.Tensor:
    return x_acgt.flip(dims=(1,))[..., [3, 2, 1, 0]]


def score(
    model: torch.nn.Module,
    seqs: list[str],
    batch_size: int,
    device: torch.device,
    *,
    reverse_complement_average: bool,
) -> np.ndarray:
    x = encode(seqs)
    values = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        prediction = model(batch)
        if reverse_complement_average:
            prediction = 0.5 * (prediction + model(reverse_complement_one_hot(batch)))
        values.append(prediction.detach().cpu())
    return torch.cat(values).numpy()


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    paths = sorted({path for pattern in args.run_glob for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("No sample_scores.csv files matched")
    model = build_c0_regressor(args.oracle_model_type)
    model.load_state_dict(torch_load(args.oracle_ckpt, map_location="cpu"), strict=True)
    model.eval().to(args.device)
    per_run: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for path in paths:
        target = run_target(path)
        df = pd.read_csv(path)
        scored = pd.DataFrame({"sample_idx": df.get("sample_idx", pd.Series(np.arange(len(df))))})
        scored["target_c0"] = target
        scored["run_dir"] = str(Path(path).parent)
        for label in ("unguided", "guided"):
            values = score(
                model,
                df[f"seq_{label}"].tolist(),
                args.batch_size,
                torch.device(args.device),
                reverse_complement_average=args.reverse_complement_average,
            )
            scored[f"oracle_{label}"] = values
            error = np.abs(values - target)
            low, high = bootstrap_mean(error, args.bootstrap, args.seed)
            summaries.append(
                {
                    "run_dir": str(Path(path).parent),
                    "target_c0": target,
                    "set": label,
                    "n_samples": len(values),
                    "oracle_score_mean": float(values.mean()),
                    "oracle_target_mae": float(error.mean()),
                    "oracle_target_mae_ci95_low": low,
                    "oracle_target_mae_ci95_high": high,
                }
            )
        improvement = np.abs(scored["oracle_unguided"] - target) - np.abs(scored["oracle_guided"] - target)
        low, high = bootstrap_mean(improvement.to_numpy(), args.bootstrap, args.seed)
        summaries.append(
            {
                "run_dir": str(Path(path).parent),
                "target_c0": target,
                "set": "paired_guidance_effect",
                "n_samples": len(scored),
                "oracle_score_mean": np.nan,
                "oracle_target_mae": float(improvement.mean()),
                "oracle_target_mae_ci95_low": low,
                "oracle_target_mae_ci95_high": high,
                "fraction_guided_improved": float((improvement > 0).mean()),
            }
        )
        per_run.append(scored)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_run, ignore_index=True).to_csv(out_dir / "oracle_per_sample.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "oracle_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out_dir / 'oracle_per_sample.csv'}")
    print(f"wrote {out_dir / 'oracle_summary.csv'}")


if __name__ == "__main__":
    main()
