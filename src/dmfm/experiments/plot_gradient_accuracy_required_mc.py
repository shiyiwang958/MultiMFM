#!/usr/bin/env python
"""Plot operational MC requirements for RMS, relative, and directional error.

Ported from dirichlet-flow-matching/scripts/plot_gradient_accuracy_required_mc.py (Fig 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


from dmfm import paths


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gc_dir", required=True)
    parser.add_argument("--motif_dir", default=str(paths.OUTPUTS / "gradient_accuracy_motif"))
    parser.add_argument("--conjunction_dir", default=str(paths.OUTPUTS / "gradient_accuracy_conjunction"))
    parser.add_argument("--output_dir", default=str(paths.OUTPUTS / "gradient_accuracy_figures"))
    parser.add_argument("--rms_threshold", type=float, default=0.20)
    parser.add_argument("--relative_threshold", type=float, default=1.40)
    parser.add_argument("--cosine_threshold", type=float, default=0.55)
    return parser.parse_args(argv)


def first_mc(summary: pd.DataFrame, length: int, valid: pd.Series) -> int:
    subset = summary[(summary["length"] == length) & valid].sort_values("mc")
    return int(subset.iloc[0]["mc"]) if len(subset) else -1


def requirements_for_metric(
    run_dir: Path,
    *,
    metric: str,
    threshold: float,
) -> pd.DataFrame:
    if metric == "rms":
        summary = pd.read_csv(run_dir / "gradient_error_summary_all_lengths.csv")
        valid = summary["mean_e_rms"] <= threshold
        criterion = f"mean E_RMS <= {threshold:g}"
    else:
        summary = pd.read_csv(run_dir / "gradient_accuracy_relative_directional_summary.csv")
        if metric == "relative":
            valid = summary["global_relative_rmse"] <= threshold
            criterion = f"global relative RMSE <= {threshold:g}"
        elif metric == "cosine":
            valid = summary["mean_cosine_similarity"] >= threshold
            criterion = f"mean cosine similarity >= {threshold:g}"
        else:
            raise ValueError(metric)
    rows = [
        {
            "length": int(length),
            "required_mc": first_mc(summary, int(length), valid),
            "criterion": criterion,
            "metric": metric,
        }
        for length in sorted(summary["length"].unique())
    ]
    return pd.DataFrame(rows)


def plot_metric(requirements: pd.DataFrame, metric: str, output_dir: Path) -> None:
    labels = {
        "rms": "Required MC samples: coordinate RMS error",
        "relative": "Required MC samples: global relative RMSE",
        "cosine": "Required MC samples: gradient direction",
    }
    colors = {"GC control": "#0072B2", "Soft motif": "#D55E00", "Two-motif conjunction": "#009E73"}
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for reward, group in requirements.groupby("reward", sort=False):
        group = group.sort_values("length")
        attained = group[group["required_mc"] > 0]
        missing = group[group["required_mc"] < 0]
        color = colors[reward]
        if len(attained):
            axis.plot(attained["length"], attained["required_mc"], marker="o", linewidth=2, color=color, label=reward)
        if len(missing):
            # The current MC grid has a maximum of 256. Mark censored points
            # at that boundary with a downward triangle and a >256 annotation.
            axis.scatter(missing["length"], [256] * len(missing), marker="v", s=65, color=color, zorder=3)
            for _, row in missing.iterrows():
                axis.annotate(
                    ">256",
                    (row["length"], 256),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=8,
                )
    criterion = requirements["criterion"].iloc[0]
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks([50, 100, 200, 400], labels=["50", "100", "200", "400"])
    axis.set_xlabel("sequence length L")
    axis.set_ylabel("minimum tested MC sample count")
    axis.set_title(labels[metric])
    axis.text(0.02, 0.03, criterion, transform=axis.transAxes, fontsize=8, va="bottom")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"required_mc_{metric}.png", dpi=240)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    runs = {
        "GC control": Path(args.gc_dir),
        "Soft motif": Path(args.motif_dir),
        "Two-motif conjunction": Path(args.conjunction_dir),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {"rms": args.rms_threshold, "relative": args.relative_threshold, "cosine": args.cosine_threshold}
    combined: list[pd.DataFrame] = []
    for metric, threshold in thresholds.items():
        per_reward = []
        for reward, run_dir in runs.items():
            table = requirements_for_metric(run_dir, metric=metric, threshold=threshold)
            table["reward"] = reward
            per_reward.append(table)
        requirement = pd.concat(per_reward, ignore_index=True)
        requirement.to_csv(out_dir / f"required_mc_{metric}.csv", index=False)
        plot_metric(requirement, metric, out_dir)
        combined.append(requirement)
    pd.concat(combined, ignore_index=True).to_csv(out_dir / "required_mc_all_metrics.csv", index=False)
    print(f"Wrote plots and tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
