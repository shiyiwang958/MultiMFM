#!/usr/bin/env python
"""Redraw the regenerated DNA figures (6, 7, 8) from ``results/dna/regen`` on CPU.

Every function takes the small CSVs written by :mod:`dmfm.regen.base_marginals`,
``dmfm.experiments.sample_c0_guidance`` and :mod:`dmfm.regen.guidance_trace` and returns a
matplotlib figure, so ``notebooks/09_dna_regenerated.ipynb`` rebuilds the figures with
``RERUN = False`` in seconds. The panel geometry, labels and colour maps follow the
submitted figures (``viridis`` heat maps with a shared 0..max scale, ``coolwarm``
symmetric difference maps, grouped marginal bars, KDE of the cyclizability scores,
50-bin density histograms for Fig 7, three log/linear trace panels for Fig 8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BASES = ("A", "C", "G", "T")
# seaborn "deep" palette, as in the published Fig 6 (real / generated / uniform)
SOURCE_COLORS = {"real": "#4C72B0", "generated": "#DD8452", "uniform": "#55A868"}


def _pivot(pos: "pd.DataFrame", source: str) -> np.ndarray:
    """``[4, L]`` frequency matrix (rows A, C, G, T) for one source."""
    sub = pos[pos.source == source].sort_values("position")
    return np.stack([sub[b].to_numpy(dtype=float) for b in BASES])


# ------------------------------------------------------------------------ Fig 6
def fig6(results_dir: str | Path, *, real_source: str = "real", figsize=(11.5, 9.0)):
    """The four Fig 6 blocks stacked into one figure."""
    import matplotlib.pyplot as plt
    import pandas as pd

    results_dir = Path(results_dir)
    pos = pd.read_csv(results_dir / "position_freqs.csv")
    glob = pd.read_csv(results_dir / "global_marginals.csv")
    scores = pd.read_csv(results_dir / "c0_scores.csv")

    mats = {s: _pivot(pos, s if s != "real" else real_source) for s in ("real", "generated", "uniform")}
    vmax = max(float(m.max()) for m in mats.values())

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.6], hspace=0.75, wspace=0.9)

    # row 1: the three heat maps
    for k, name in enumerate(("real", "generated", "uniform")):
        ax = fig.add_subplot(gs[0, 2 * k : 2 * k + 2])
        im = ax.imshow(mats[name], aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax,
                       extent=[-0.5, mats[name].shape[1] - 0.5, 3.5, -0.5])
        ax.set_yticks(range(4), BASES)
        ax.set_title(name)
        ax.set_xlabel("position")
        if k == 0:
            ax.set_ylabel("base")
        if k == 2:
            fig.colorbar(im, ax=ax, fraction=0.05)

    # row 2: the two difference maps
    for k, name in enumerate(("generated", "uniform")):
        d = mats[name] - mats["real"]
        lim = float(np.abs(d).max())
        ax = fig.add_subplot(gs[1, 3 * k : 3 * k + 3])
        im = ax.imshow(d, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                       extent=[-0.5, d.shape[1] - 0.5, 3.5, -0.5])
        ax.set_yticks(range(4), BASES)
        ax.set_title(f"{name} - real")
        ax.set_xlabel("position")
        if k == 0:
            ax.set_ylabel("base")
        fig.colorbar(im, ax=ax, fraction=0.05)

    # row 3 left: global marginals
    ax = fig.add_subplot(gs[2, 0:3])
    width = 0.26
    xs = np.arange(4)
    for k, name in enumerate(("real", "generated", "uniform")):
        src = real_source if name == "real" else name
        vals = [float(glob[(glob.source == src) & (glob.base == b)].freq.iloc[0]) for b in BASES]
        ax.bar(xs + (k - 1) * width, vals, width, label=name, color=SOURCE_COLORS[name])
    ax.set_xticks(xs, BASES)
    ax.set_xlabel("base")
    ax.set_ylabel("freq")
    ax.set_title("Global base marginals")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    # row 3 right: cyclizability distributions
    ax = fig.add_subplot(gs[2, 3:6])
    _kde_panel(ax, scores)
    return fig


def _kde_panel(ax, scores, column: str = "c0_pred_forward") -> None:
    from scipy import stats

    lo = float(scores[column].min())
    hi = float(scores[column].max())
    grid = np.linspace(lo - 0.1, hi + 0.1, 400)
    for name in ("real", "generated", "uniform"):
        v = scores.loc[scores.source == name, column].to_numpy(dtype=float)
        if v.size == 0:
            continue
        ax.plot(grid, stats.gaussian_kde(v)(grid), label=name, color=SOURCE_COLORS[name], linewidth=2.5)
    ax.set_xlabel("cyclizability")
    ax.set_ylabel("Density")
    ax.set_title("Cyclizability Score Distributions")
    ax.legend(title="source")
    ax.grid(alpha=0.25)


# ------------------------------------------------------------------------ Fig 7
def fig7(runs, *, target: float = 0.30, bins: int = 50, figsize=(13.5, 3.6), titles=None):
    """Fig 7's three histograms.

    ``runs`` is a list of ``(label, sample_scores.csv path)`` in increasing guidance
    strength. The histogram is drawn exactly as ``sample_c0_guidance`` drew
    ``histogram.png``: shared 50 bins over the pooled range, ``density=True``, alpha 0.55.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(1, len(runs), figsize=figsize, squeeze=False)
    for ax, (label, path) in zip(axes[0], runs):
        df = pd.read_csv(path)
        unguided = df["unguided"].to_numpy(dtype=float)
        guided = df["guided"].to_numpy(dtype=float)
        edges = np.histogram_bin_edges(np.concatenate([unguided, guided]), bins=bins)
        ax.hist(unguided, bins=edges, alpha=0.55, density=True, label="unguided")
        ax.hist(guided, bins=edges, alpha=0.55, density=True, label="guided")
        ax.axvline(target, color="black", linestyle="--", label=f"target {target:g}")
        ax.set_xlabel("Cyclizability C0 score")
        ax.set_ylabel("density")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Unguided vs guided score distribution")
    fig.tight_layout()
    return fig


def fig7_panels(runs, *, target: float = 0.30, bins: int = 50, figsize=(8.0, 4.5)):
    """One standalone figure per guidance strength, so the paper can keep its three
    ``subfigure`` blocks and their sub-captions.

    Same drawing code as :func:`fig7` and as the original ``sample_c0_guidance``
    (``figsize=(8, 4.5)``, 50 shared bins, ``density=True``, alpha 0.55, dashed target line,
    title "Unguided vs guided score distribution"). Returns ``[(label, figure), ...]``.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    out = []
    for label, path in runs:
        df = pd.read_csv(path)
        unguided = df["unguided"].to_numpy(dtype=float)
        guided = df["guided"].to_numpy(dtype=float)
        edges = np.histogram_bin_edges(np.concatenate([unguided, guided]), bins=bins)
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(unguided, bins=edges, alpha=0.55, density=True, label="unguided")
        ax.hist(guided, bins=edges, alpha=0.55, density=True, label="guided")
        ax.axvline(target, color="black", linestyle="--", label=f"target {target:g}")
        ax.set_xlabel("Cyclizability C0 score")
        ax.set_ylabel("density")
        ax.set_title("Unguided vs guided score distribution")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out.append((label, fig))
    return out


# ------------------------------------------------------------------------ Fig 8
def fig8(trace_csv: str | Path, *, sample_idx: int | None = None, target: float = 0.30,
         guide_window: tuple[float, float] | None = None, figsize=(15.0, 3.6)):
    """The three Fig 8 trace panels for one sample."""
    import matplotlib.pyplot as plt
    import pandas as pd

    tr = pd.read_csv(trace_csv)
    if sample_idx is None:
        sample_idx = int(tr.sample_idx.iloc[0])
    tr = tr[tr.sample_idx == sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        ("grad_norm", r"$\|\nabla_x V_t(x_t)\|$", "Gradient Norm", True),
        ("value", r"$V_t(x_t)$", "Value", False),
        ("endpoint_hard_score", "predicted endpoint hard mean", "Predicted Endpoint Score", False),
    ]
    for ax, (col, ylabel, title, logy) in zip(axes, panels):
        for arm in ("unguided", "guided"):
            sub = tr[tr.arm == arm].sort_values("t")
            ax.plot(sub["t"].to_numpy(), sub[col].to_numpy(), label=arm)
        if logy:
            ax.set_yscale("log")
        if col == "endpoint_hard_score":
            ax.axhline(target, color="black", linestyle=":", label=f"target {target:g}")
        if guide_window is not None:
            for v in guide_window:
                ax.axvline(v, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("time t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
