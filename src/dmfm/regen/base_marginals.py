#!/usr/bin/env python
"""Fig 6 and Table 16: base-model marginals, C0 distributions and correlations.

**New implementation, following the authors' own recipe.** The code that produced the
published Fig 6 PDFs (dated 2026-05-03, seaborn style) and Table 16 does not survive. What
does survive is the authors' June re-creation of the same four PDFs for the ``split65k``
teacher, ``scripts/make_yeast_split_sample_figures.py`` of GitHub ``tullebulle/DNA-MFM``
(commit d40b0f6, 2026-06-09; verbatim copy in
``results/dna/original_scripts/make_yeast_split_sample_figures.py``). Its protocol is used
here for the headline numbers: 4,096 generated sequences integrated with 128 exact
denoiser-flow steps on ``linspace(0, 1)``, 4,096 real windows drawn at random from the
held-out ``test`` split, 4,096 i.i.d. uniform sequences, forward (not RC-averaged) guide
scores. That script draws no Table 16, so the correlations are added on top, exactly as the
Table 16 caption describes. The panels are:

Fig 6 panels (``figures/{base,gen_vs_real,marginals,cycl_dist}.pdf`` of the submission)

1. ``base.pdf``      position-wise base frequency heat maps, one per source
                     (real / generated / uniform), rows A C G T, columns 0..L-1.
2. ``gen_vs_real.pdf`` the two difference heat maps (generated - real, uniform - real).
3. ``marginals.pdf`` global base marginals (A, C, G, T) as grouped bars.
4. ``cycl_dist.pdf`` KDE of the cyclizability score of each source.

Table 16 = Pearson correlation of the real statistics against generated and against
uniform, both for the 4 global marginals and for the flattened 4 x L position-wise
frequency table (the published p-values, 9.5e-4 on 4 points and 6.02e-157 on 200
points, identify those two sample sizes).

Differences from that recipe, all forced: the parent-disjoint base DFM replaces the purged
``split65k`` teacher, the parent-disjoint split replaces the random 65k/8k split, and the
shipped ``park_cnn`` guide replaces the purged Keras-converted ``dinko/C0free_torch.pt``.
Because Table 16's position-wise r depends on the sample size, ``correlations.csv`` also
reports r at nested prefix sizes of a larger generated set and against other real subsets
(``full``, ``a_train``, all ``test`` windows).

Usage::

    python -m dmfm.regen.base_marginals --n 10000 --shard 0 --num_shards 1   # sample
    python -m dmfm.regen.base_marginals --n 10000 --num_shards 1 --reduce    # statistics

Sharding: the initial noise is drawn once from a single seeded CPU generator and sliced,
so shard ``k`` of ``K`` produces exactly the rows it would produce in a single run, and
``--skip_existing`` makes a preempted shard rerunnable on its own. ``--reduce`` merges the
per-shard sample files into the tracked CSVs under ``results/dna/regen/fig6_table16``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from dmfm import api, paths

BASES = ("A", "C", "G", "T")


# --------------------------------------------------------------------------- statistics
def position_frequencies(tokens: np.ndarray, alphabet_size: int = 4) -> np.ndarray:
    """``[L, alphabet_size]`` frequency of each base at each position."""
    tokens = np.asarray(tokens)
    n, length = tokens.shape
    counts = np.zeros((length, alphabet_size), dtype=np.float64)
    for b in range(alphabet_size):
        counts[:, b] = (tokens == b).sum(axis=0)
    return counts / float(n)


def global_marginals(tokens: np.ndarray, alphabet_size: int = 4) -> np.ndarray:
    """``[alphabet_size]`` overall base frequency."""
    tokens = np.asarray(tokens)
    return np.array([(tokens == b).mean() for b in range(alphabet_size)], dtype=np.float64)


def pearson(a, b) -> tuple[float, float]:
    """Pearson r and two-sided p-value (``scipy.stats.pearsonr``)."""
    from scipy import stats

    res = stats.pearsonr(np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel())
    return float(res[0]), float(res[1])


# --------------------------------------------------------------------------- sampling
def _noise(n: int, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn((n, length, 4), generator=g, dtype=torch.float32)


@torch.no_grad()
def sample_generated(
    base,
    cfg,
    *,
    n: int,
    length: int = 50,
    batch_size: int = 512,
    nfe: int = 64,
    t_max: float = 0.95,
    seed: int = 0,
    device=None,
    offset: int = 0,
    total: int | None = None,
    verbose: bool = True,
    return_margin: bool = False,
):
    """Unguided base-DFM samples as ``int8[n, length]`` token ids (A=0, C=1, G=2, T=3).

    ``offset``/``total`` select a contiguous slice of the single seeded noise tensor, so a
    sharded run is bit-identical to the corresponding slice of an unsharded one. The noise
    is always drawn by a seeded *CPU* generator and only then moved to ``device``, so the
    randomness is identical on CPU and GPU and the two differ only in float arithmetic.

    ``return_margin=True`` also returns the per-token decision margin of the final state
    (largest minus second-largest coordinate, ``float32[n, length]``). It is the evidence
    that float-level differences between devices cannot change the decoded sequences: a
    margin far above the float32 noise floor means ``argmax`` is decided by a wide gap.
    """
    device = torch.device(device) if device is not None else next(base.parameters()).device
    eps_all = _noise(int(total or (offset + n)), length, seed)[offset : offset + n]
    out = np.empty((n, length), dtype=np.int8)
    margin = np.empty((n, length), dtype=np.float32) if return_margin else None
    t0 = time.time()
    for i in range(0, n, batch_size):
        x0 = eps_all[i : i + batch_size].to(device)
        x = api.sample_unguided(base, cfg, x0, nfe=nfe, t_max=t_max)
        out[i : i + x0.shape[0]] = x.argmax(-1).to(torch.int8).cpu().numpy()
        if margin is not None:
            top2 = x.float().topk(2, dim=-1).values
            margin[i : i + x0.shape[0]] = (top2[..., 0] - top2[..., 1]).cpu().numpy()
        if verbose:
            print(f"  sampled {min(i + batch_size, n)}/{n}  ({time.time() - t0:.1f}s)", flush=True)
    return (out, margin) if return_margin else out


def margin_summary(margin: np.ndarray) -> dict:
    """Decision-margin statistics: the device-independence evidence for the decoded tokens.

    ``argmax`` over the final state can only differ between two float implementations where
    the top-two gap is within their disagreement. float32 carries ~1e-7 relative precision and
    the integration compounds it over the trajectory, so a margin floor orders of magnitude
    above that leaves no token whose decode is in doubt.
    """
    m = np.asarray(margin, dtype=np.float64).ravel()
    return {
        "n_tokens": int(m.size),
        "min": float(m.min()),
        "p0.01": float(np.quantile(m, 1e-4)),
        "p1": float(np.quantile(m, 0.01)),
        "median": float(np.median(m)),
        "frac_below_1e-4": float((m < 1e-4).mean()),
        "frac_below_1e-3": float((m < 1e-3).mean()),
        "frac_below_1e-2": float((m < 1e-2).mean()),
    }


def uniform_tokens(n: int, length: int = 50, seed: int = 1) -> np.ndarray:
    """i.i.d. uniform A/C/G/T baseline."""
    return np.random.default_rng(seed).integers(0, 4, size=(n, length), dtype=np.int64).astype(np.int8)


@torch.no_grad()
def c0_scores(guide, tokens: np.ndarray, *, device=None, batch_size: int = 4096) -> dict[str, np.ndarray]:
    """Guide-CNN cyclizability of hard sequences: forward and reverse-complement-averaged."""
    device = torch.device(device) if device is not None else next(guide.parameters()).device
    tok = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
    fwd, rcm = [], []
    for i in range(0, tok.shape[0], batch_size):
        x = api.one_hot(tok[i : i + batch_size].to(device))
        f = guide(x)
        r = guide(api.reverse_complement(x))
        fwd.append(f.float().cpu().numpy())
        rcm.append(((f + r) / 2).float().cpu().numpy())
    return {"forward": np.concatenate(fwd), "rc_mean": np.concatenate(rcm)}


# --------------------------------------------------------------------------- driver
def device_name(device) -> str:
    """Human-readable accelerator name, e.g. 'NVIDIA H100 80GB HBM3' or 'cpu'.

    Recorded per run so the exact card behind any number stays recoverable: `gpu_requeue` is a
    mixed pool (A100 40/80 GB, H100, H200, RTX Pro, plus MIG slices) while `kempner_requeue` is
    H100-only, and results from the two may share one output tree.
    """
    try:
        if torch.device(device).type == "cuda":
            return torch.cuda.get_device_name(torch.device(device))
    except Exception:  # pragma: no cover - no CUDA, or a driver hiccup
        pass
    return str(device)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fig 6 / Table 16 base-model marginals (regenerated).")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n", type=int, default=10000, help="generated sample count (the first --headline_n are the headline set)")
    p.add_argument("--headline_n", type=int, default=4096,
                   help="generated / uniform / real sample count of the headline figure and Table 16 (June script: 4096)")
    p.add_argument("--real_split", default="test", choices=["full", "a_train", "a_val", "b_train", "b_val", "test"],
                   help="split the headline real sample is drawn from (June script: test)")
    p.add_argument("--real_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--nfe", type=int, default=128, help="unguided integration steps (June script: 128)")
    p.add_argument("--t_max", type=float, default=1.0, help="integration end time (June script: 1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--uniform_seed", type=int, default=1)
    p.add_argument("--ckpt", default=None, help="base DFM checkpoint (default: checkpoints/dna/base/L<length>/best.pt)")
    p.add_argument("--guide_ckpt", default=None)
    p.add_argument("--sample_dir", default=None, help="where per-shard token files go (default outputs/dna/regen/fig6)")
    p.add_argument("--out_dir", default=None, help="tracked CSV directory (default results/dna/regen/fig6_table16)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--reduce", action="store_true", help="merge shards and write the tracked CSVs")
    p.add_argument("--prefix_sizes", type=int, nargs="*", default=None,
                   help="nested prefix sizes for the correlation-vs-n table (default: 1000, 2000, 4096, n)")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _shard_bounds(n: int, shard: int, num_shards: int) -> tuple[int, int]:
    edges = np.linspace(0, n, num_shards + 1).round().astype(int)
    return int(edges[shard]), int(edges[shard + 1])


def run_shard(args) -> Path:
    sample_dir = Path(args.sample_dir or (paths.OUTPUTS / "regen" / "fig6"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = _shard_bounds(args.n, args.shard, args.num_shards)
    out = sample_dir / f"generated_tokens_shard{args.shard:03d}of{args.num_shards:03d}.npz"
    if args.skip_existing and out.exists():
        print(f"shard {args.shard}: {out} exists, skipping", flush=True)
        return out
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base, cfg = api.load_base(args.length, device, ckpt=args.ckpt)
    print(f"shard {args.shard}: sampling rows [{lo}, {hi}) of {args.n} on {device}", flush=True)
    tok, margin = sample_generated(
        base, cfg, n=hi - lo, length=args.length, batch_size=args.batch_size, nfe=args.nfe,
        t_max=args.t_max, seed=args.seed, device=device, offset=lo, total=args.n,
        return_margin=True,
    )
    ms = margin_summary(margin)
    print(f"shard {args.shard}: decode margin min={ms['min']:.4f} median={ms['median']:.4f} "
          f"frac<1e-3={ms['frac_below_1e-3']:.3g}", flush=True)
    np.savez_compressed(
        out, tokens=tok, lo=lo, hi=hi, n=args.n, seed=args.seed, nfe=args.nfe, t_max=args.t_max,
        margin_summary=json.dumps(ms),
        device=str(device),
        device_name=device_name(device),
        torch_version=torch.__version__,
        slurm=json.dumps({k: os.environ.get(v) for k, v in {
            "job_id": "SLURM_JOB_ID", "array_job_id": "SLURM_ARRAY_JOB_ID",
            "array_task_id": "SLURM_ARRAY_TASK_ID", "partition": "SLURM_JOB_PARTITION",
            "account": "SLURM_JOB_ACCOUNT", "nodelist": "SLURM_JOB_NODELIST"}.items()}),
    )
    print(f"shard {args.shard}: wrote {out}", flush=True)
    return out


def reduce_shards(args) -> Path:
    import pandas as pd

    sample_dir = Path(args.sample_dir or (paths.OUTPUTS / "regen" / "fig6"))
    out_dir = Path(args.out_dir or (paths.RESULTS / "regen" / "fig6_table16"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(sample_dir.glob(f"generated_tokens_shard*of{args.num_shards:03d}.npz"))
    if len(files) != args.num_shards:
        raise SystemExit(f"expected {args.num_shards} shard files in {sample_dir}, found {len(files)}: {files}")
    parts = []
    shard_runs = []
    for f in files:
        z = np.load(f)
        parts.append((int(z["lo"]), np.asarray(z["tokens"])))
        # record the sampling settings actually used by the shards
        args.nfe, args.t_max, args.seed = int(z["nfe"]), float(z["t_max"]), int(z["seed"])
        shard_runs.append({
            "file": f.name,
            "rows": [int(z["lo"]), int(z["hi"])],
            "device": str(z["device"]) if "device" in z else "unrecorded",
            "device_name": str(z["device_name"]) if "device_name" in z else "unrecorded",
            "torch": str(z["torch_version"]) if "torch_version" in z else "unrecorded",
            "slurm": json.loads(str(z["slurm"])) if "slurm" in z else {},
            "decode_margin": json.loads(str(z["margin_summary"])) if "margin_summary" in z else {},
        })
    parts.sort(key=lambda p: p[0])
    generated = np.concatenate([p[1] for p in parts], axis=0)
    if generated.shape[0] != args.n:
        raise SystemExit(f"merged {generated.shape[0]} rows, expected {args.n}")

    headline_n = int(min(args.headline_n, args.n))
    uniform = uniform_tokens(args.n, args.length, seed=args.uniform_seed)
    windows = api.load_windows(args.length)
    all_tokens = windows["seqs"].numpy().astype(np.int8)
    measured = windows.get("c0")

    # Headline real sample (June script: 4,096 random windows of the held-out test split).
    pool_idx = (np.arange(all_tokens.shape[0]) if args.real_split == "full"
                else api.split_indices(args.length, args.real_split).numpy())
    g = torch.Generator().manual_seed(int(args.real_seed))
    perm = torch.randperm(len(pool_idx), generator=g).numpy()
    real_idx = np.sort(pool_idx[perm[: min(headline_n, len(pool_idx))]])
    real_sets = {
        f"{args.real_split}_sample": all_tokens[real_idx],        # headline
        "full": all_tokens,
        "a_train": all_tokens[api.split_indices(args.length, "a_train").numpy()],
        "test": all_tokens[api.split_indices(args.length, "test").numpy()],
    }
    headline_real = f"{args.real_split}_sample"

    # ---- position-wise and global frequencies (headline sources + reference sets)
    sources = {
        "real": real_sets[headline_real],
        "generated": generated[:headline_n],
        "uniform": uniform[:headline_n],
        "generated_all": generated,
        **{f"real_{k}": v for k, v in real_sets.items() if k != headline_real},
    }
    pos_rows, glob_rows = [], []
    for name, tok in sources.items():
        pf = position_frequencies(tok)
        gm = global_marginals(tok)
        for pos in range(pf.shape[0]):
            pos_rows.append({"source": name, "n_sequences": int(tok.shape[0]), "position": pos,
                             **{b: float(pf[pos, i]) for i, b in enumerate(BASES)}})
        for i, b in enumerate(BASES):
            glob_rows.append({"source": name, "n_sequences": int(tok.shape[0]), "base": b, "freq": float(gm[i])})
    pd.DataFrame(pos_rows).to_csv(out_dir / "position_freqs.csv", index=False)
    pd.DataFrame(glob_rows).to_csv(out_dir / "global_marginals.csv", index=False)

    # ---- Table 16 correlations: headline + sensitivity to n and to the real reference set
    prefix_sizes = sorted({*(args.prefix_sizes or [1000, 2000]), headline_n, args.n})
    prefix_sizes = [k for k in prefix_sizes if k <= args.n]
    corr_rows = []
    for real_name, real_tok in real_sets.items():
        real_pf, real_gm = position_frequencies(real_tok), global_marginals(real_tok)
        for k in prefix_sizes:
            for name, tok in (("generated", generated), ("uniform", uniform)):
                pf = position_frequencies(tok[:k])
                gm = global_marginals(tok[:k])
                r_g, p_g = pearson(gm, real_gm)
                r_p, p_p = pearson(pf.T.ravel(), real_pf.T.ravel())
                common = {"comparison": f"{name} vs. real", "real_set": real_name, "n_real": int(real_tok.shape[0]),
                          "n_sequences": k, "headline": bool(real_name == headline_real and k == headline_n)}
                corr_rows.append({"scope": "global", **common, "n_points": 4, "pearson_r": r_g, "p_value": p_g})
                corr_rows.append({"scope": "position-wise", **common, "n_points": int(pf.size),
                                  "pearson_r": r_p, "p_value": p_p})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "correlations.csv", index=False)

    # ---- cyclizability distributions of the three headline sources (forward guide score, as
    # in the June script; measured C0 kept for the real windows)
    guide = api.load_c0("guide", device, ckpt=args.guide_ckpt)
    score_rows = []
    index_of = {"real": real_idx, "generated": np.arange(headline_n), "uniform": np.arange(headline_n)}
    for name in ("real", "generated", "uniform"):
        tok = sources[name]
        sc = c0_scores(guide, tok, device=device)["forward"]
        meas = (measured.numpy()[real_idx] if (name == "real" and measured is not None) else None)
        for j in range(tok.shape[0]):
            score_rows.append({"source": name, "index": int(index_of[name][j]),
                               "c0_pred_forward": round(float(sc[j]), 5),
                               "c0_measured": (round(float(meas[j]), 5) if meas is not None else "")})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out_dir / "c0_scores.csv", index=False)

    meta = {
        "item": "Fig 6 + Table 16 (regenerated)",
        "status": "new implementation of the lost producer, following the authors' June recipe "
                  "(results/dna/original_scripts/make_yeast_split_sample_figures.py)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_rev": _git_rev(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "headline": {"n_generated": headline_n, "n_uniform": headline_n, "real_set": headline_real,
                     "n_real": int(real_sets[headline_real].shape[0]), "real_seed": int(args.real_seed)},
        "n_generated_total": int(args.n),
        "n_real_sets": {k: int(v.shape[0]) for k, v in real_sets.items()},
        "length": int(args.length),
        "nfe": int(args.nfe),
        "t_max": float(args.t_max),
        "seed": int(args.seed),
        "uniform_seed": int(args.uniform_seed),
        "num_shards": int(args.num_shards),
        "shard_runs": shard_runs,
        "devices": sorted({r["device"] for r in shard_runs}),
        "device_names": sorted({r.get("device_name", "unrecorded") for r in shard_runs}),
        "decode_margin_min_over_shards": (
            min((r["decode_margin"]["min"] for r in shard_runs if r["decode_margin"]), default=None)),
        "decode_margin_max_frac_below_1e-3": (
            max((r["decode_margin"]["frac_below_1e-3"] for r in shard_runs if r["decode_margin"]), default=None)),
        "device_note": (
            "Initial noise is always drawn by a seeded CPU generator and only then moved to the "
            "sampling device, so CPU and GPU runs share identical randomness and differ only in "
            "float arithmetic (fp32 on CPU; fp32/TF32 on GPU). `shard_runs[*].decode_margin` gives "
            "the top-1 minus top-2 gap of the final state per token: a floor far above the float32 "
            "noise floor means no token's argmax is close enough to be flipped by that difference. "
            "These figures are in any case distributional summaries over thousands of samples."),
        "base_ckpt": paths.rel(args.ckpt or paths.base_ckpt(args.length)),
        "guide_ckpt": paths.rel(args.guide_ckpt or paths.c0_guide()),
        "data": paths.rel(paths.data_pt(args.length)),
        "prefix_sizes": prefix_sizes,
        "c0_mean": {k: float(scores[scores.source == k].c0_pred_forward.mean()) for k in ("real", "generated", "uniform")},
        "c0_std": {k: float(scores[scores.source == k].c0_pred_forward.std()) for k in ("real", "generated", "uniform")},
        "command": f"python -m dmfm.regen.base_marginals --n {args.n} --num_shards {args.num_shards} "
                   f"--nfe {args.nfe} --t_max {args.t_max} --reduce",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(corr[corr.headline].to_string(index=False), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return out_dir


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.reduce:
        reduce_shards(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
