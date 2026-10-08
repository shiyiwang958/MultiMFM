#!/usr/bin/env python
"""Recompute appendix Tables 6/7 (yeast C0 sequence diversity) for the parent-disjoint model.

Metric definitions are NOT reimplemented: every Hamming/uniqueness/novelty statistic is
computed by the original functions of DNA-MFM@187fe7b `scripts/analyze_yeast_c0_diversity.py`,
vendored verbatim as `dmfm.experiments._yeast_c0_diversity_metrics` (originally
`vendor/dna_mfm_187fe7b/`, git blob 5dd21edb...). Only the reference
sets change: the old leaky row split (train/val/test) is replaced by the parent-disjoint split
(a_train = generator training parents, a_val = generator selection parents, test = held-out
parents). The held-out ratios use the same formula as the original
`write_yeast_split_diversity_markdown.add_heldout_comparisons` (generated / test_sample).

Additions that are NOT in the old tables are written to separately named columns:
  * oracle-scored mean (independent B_train oracle, from each run's oracle/ subdir),
  * nearest-a_train distance / exact novelty of real held-out windows (memorization control),
  * a size-matched held-out null: random n-window subsets of `test`, n = generated set size,
    because nearest-generated distance depends strongly on set size.
CPU only.

Ported from dirichlet-flow-matching/workdir/rebuttal_parent_c0_diversity_20260925/analyze_parent_c0_diversity.py
(parent-disjoint recompute of Tables 4-5; the published Tables 4-5 came from the purged
split65k pipeline, see docs/provenance/dna/README.md). Run on the shipped n=1000 sets it
reproduces workdir/rebuttal_parent_c0_diversity_n1000_20260925 exactly.

Port changes: package imports; repo-relative defaults; when a run directory has no
`sequences_{guided,unguided}.fa` (results/dna keeps only `sample_scores.csv`), the sequences
are taken from the `seq_*` columns of `sample_scores.csv` (the original asserted that the two
agree: `fasta_equals_csv_in_order`), and `sample_counts.csv` records `fasta_source=csv`.

    python -m dmfm.experiments.analyze_c0_diversity \
        --run_dirs results/dna/c0_guidance/n1000/target_m1p0 results/dna/c0_guidance/n1000/target_1p0 \
        --out_dir outputs/dna/c0_diversity
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm import paths

# The original loaded the verbatim DNA-MFM@187fe7b analyzer out of a ``vendor/``
# directory next to the run; it is vendored as a module here instead.
from dmfm.experiments import _yeast_c0_diversity_metrics as orig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", default=str(paths.REPO_ROOT))
    p.add_argument(
        "--run_dirs",
        nargs="+",
        default=[
            # n=1000 reruns of the Table 1 pilot (copied to results/dna from
            # workdir/yeast_parent_glass_n1000_target{m1p0,1p0}_g8.0_seed0); paths are
            # relative to --repo_root. The original script's defaults were the n=32 pilot,
            # workdir/yeast_parent_glass_pilot_target{m1p0,1p0}_g8.0_seed0.
            "results/dna/c0_guidance/n1000/target_m1p0",
            "results/dna/c0_guidance/n1000/target_1p0",
        ],
    )
    p.add_argument("--data_pt", default=str(paths.data_pt(50).relative_to(paths.REPO_ROOT)))
    p.add_argument("--split_pt", default=str(paths.split_pt(50).relative_to(paths.REPO_ROOT)))
    p.add_argument("--out_dir", default=str(paths.OUTPUTS / "c0_diversity"))
    # Same defaults as the original analyzer / the original Table 6-7 command.
    p.add_argument("--k_nearest_train", type=int, default=5)
    p.add_argument("--gen_chunk", type=int, default=64)
    p.add_argument("--train_ref_chunk", type=int, default=2048)
    p.add_argument("--pair_chunk", type=int, default=128)
    p.add_argument("--max_pairwise_n", type=int, default=3000)
    p.add_argument("--baseline_n", type=int, default=1000, help="old --train_baseline_n")
    p.add_argument("--seed", type=int, default=0)
    # Additions.
    p.add_argument("--size_matched_draws", type=int, default=1000)
    p.add_argument(
        "--oracle_ckpt",
        default=str(paths.c0_oracle().relative_to(paths.REPO_ROOT)),
        help="If it exists, re-score sequences on CPU and check against the stored oracle CSVs. '' to skip.",
    )
    return p.parse_args(argv)


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    headers, seqs, cur = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            headers.append(line[1:])
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    if len(headers) != len(seqs):
        raise ValueError(f"{path}: {len(headers)} headers vs {len(seqs)} sequences")
    return headers, seqs


def sample_counts_and_identity(run_dirs: list[Path], out_dir: Path) -> dict:
    rows, per_run = [], {}
    for run_dir in run_dirs:
        meta = json.loads((run_dir / "args.json").read_text())
        df = pd.read_csv(run_dir / "sample_scores.csv")
        oracle = pd.read_csv(run_dir / "oracle" / "oracle_per_sample.csv")
        info = {"target_c0": float(meta["target_c0"]), "df": df}
        for kind in ("guided", "unguided"):
            csv_seqs = df[f"seq_{kind}"].astype(str).str.upper().tolist()
            fasta = run_dir / f"sequences_{kind}.fa"
            if fasta.exists():
                headers, seqs = read_fasta(fasta)
                fasta_source = "fasta"
            else:  # port addition: results/dna keeps only sample_scores.csv
                headers = [f"{kind}_{int(i)}" for i in df["sample_idx"]]
                seqs = list(csv_seqs)
                fasta_source = "csv"
            info[kind] = seqs
            info[f"{kind}_headers"] = headers
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "target_c0": float(meta["target_c0"]),
                    "set": kind,
                    "n_fasta": len(seqs),
                    "n_sample_scores_csv": len(csv_seqs),
                    "n_oracle_csv": int(oracle[f"oracle_{kind}"].notna().sum()),
                    "fasta_equals_csv_in_order": seqs == csv_seqs,
                    "seq_lengths": ",".join(str(x) for x in sorted({len(s) for s in seqs})),
                    "n_exact_unique": len(set(seqs)),
                    "n_samples_args_json": int(meta["n_samples"]),
                    "seed_args_json": int(meta["seed"]),
                    **({} if fasta_source == "fasta" else {"fasta_source": fasta_source}),
                }
            )
        per_run[float(meta["target_c0"])] = info
    counts = pd.DataFrame(rows)
    counts.to_csv(out_dir / "sample_counts.csv", index=False)

    targets = sorted(per_run)
    a, b = per_run[targets[0]], per_run[targets[-1]]
    ua, ub = a["unguided"], b["unguided"]
    n_common = min(len(ua), len(ub))
    identity = {
        "compared": f"unguided target {targets[0]:+g} vs unguided target {targets[-1]:+g}",
        "n_a": len(ua),
        "n_b": len(ub),
        "identical_in_order": ua == ub,
        "identical_as_multisets": sorted(ua) == sorted(ub),
        "n_positions_differing_in_order": int(sum(x != y for x, y in zip(ua, ub)) + abs(len(ua) - len(ub))),
        "n_shared_distinct_sequences": len(set(ua) & set(ub)),
        "fasta_headers_identical": a["unguided_headers"] == b["unguided_headers"],
        "guide_scores_identical": bool(
            np.array_equal(a["df"]["unguided"].to_numpy(), b["df"]["unguided"].to_numpy())
        ),
        "guided_sets_n_shared_sequences": len(set(a["guided"]) & set(b["guided"])),
        "guided_vs_own_unguided_n_identical_pairs": {
            f"{t:+g}": int(sum(x == y for x, y in zip(per_run[t]["guided"], per_run[t]["unguided"]))) for t in targets
        },
        "n_compared_in_order": n_common,
    }
    oa = pd.read_csv(run_dirs[0] / "oracle" / "oracle_per_sample.csv")["oracle_unguided"].to_numpy()
    ob = pd.read_csv(run_dirs[-1] / "oracle" / "oracle_per_sample.csv")["oracle_unguided"].to_numpy()
    identity["oracle_unguided_scores_identical"] = bool(np.array_equal(oa, ob))
    (out_dir / "unguided_identity.json").write_text(json.dumps(identity, indent=2))
    return {"counts": counts, "identity": identity, "per_run": per_run}


def kmer_freq(tok: np.ndarray, k: int) -> np.ndarray:
    code = np.zeros((tok.shape[0], tok.shape[1] - k + 1), dtype=np.int64)
    for j in range(k):
        code = code * 4 + tok[:, j : tok.shape[1] - k + 1 + j]
    counts = np.bincount(code.ravel(), minlength=4**k).astype(np.float64)
    return counts / counts.sum()


def token_order_check(full: np.ndarray, a_train: np.ndarray, per_run: dict, out_dir: Path, args, oracle_model=None) -> dict:
    """Verify A,C,G,T = 0,1,2,3 empirically.

    For each of the 24 letter->token mappings, encode the (distinct) unguided FASTA sequences and
    compare their di-/tri-nucleotide spectra with the data tokens (L1 distance). The true mapping
    should minimise the distance. Mean nearest-a_train Hamming is also reported, but it is NOT
    diagnostic here (generated windows are not near-copies of training windows, so nearest-neighbour
    distance at L=50 is dominated by composition).

    Round-trip test (most diagnostic): the guide scores in sample_scores.csv were computed by the
    sampler on the model's own tokens; the oracle scores FASTA letters re-encoded with a mapping.
    Only the mapping that inverts the sampler's decode gives high guide/oracle agreement.
    """
    data_comp = np.bincount(full.reshape(-1), minlength=4) / full.size
    ref2, ref3 = kmer_freq(full.astype(np.int64), 2), kmer_freq(full.astype(np.int64), 3)
    gen_seqs = sorted({s for info in per_run.values() for s in info["unguided"]})
    gen_str = "".join(gen_seqs)
    gen_comp = {ch: gen_str.count(ch) / len(gen_str) for ch in "ACGT"}
    rt_seqs, rt_guide = [], []
    for info in per_run.values():
        for kind in ("guided", "unguided"):
            rt_seqs += info["df"][f"seq_{kind}"].astype(str).str.upper().tolist()
            rt_guide += info["df"][kind].astype(float).tolist()
    rt = pd.DataFrame({"seq": rt_seqs, "guide": rt_guide}).drop_duplicates("seq")
    perm_rows = []
    for perm in itertools.permutations("ACGT"):
        lut = np.full(256, -1, dtype=np.int16)
        for i, ch in enumerate(perm):
            lut[ord(ch)] = i
        tok = np.stack([lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)] for s in gen_seqs]).astype(np.int16)
        nearest, _ = orig.nearest_reference_distances(
            tok, a_train, k_nearest=1, gen_chunk=args.gen_chunk, ref_chunk=args.train_ref_chunk
        )
        perm_rows.append(
            {
                "mapping_0123": "".join(perm),
                "dinuc_L1_vs_data": float(np.abs(kmer_freq(tok.astype(np.int64), 2) - ref2).sum()),
                "trinuc_L1_vs_data": float(np.abs(kmer_freq(tok.astype(np.int64), 3) - ref3).sum()),
                "unguided_mean_nearest_a_train_frac_NOT_DIAGNOSTIC": float(nearest.mean() / tok.shape[1]),
            }
        )
        if oracle_model is not None:
            rt_tok = np.stack([lut[np.frombuffer(q.encode("ascii"), dtype=np.uint8)] for q in rt["seq"]])
            o = oracle_score_tokens(oracle_model, rt_tok)
            perm_rows[-1]["roundtrip_pearson_guide_vs_oracle"] = float(np.corrcoef(o, rt["guide"])[0, 1])
            perm_rows[-1]["roundtrip_mae_guide_vs_oracle"] = float(np.abs(o - rt["guide"].to_numpy()).mean())
    sort_col = "roundtrip_pearson_guide_vs_oracle" if oracle_model is not None else "trinuc_L1_vs_data"
    perm_df = pd.DataFrame(perm_rows).sort_values(sort_col, ascending=oracle_model is None, ignore_index=True)
    perm_df.to_csv(out_dir / "token_order_permutation_check.csv", index=False)
    by_di = perm_df.sort_values("dinuc_L1_vs_data", ignore_index=True)
    by_tri = perm_df.sort_values("trinuc_L1_vs_data", ignore_index=True)
    result = {
        "n_unguided_sequences_used": len(gen_seqs),
        "data_token_fraction_0123": [float(x) for x in data_comp],
        "unguided_fasta_base_fraction_ACGT": gen_comp,
        "best_mapping_by_trinuc_L1": by_tri.iloc[0]["mapping_0123"],
        "trinuc_L1_best_and_second": [float(by_tri.iloc[0]["trinuc_L1_vs_data"]), float(by_tri.iloc[1]["trinuc_L1_vs_data"])],
        "second_best_mapping_by_trinuc_L1": by_tri.iloc[1]["mapping_0123"],
        "best_mapping_by_dinuc_L1": by_di.iloc[0]["mapping_0123"],
        "dinuc_L1_best_and_second": [float(by_di.iloc[0]["dinuc_L1_vs_data"]), float(by_di.iloc[1]["dinuc_L1_vs_data"])],
        "roundtrip_n_distinct_sequences": int(len(rt)),
        "roundtrip_top3": perm_df.head(3).to_dict(orient="records") if oracle_model is not None else None,
        "code_evidence": (
            "data/prepare_yeast_parent_splits.py and scripts/score_yeast_c0_oracle.py use "
            "DNA_TO_INT={'A':0,'C':1,'G':2,'T':3}; DNA-MFM scripts/sample_yeast_c0_guidance_hist.py decodes argmax "
            "with ['A','C','G','T']"
        ),
    }
    return result


def split_sanity(payload: dict, split: dict) -> dict:
    parent = payload["parent_id"].numpy()
    out = {}
    parent_sets = {}
    for name in ("a_train", "a_val", "b_train", "b_val", "test"):
        idx = split[f"{name}_idx"].numpy()
        parent_sets[name] = set(parent[idx].tolist())
        out[name] = {"n_windows": int(len(idx)), "n_parents": len(parent_sets[name])}
    names = list(parent_sets)
    out["parent_sets_pairwise_disjoint"] = all(
        not (parent_sets[x] & parent_sets[y]) for x, y in itertools.combinations(names, 2)
    )
    return out


def subsample_idx(n_total: int, n: int, seed: int) -> np.ndarray:
    # Identical to the draw inside orig.summarize_reference_baseline.
    rng = np.random.default_rng(seed)
    n = min(int(n), n_total)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def size_matched_null(
    test: np.ndarray, n: int, draws: int, seed: int, pair_chunk: int, max_pairwise_n: int, test_nearest_a_train: np.ndarray
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(draws):
        idx = np.sort(rng.choice(test.shape[0], size=n, replace=False))
        sample = test[idx]
        pw = orig.pairwise_generated_distances(sample, chunk=pair_chunk, max_n=max_pairwise_n, seed=seed)
        ng = orig.nearest_generated_distances(sample, chunk=pair_chunk)
        rows.append(
            {
                "draw": d,
                "n": n,
                "pairwise_frac_mean": float(pw.mean() / sample.shape[1]),
                "nearest_gen_frac_mean": float(ng.mean() / sample.shape[1]),
                "exact_unique_frac": float(np.unique(sample, axis=0).shape[0] / n),
                "nearest_a_train_frac_mean": float(test_nearest_a_train[idx].mean() / sample.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def load_oracle(ckpt: Path, repo_root: Path) -> torch.nn.Module:
    # Original: importlib-loaded repo_root/dinko/c0_regressors.py; that file is
    # dmfm.regressors.c0 here (verbatim), so import it directly.
    from dmfm.regressors.c0 import build_c0_regressor

    model = build_c0_regressor("park_cnn")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def oracle_score_tokens(model: torch.nn.Module, tok: np.ndarray) -> np.ndarray:
    """Same as scripts/score_yeast_c0_oracle.py --reverse_complement_average, on CPU."""
    x = torch.nn.functional.one_hot(torch.as_tensor(tok, dtype=torch.long), num_classes=4).float()
    with torch.no_grad():
        pred = 0.5 * (model(x) + model(x.flip(dims=(1,))[..., [3, 2, 1, 0]]))
    return pred.numpy()


def oracle_rescore_check(model: torch.nn.Module, run_dirs: list[Path]) -> dict:
    out = {}
    for run_dir in run_dirs:
        df = pd.read_csv(run_dir / "sample_scores.csv")
        stored = df[["sample_idx"]].merge(pd.read_csv(run_dir / "oracle" / "oracle_per_sample.csv"), on="sample_idx", how="left")
        for kind in ("guided", "unguided"):
            pred = oracle_score_tokens(model, orig.seqs_to_tokens(df[f"seq_{kind}"].tolist()))
            diff = np.abs(pred - stored[f"oracle_{kind}"].to_numpy())
            out[f"{run_dir.name}/{kind}"] = {"max_abs_diff_vs_stored": float(diff.max()), "rescored_mean": float(pred.mean())}
    return out


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, cols: list[str], labels: list[str]) -> str:
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(str(int(row[c])) if c in ("n", "draws", "n_samples") else fmt(row[c]) for c in cols) + " |"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)
    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [repo / r for r in args.run_dirs]

    # ---- Task 1: counts + unguided identity -------------------------------------------------
    t1 = sample_counts_and_identity(run_dirs, out_dir)
    print(t1["counts"].to_string(index=False))
    print(json.dumps(t1["identity"], indent=2))

    # ---- References ---------------------------------------------------------------------------
    payload = torch.load(repo / args.data_pt, map_location="cpu", weights_only=False)
    split = torch.load(repo / args.split_pt, map_location="cpu", weights_only=False)
    full = orig.load_train_tokens(repo / args.data_pt)  # original loader: payload["seqs"] -> int16 [N, L]
    references = {
        "full": full,
        "a_train": full[split["a_train_idx"].numpy()],
        "a_val": full[split["a_val_idx"].numpy()],
        "test": full[split["test_idx"].numpy()],
    }
    sanity = split_sanity(payload, split)
    oracle_model = None
    if args.oracle_ckpt and (repo / args.oracle_ckpt).exists():
        oracle_model = load_oracle(repo / args.oracle_ckpt, repo)
    tok = token_order_check(full, references["a_train"], t1["per_run"], out_dir, args, oracle_model)
    print(json.dumps({"split_sanity": sanity, "token_order": tok}, indent=2))

    # ---- Task 2: generated rows (original summarize_one) --------------------------------------
    summaries, per_sample_parts = [], []
    for run_dir in run_dirs:
        csv_path = run_dir / "sample_scores.csv"
        meta = orig.run_metadata(csv_path)
        df = pd.read_csv(csv_path)
        oracle = df[["sample_idx"]].merge(
            pd.read_csv(run_dir / "oracle" / "oracle_per_sample.csv"), on="sample_idx", how="left", validate="one_to_one"
        )
        for seq_col in ["seq_unguided", "seq_guided"]:
            summary, per_sample = orig.summarize_one(
                df, seq_col, references, meta, csv_path,
                k_nearest_train=args.k_nearest_train, gen_chunk=args.gen_chunk,
                train_ref_chunk=args.train_ref_chunk, pair_chunk=args.pair_chunk,
                max_pairwise_n=args.max_pairwise_n, seed=args.seed,
            )
            kind = "guided" if seq_col == "seq_guided" else "unguided"
            ov = oracle[f"oracle_{kind}"].to_numpy(dtype=float)
            summary["guide_score_mean"] = summary["score_mean"]  # alias: guide regressor (A_train) score
            summary["oracle_score_mean"] = float(ov.mean())
            summary["oracle_score_std"] = float(ov.std(ddof=0))
            summary["oracle_target_mae"] = float(np.abs(ov - float(meta["target_c0"])).mean())
            summary["guide_target_mae"] = float(np.abs(df[kind].to_numpy() - float(meta["target_c0"])).mean())
            per_sample["oracle_score"] = ov
            summaries.append(summary)
            per_sample_parts.append(per_sample)

    # ---- Task 3: real-data baselines (original summarize_reference_baseline) ------------------
    baselines = []
    for set_name, ref_name in [
        ("full_sample", "full"),
        ("a_train_sample", "a_train"),
        ("a_val_sample", "a_val"),
        ("test_sample", "test"),
    ]:
        row = orig.summarize_reference_baseline(
            references[ref_name], set_name=set_name, n=args.baseline_n,
            k_nearest_train=args.k_nearest_train, pair_chunk=args.pair_chunk,
            max_pairwise_n=args.max_pairwise_n, seed=args.seed,
        )
        # Addition (not in old Table 6): distance from the same 1,000 real windows to a_train.
        if ref_name in ("a_val", "test"):
            idx = subsample_idx(references[ref_name].shape[0], args.baseline_n, args.seed)
            sample = references[ref_name][idx]
            near, _ = orig.nearest_reference_distances(
                sample, references["a_train"], k_nearest=args.k_nearest_train,
                gen_chunk=args.gen_chunk, ref_chunk=args.train_ref_chunk,
            )
            row["nearest_a_train_frac_mean"] = float(near.mean() / sample.shape[1])
            row["novel_exact_vs_a_train_frac"] = float((near > 0).mean())
        baselines.append(row)
    baseline_df = pd.DataFrame(baselines)
    heldout = baseline_df[baseline_df["set"] == "test_sample"].iloc[0]

    gen = pd.DataFrame(summaries).sort_values(["target_c0", "set"], ignore_index=True)
    # Same formula as original write_yeast_split_diversity_markdown.add_heldout_comparisons.
    gen["pairwise_vs_test_sample"] = gen["pairwise_gen_frac_mean"] / float(heldout["pairwise_gen_frac_mean"])
    gen["nearest_gen_vs_test_sample"] = gen["nearest_gen_frac_mean"] / float(heldout["nearest_gen_frac_mean"])

    # ---- Addition: size-matched held-out null ------------------------------------------------
    # Per-window nearest-a_train Hamming for every test window (original nearest_reference_distances).
    test_nearest_a_train, _ = orig.nearest_reference_distances(
        references["test"], references["a_train"], k_nearest=1, gen_chunk=args.gen_chunk, ref_chunk=args.train_ref_chunk
    )
    null_parts = []
    for n in sorted(gen["n_samples"].unique()):
        null_parts.append(
            size_matched_null(
                references["test"], int(n), args.size_matched_draws, args.seed, args.pair_chunk,
                args.max_pairwise_n, test_nearest_a_train,
            )
        )
    null = pd.concat(null_parts, ignore_index=True)
    null.to_csv(out_dir / "size_matched_test_null_draws.csv", index=False)
    null_summary = (
        null.groupby("n")
        .agg(
            draws=("draw", "size"),
            pairwise_frac_mean=("pairwise_frac_mean", "mean"),
            pairwise_frac_p025=("pairwise_frac_mean", lambda v: np.quantile(v, 0.025)),
            pairwise_frac_p975=("pairwise_frac_mean", lambda v: np.quantile(v, 0.975)),
            nearest_gen_frac_mean=("nearest_gen_frac_mean", "mean"),
            nearest_gen_frac_p025=("nearest_gen_frac_mean", lambda v: np.quantile(v, 0.025)),
            nearest_gen_frac_p975=("nearest_gen_frac_mean", lambda v: np.quantile(v, 0.975)),
            exact_unique_frac_mean=("exact_unique_frac", "mean"),
            nearest_a_train_frac_mean=("nearest_a_train_frac_mean", "mean"),
            nearest_a_train_frac_p025=("nearest_a_train_frac_mean", lambda v: np.quantile(v, 0.025)),
            nearest_a_train_frac_p975=("nearest_a_train_frac_mean", lambda v: np.quantile(v, 0.975)),
        )
        .reset_index()
    )
    null_summary.to_csv(out_dir / "size_matched_test_null_summary.csv", index=False)
    for i, row in gen.iterrows():
        nd = null[null["n"] == row["n_samples"]]
        ns = null_summary[null_summary["n"] == row["n_samples"]].iloc[0]
        gen.loc[i, "pairwise_vs_test_n_matched"] = row["pairwise_gen_frac_mean"] / ns["pairwise_frac_mean"]
        gen.loc[i, "nearest_gen_vs_test_n_matched"] = row["nearest_gen_frac_mean"] / ns["nearest_gen_frac_mean"]
        gen.loc[i, "pairwise_pct_rank_in_test_null"] = float((nd["pairwise_frac_mean"] <= row["pairwise_gen_frac_mean"]).mean())
        gen.loc[i, "nearest_gen_pct_rank_in_test_null"] = float((nd["nearest_gen_frac_mean"] <= row["nearest_gen_frac_mean"]).mean())
        gen.loc[i, "nearest_a_train_pct_rank_in_test_null"] = float(
            (nd["nearest_a_train_frac_mean"] <= row["nearest_a_train_frac_mean"]).mean()
        )

    # ---- Optional: CPU re-score of stored oracle values --------------------------------------
    oracle_check = None
    if oracle_model is not None:
        oracle_check = oracle_rescore_check(oracle_model, run_dirs)
        print(json.dumps({"oracle_rescore_check": oracle_check}, indent=2))

    # ---- Write --------------------------------------------------------------------------------
    full_summary = pd.concat([baseline_df, gen], ignore_index=True)
    full_summary.to_csv(out_dir / "diversity_summary_full.csv", index=False)
    pd.concat(per_sample_parts, ignore_index=True).to_csv(out_dir / "diversity_per_sample.csv", index=False)

    t7_cols = [
        "target_c0", "set", "n_samples", "guide_score_mean", "oracle_score_mean", "exact_unique_frac",
        "novel_exact_vs_a_train_frac", "novel_exact_vs_test_frac", "nearest_a_train_frac_mean",
        "nearest_test_frac_mean", "nearest_gen_frac_mean", "pairwise_gen_frac_mean",
        "pairwise_vs_test_sample", "nearest_gen_vs_test_sample",
    ]
    t7_labels = [
        "target", "set", "n", "score mean (guide)", "score mean (oracle)", "exact unique", "novel vs train",
        "novel vs test", "nearest train", "nearest test", "nearest generated", "pairwise generated",
        "pairwise / held-out", "nearest gen / held-out",
    ]
    t7x_cols = [
        "target_c0", "set", "n_samples", "pairwise_vs_test_n_matched", "pairwise_pct_rank_in_test_null",
        "nearest_gen_vs_test_n_matched", "nearest_gen_pct_rank_in_test_null", "nearest_a_train_pct_rank_in_test_null",
        "nearest_a_val_frac_mean",
        "nearest_full_frac_mean", "novel_exact_vs_full_frac", "guide_target_mae", "oracle_target_mae",
    ]
    t7x_labels = [
        "target", "set", "n", "pairwise / held-out (n-matched)", "pairwise pct-rank in n-matched test null",
        "nearest gen / held-out (n-matched)", "nearest-gen pct-rank in n-matched test null",
        "nearest-train pct-rank in n-matched test null",
        "nearest a_val", "nearest full", "novel vs full", "guide target MAE", "oracle target MAE",
    ]
    t6_cols = ["set", "n_samples", "exact_unique_frac", "nearest_gen_frac_mean", "pairwise_gen_frac_mean", "pairwise_gen_frac_median"]
    t6_labels = ["set", "n", "exact unique", "nearest generated", "pairwise generated", "pairwise median"]
    t6x_cols = ["set", "n_samples", "nearest_a_train_frac_mean", "novel_exact_vs_a_train_frac"]
    t6x_labels = ["set", "n", "nearest a_train (addition)", "novel vs a_train (addition)"]

    gen[t7_cols].to_csv(out_dir / "table7_generated.csv", index=False)
    gen[t7x_cols].to_csv(out_dir / "table7_additions.csv", index=False)
    baseline_df[t6_cols + ["nearest_a_train_frac_mean", "novel_exact_vs_a_train_frac"]].to_csv(
        out_dir / "table6_baselines.csv", index=False
    )

    ns = null_summary.copy()
    md = [
        "# Tables 6/7 recomputed on the parent-disjoint model (auto-generated)",
        "",
        "Normalized Hamming fraction = mismatches / L (L = 50). train = a_train (generator training parents),",
        "test = held-out test parents, held-out = the 1,000-window test_sample baseline.",
        "",
        "## Table 6: real-data baselines (1,000 windows, seed 0)",
        "",
        md_table(baseline_df, t6_cols, t6_labels),
        "",
        "Addition (not in old Table 6): distance of the same 1,000 real windows to a_train.",
        "",
        md_table(baseline_df[baseline_df["set"].isin(["a_val_sample", "test_sample"])], t6x_cols, t6x_labels),
        "",
        "## Table 7: generated samples",
        "",
        "score mean (guide) = A_train guide regressor, from sample_scores.csv; score mean (oracle) = independent",
        "B_train oracle (reverse-complement averaged), from oracle/oracle_per_sample.csv.",
        "",
        md_table(gen, t7_cols, t7_labels),
        "",
        "## Table 7 additions: size-matched held-out null and extras",
        "",
        md_table(gen, t7x_cols, t7x_labels),
        "",
        "Size-matched test null (random n-window subsets of test, "
        f"{args.size_matched_draws} draws, seed {args.seed}):",
        "",
        md_table(
            ns,
            ["n", "draws", "pairwise_frac_mean", "pairwise_frac_p025", "pairwise_frac_p975",
             "nearest_gen_frac_mean", "nearest_gen_frac_p025", "nearest_gen_frac_p975",
             "nearest_a_train_frac_mean", "nearest_a_train_frac_p025", "nearest_a_train_frac_p975"],
            ["n", "draws", "pairwise mean", "pairwise 2.5%", "pairwise 97.5%", "nearest-gen mean",
             "nearest-gen 2.5%", "nearest-gen 97.5%", "nearest-a_train mean", "nearest-a_train 2.5%",
             "nearest-a_train 97.5%"],
        ),
        "",
    ]
    (out_dir / "tables.md").write_text("\n".join(md))
    meta_out = {
        "args": vars(args),
        "original_code": "DNA-MFM@187fe7b175417a5a42a8759b105707f47bfdea17 scripts/analyze_yeast_c0_diversity.py (blob 5dd21edbc383ae08dfc34f7b70de70ff215704ce), utils/yeast_splits.py (blob 03d6d1faf5519cb7f04de22fb0aebaa187f35297)",
        "split_sanity": sanity,
        "token_order_check": tok,
        "unguided_identity": t1["identity"],
        "oracle_rescore_check": oracle_check,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta_out, indent=2, default=str))
    print("\n".join(md))
    print(f"wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
