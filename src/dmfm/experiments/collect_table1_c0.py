#!/usr/bin/env python
"""Collect the sharded dMFM Table 1 run: merge shards, oracle-score, write results + Table 1 summary.

Input: ``<run_root>/target_{1p0,m1p0}/shards/<start>_<stop>/`` written by
``scripts/dna/table1_c0_guidance_dmfm.sbatch`` (``dmfm.experiments.sample_c0_guidance_dmfm --paired``).

Per target:
1. check the shards cover sample ids ``0..n-1`` exactly once, merge them into
   ``<run_root>/target_<tag>/merged/`` (``sample_scores.csv``, FASTA, ``summary.json``, plots, ``args.json``);
2. score with the parent-disjoint oracle exactly as Table 1 (``dmfm.experiments.score_c0_oracle``,
   ``park_cnn``, reverse-complement averaged, ``--bootstrap 2000 --seed 0``) into ``merged/oracle/``;
3. copy ``args.json``, ``summary.json``, ``sample_scores.csv`` and ``oracle/*.csv`` to
   ``<results_dir>/target_<tag>/`` (same layout as ``results/dna/c0_guidance/n1000``).

Then write ``--summary_json`` in the schema of ``results/dna/c0_guidance/table1_n1000_summary.json``:
Table 1 cells with percentile-bootstrap CIs of the mean over pairs (2,000 resamples,
``numpy.random.default_rng(0)`` re-seeded per statistic, ``rng.choice``, float32 oracle scores --
the method of ``score_c0_oracle`` and of the submitted Table 1), formatted cells, drop-in LaTeX rows,
the Sec. 3.1 in-text numbers, and the GLASS n=1000 values as provenance (not in the paper).

    python -m dmfm.experiments.collect_table1_c0 --run_root outputs/dna/table1_dmfm_n1000 \\
        --results_dir results/dna/c0_guidance/dmfm_n1000 \\
        --summary_json results/dna/c0_guidance/table1_dmfm_n1000_summary.json --jobs 12345
"""

from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from dmfm import paths

TARGETS = {"+1": ("1p0", 1.0), "-1": ("m1p0", -1.0)}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", default=str(paths.OUTPUTS / "table1_dmfm_n1000"))
    p.add_argument("--results_dir", default=str(paths.RESULTS / "c0_guidance" / "dmfm_n1000"))
    p.add_argument("--summary_json", default=str(paths.RESULTS / "c0_guidance" / "table1_dmfm_n1000_summary.json"))
    p.add_argument("--glass_summary_json", default=str(paths.RESULTS / "c0_guidance" / "table1_n1000_summary.json"),
                   help="Existing GLASS n=1000 summary, copied in as provenance.")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--oracle_ckpt", default=str(paths.c0_oracle()))
    p.add_argument("--device", default=None, help="Oracle device (default: cuda if available, else cpu).")
    p.add_argument("--jobs", nargs="*", default=[], help="Slurm job ids of the generation run (recorded).")
    p.add_argument("--glass_run_root", default=None,
                   help="Tree of the GLASS arm run under the SAME scale rule (matched baseline). Its cells and the "
                        "per-sample paired difference go into the summary; without it the summary has the dMFM arm only.")
    p.add_argument("--glass_results_dir", default=None, help="Where to copy the GLASS arm's small results.")
    p.add_argument("--withdrawn_run_root", default=None,
                   help="A superseded run to record (not print) as withdrawn, e.g. job 49277977's tree.")
    p.add_argument("--allow_partial", action="store_true",
                   help="Collect whatever shards have finished (n < --n_samples); the summary is marked PARTIAL.")
    return p.parse_args(argv)


def bootstrap_choice(values, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """``score_c0_oracle.bootstrap_mean``: percentile bootstrap of the mean, default_rng(seed).choice."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(values), size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def table1_cells(per_sample: pd.DataFrame, target: float) -> dict:
    u = per_sample["oracle_unguided"].to_numpy(np.float32)
    g = per_sample["oracle_guided"].to_numpy(np.float32)
    t = np.float32(target)
    eu, eg = np.abs(u - t), np.abs(g - t)
    imp = eu - eg
    return {"n_pairs": int(len(u)),
            "oracle_mean_unguided": float(u.mean()), "oracle_mean_guided": float(g.mean()),
            "unguided_mae": float(eu.mean()), "unguided_mae_ci": bootstrap_choice(eu),
            "guided_mae": float(eg.mean()), "guided_mae_ci": bootstrap_choice(eg),
            "paired_improvement": float(imp.mean()), "paired_improvement_ci": bootstrap_choice(imp),
            "frac_pairs_improved": float((imp > 0).mean())}


def merge_target(root: Path, tag: str, n: int, allow_partial: bool = False) -> Path:
    from dmfm.experiments.sample_c0_guidance_dmfm import write_paired_outputs

    shard_dirs = sorted((root / f"target_{tag}" / "shards").glob("*_*"))
    done = [d for d in shard_dirs if (d / "sample_scores.csv").exists()]
    if not done:
        if allow_partial:
            print(f"target {tag}: no finished shards yet, skipping (partial collection)", flush=True)
            return None
        raise SystemExit(f"no finished shards under {root / f'target_{tag}' / 'shards'}")
    df = pd.concat([pd.read_csv(d / "sample_scores.csv") for d in done], ignore_index=True)
    df = df.sort_values("sample_idx").reset_index(drop=True)
    ids = df["sample_idx"].to_numpy()
    if len(set(ids.tolist())) != len(ids):
        raise SystemExit(f"target {tag}: duplicate sample ids across shards")
    if allow_partial and len(ids) < n:
        print(f"target {tag}: PARTIAL collection, {len(ids)}/{n} pairs from {len(done)} shards", flush=True)
    elif len(ids) != n or not np.array_equal(ids, np.arange(n)):
        missing = sorted(set(range(n)) - set(ids.tolist()))
        raise SystemExit(f"target {tag}: shards cover {len(set(ids.tolist()))}/{n} ids "
                         f"(missing e.g. {missing[:5]}; duplicates {len(ids) - len(set(ids.tolist()))}); "
                         f"resubmit the sbatch (finished shards are skipped)")
    args = json.loads((done[0] / "args.json").read_text())
    for d in done[1:]:
        other = json.loads((d / "args.json").read_text())
        diff = {k for k in set(args) | set(other) if k not in ("sample_start", "sample_stop", "out_dir")
                and args.get(k) != other.get(k)}
        if diff:
            raise SystemExit(f"shard {d} has different args: {sorted(diff)}")
    merged = root / f"target_{tag}" / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    args.update(sample_start=0, sample_stop=None, out_dir=str(merged), shards=[d.name for d in done])
    envs = {d.name: (json.loads((d / "run_env.json").read_text()) if (d / "run_env.json").exists() else None) for d in done}
    (merged / "run_env.json").write_text(json.dumps(envs, indent=2, sort_keys=True))
    (merged / "args.json").write_text(json.dumps(args, indent=2, sort_keys=True))
    write_paired_outputs(Namespace(**args), merged, df)
    return merged


def collect_arm(cli, run_root: Path, results_dir: Path, device: str) -> tuple[dict, dict]:
    """Merge one arm's shards, oracle-score them as Table 1 does, copy the small results, return cells."""
    from dmfm.experiments import score_c0_oracle

    cells, run_args = {}, {}
    for key, (tag, target) in TARGETS.items():
        merged = merge_target(run_root, tag, cli.n_samples, allow_partial=cli.allow_partial)
        if merged is None:
            continue
        score_c0_oracle.main(["--oracle_ckpt", cli.oracle_ckpt, "--oracle_model_type", "park_cnn",
                              "--run_glob", str(merged / "sample_scores.csv"), "--out_dir", str(merged / "oracle"),
                              "--batch_size", "512", "--bootstrap", "2000", "--seed", "0",
                              "--reverse_complement_average", "--device", device])
        dest = results_dir / f"target_{tag}"
        (dest / "oracle").mkdir(parents=True, exist_ok=True)
        for name in ("args.json", "summary.json", "sample_scores.csv", "run_env.json"):
            shutil.copy2(merged / name, dest / name)
        for name in ("oracle_per_sample.csv", "oracle_summary.csv"):
            shutil.copy2(merged / "oracle" / name, dest / "oracle" / name)
        per = pd.read_csv(dest / "oracle" / "oracle_per_sample.csv")
        c = table1_cells(per, target)
        stored = pd.read_csv(dest / "oracle" / "oracle_summary.csv").set_index("set")
        c["_max_abs_diff_vs_stored_oracle_summary"] = float(np.nanmax(np.abs(np.array([
            c["oracle_mean_unguided"] - stored.loc["unguided", "oracle_score_mean"],
            c["oracle_mean_guided"] - stored.loc["guided", "oracle_score_mean"],
            c["unguided_mae"] - stored.loc["unguided", "oracle_target_mae"],
            c["guided_mae"] - stored.loc["guided", "oracle_target_mae"],
            c["paired_improvement"] - stored.loc["paired_guidance_effect", "oracle_target_mae"],
            *(np.array(c["guided_mae_ci"]) - stored.loc["guided", ["oracle_target_mae_ci95_low", "oracle_target_mae_ci95_high"]].to_numpy(float)),
        ]))))
        # The unguided branch does not depend on the posterior sampler: with the same per-sample
        # noise it must reproduce the unguided sequences of the stored GLASS n=1000 run.
        ref_csv = paths.RESULTS / "c0_guidance" / "n1000" / f"target_{tag}" / "sample_scores.csv"
        if ref_csv.exists():
            new = pd.read_csv(dest / "sample_scores.csv").set_index("sample_idx").sort_index()
            ref = pd.read_csv(ref_csv).set_index("sample_idx").sort_index()
            shared = new.index.intersection(ref.index)
            c["_unguided_vs_stored_glass_n1000_run"] = {
                "n_shared": int(len(shared)),
                "identical_unguided_sequences": int((new.loc[shared, "seq_unguided"] == ref.loc[shared, "seq_unguided"]).sum()),
                "max_abs_diff_unguided_guide_score": float(np.abs(new.loc[shared, "unguided"] - ref.loc[shared, "unguided"]).max()),
            }
        g = json.loads((dest / "summary.json").read_text())
        c["guide_scored"] = {"unguided_mean": g["unguided"]["mean"], "guided_mean": g["guided"]["mean"],
                             "unguided_mae": g["unguided"]["mean_abs_to_target"],
                             "guided_mae": g["guided"]["mean_abs_to_target"],
                             "guided_frac_within_0.10": g["guided"]["frac_within_0.10"]}
        envs = json.loads((dest / "run_env.json").read_text())
        c["_execution"] = {
            "devices": sorted({str(e.get("device_name") if e else None) for e in envs.values()}),
            "slurm_accounts": sorted({str(e.get("slurm_job_account") if e else None) for e in envs.values()}),
            "slurm_partitions": sorted({str(e.get("slurm_job_partition") if e else None) for e in envs.values()}),
            "slurm_jobs": sorted({str(e.get("slurm_array_job_id") or e.get("slurm_job_id")) if e else "None" for e in envs.values()}),
            "tf32": sorted({bool(e.get("tf32_matmul")) if e else None for e in envs.values()}, key=str),
            "oracle_device": device,
        }
        c["_shards_included"] = list(json.loads((dest / "args.json").read_text()).get("shards", []))
        cells[key] = c
        run_args[key] = json.loads((dest / "args.json").read_text())
    return cells, run_args


def paired_difference(dir_a: Path, dir_b: Path, tag: str, target: float) -> dict:
    """Per-sample paired difference in guided oracle error, arm A minus arm B (same initial noise)."""
    a = pd.read_csv(dir_a / f"target_{tag}" / "oracle" / "oracle_per_sample.csv").set_index("sample_idx").sort_index()
    b = pd.read_csv(dir_b / f"target_{tag}" / "oracle" / "oracle_per_sample.csv").set_index("sample_idx").sort_index()
    shared = a.index.intersection(b.index)
    t = np.float32(target)
    d = (np.abs(a.loc[shared, "oracle_guided"].to_numpy(np.float32) - t)
         - np.abs(b.loc[shared, "oracle_guided"].to_numpy(np.float32) - t))
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"n_pairs": int(len(shared)), "mean": float(d.mean()), "sem": sem,
            "ci95_bootstrap": bootstrap_choice(d), "frac_a_better": float((d < 0).mean()),
            "unguided_identical_across_arms": bool(
                (a.loc[shared, "oracle_unguided"].to_numpy() == b.loc[shared, "oracle_unguided"].to_numpy()).all())}


def main(argv=None) -> None:
    cli = parse_args(argv)
    import torch

    device = cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    root, res = Path(cli.run_root), Path(cli.results_dir)
    cells, run_args = collect_arm(cli, root, res, device)
    glass_cells, glass_args, paired = {}, {}, {}
    if cli.glass_run_root:
        gres = Path(cli.glass_results_dir or (res.parent / (res.name + "_glass_matched")))
        glass_cells, glass_args = collect_arm(cli, Path(cli.glass_run_root), gres, device)
        paired = {k: paired_difference(res, gres, TARGETS[k][0], TARGETS[k][1])
                  for k in TARGETS if k in cells and k in glass_cells}
    fmt = lambda m, ci: f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    tex = lambda m, ci: f"${m:.3f}\\,[{ci[0]:.3f},\\,{ci[1]:.3f}]$"
    r = lambda x, d=2: float(f"{x:.{d}f}")
    formatted = {k: {"target": k, "mean_oracle_score_unguided": f"{v['oracle_mean_unguided']:.3f}",
                     "mean_oracle_score_guided": f"{v['oracle_mean_guided']:.3f}",
                     "unguided_mae": fmt(v["unguided_mae"], v["unguided_mae_ci"]),
                     "guided_mae": fmt(v["guided_mae"], v["guided_mae_ci"]),
                     "paired_improvement": fmt(v["paired_improvement"], v["paired_improvement_ci"])}
                 for k, v in cells.items()}
    latex = {k: (f"${k}$ & ${v['oracle_mean_unguided']:.3f}$ & ${v['oracle_mean_guided']:.3f}$ & "
                 f"{tex(v['unguided_mae'], v['unguided_mae_ci'])} & {tex(v['guided_mae'], v['guided_mae_ci'])} & "
                 f"{tex(v['paired_improvement'], v['paired_improvement_ci'])} \\\\")
             for k, v in cells.items()}
    glass_path = Path(cli.glass_summary_json)
    glass = json.loads(glass_path.read_text()) if glass_path.exists() else {}
    oracle_mae = glass.get("in_text_sec3_1", {}).get("oracle_own_heldout_mae", 0.16)
    p1, m1 = cells.get("+1"), cells.get("-1")
    suggested = ((f"Guidance reduces oracle MAE from {p1['unguided_mae']:.2f} to {p1['guided_mae']:.2f} for $y^\\star=+1$ "
                  f"and from {m1['unguided_mae']:.2f} to {m1['guided_mae']:.2f} for $y^\\star=-1$. ... The guided errors of "
                  f"{p1['guided_mae']:.2f} and {m1['guided_mae']:.2f} compare with the oracle's own MAE of {oracle_mae:.2f} "
                  f"against measured C0 on the held-out parents.")
                 if p1 and m1 else "incomplete: not every target has finished shards yet")
    a = run_args[next(iter(run_args))]
    norm = bool(a.get("grad_normalize"))
    scale_rule = ("scale-free: the value gradient is rescaled to exactly grad_clip "
                  f"({a['grad_clip']}) at every guided step" if norm else
                  f"stock Table 1 rule: the gradient is clipped down to grad_clip ({a['grad_clip']}) only")
    gfmt = {k: {"target": k, "guided_mae": fmt(v["guided_mae"], v["guided_mae_ci"]),
                "unguided_mae": fmt(v["unguided_mae"], v["unguided_mae_ci"]),
                "paired_improvement": fmt(v["paired_improvement"], v["paired_improvement_ci"])}
            for k, v in glass_cells.items()}
    partial = (len(cells) < len(TARGETS)
               or any(c["n_pairs"] < cli.n_samples for c in cells.values())
               or (bool(glass_cells) and (len(glass_cells) < len(TARGETS)
                                          or any(c["n_pairs"] < cli.n_samples for c in glass_cells.values()))))
    n_of = lambda d, k: (d[k]["n_pairs"] if k in d else 0)
    summary = {
        "_status": (f"PRELIMINARY / PARTIAL: dMFM n = {n_of(cells, '+1')} (+1) / {n_of(cells, '-1')} (-1), matched GLASS "
                    f"n = {n_of(glass_cells, '+1')} (+1) / {n_of(glass_cells, '-1')} (-1), of {cli.n_samples} pairs per "
                    "target. Superseded by the full n = 1000 run; rerun the collector when the remaining shards finish. "
                    "Do not cite as Table 1.") if partial else "complete",
        "_what": ("Table 1 (Sec. 3.1) for the n = 1000 dMFM-guided run (authors' decision 2026-09-29: Table 1 reports "
                  "dMFM guidance). The dMFM posterior is the 4-step ESD student checkpoints/dna/dmfm_4step/L50 composed "
                  "in 4 steps, the artifact distilled and checkpoint-selected for that jump (Table 18's pairing), and "
                  "both arms use the same scale-free guidance rule. The GLASS arm under the identical rule is in "
                  "'matched_glass_arm' and the per-sample paired difference in 'dmfm_vs_glass_matched'; that pair, not "
                  "the dMFM column alone, is the like-for-like comparison. "
                  "Built by dmfm.experiments.collect_table1_c0 (scripts/dna/collect_table1_dmfm.sh)."),
        "_scale_rule": (f"{scale_rule}. Table 1's constants (guidance_frac {a['guidance_frac']}, coeff_cap "
                        f"{a['coeff_cap']}, grad_clip {a['grad_clip']}) were tuned to the GLASS-4 gradient magnitude "
                        "(mean norm ~1-3, clip binding on 1-5% of guided steps). Every dMFM posterior's gradient there "
                        "has norm 5-125 and binds on 6-52% of steps, so under the stock rule swapping the posterior "
                        "multiplies the effective guidance strength 5-12x and the comparison measures drive, not "
                        "posterior quality. The scale-free rule is applied identically to both arms; nothing is tuned "
                        "per arm. See docs/dmfm_glass_parity.md."),
        "_partial_sampling": (
            "Shards tile sample_idx in contiguous blocks of 200 (0-199, 200-399, ...), but sample_idx indexes "
            "independent seeded draws in unconditional generation -- initial noise seed+i, MC pool seed+1e6+i, no data "
            "ordering -- so the ids are exchangeable and a finished subset of shards is an i.i.d. subsample of the same "
            "pool, not a biased slice. A partial number is therefore unbiased, only wider: its bootstrap CI at reduced n "
            "is the honest uncertainty. Each arm/target's included shards are listed as _shards_included."
            if True else ""),
        "_reward": (
            f"The implemented terminal reward is reward_scale * -0.5 ((f_guide(x1) - y*) / reward_sigma)^2 = "
            f"{float(a['reward_scale']) * 0.5 / float(a['reward_sigma']) ** 2:.3f} * -(f_guide(x1) - y*)^2 at the "
            f"settings used here (reward_sigma {a['reward_sigma']}, reward_scale {a['reward_scale']}), NOT eq (17)'s "
            "-(f - y*)^2. Inside the value V(x) = log mean_j exp(r_j) that constant is an inverse temperature: it "
            "reweights the MC average over posterior samples and so changes the gradient direction, which means it "
            "cannot be absorbed into the tuned guidance strength. The same constant appears in every DNA C0 args.json, "
            "including the submitted n=32 pilot, so this is a disclosure gap in eq (17) rather than a defect in these "
            "runs; the paper should state the effective inverse temperature."),
        "_caveat_posterior_accuracy": (
            "The guidance signal rides on the coarse posterior's discretisation error, and the strength constants are "
            "calibrated against that error. A *more accurate* posterior guides less, not more: GLASS with 32 Euler "
            "steps leaves the guided oracle MAE at the unguided value (1.1905 vs 1.1907 at y*=+1; 0.8098 vs 0.8098 at "
            "y*=-1), and the well-integrated reference gradient (GLASS-25 RK4 to 0.999) has norm ~0.15 against 3-125 "
            "for every estimator actually used for guidance. So 'use a better posterior sampler' is not, in this "
            "setup, a route to better steering, and the constants cannot be transported between samplers without "
            "renormalisation. This is a property of the experiment as published, not of dMFM "
            "(docs/dmfm_glass_parity.md section 2)."),
        "_withdrawn_runs": {
            "slurm_49277977": {
                "tree": cli.withdrawn_run_root or "outputs/dna/table1_dmfm_n1000",
                "what": "first n=1000 'Table 1 with dMFM' run: composed 4 flow-map steps of the DIAGONAL student "
                        "checkpoints/dna/dmfm/L50, which the rest of the paper uses as a ONE-step map, and used the "
                        "stock clip-down rule, so it was both mis-paired and 5-12x over-driven.",
                "status": "WITHDRAWN, do not print as Table 1 (REPRODUCIBILITY_PLAN.md section 7 item 8); complete "
                          "n=1000 arrays are kept on disk as provenance.",
                "guided_oracle_mae_n1000": {"+1": 0.6103, "-1": 0.4081},
                "paired_improvement_n1000": {"+1": 0.5634, "-1": 0.4244},
                "guided_oracle_mae_at_n64_parity_run": {"+1": 0.632, "-1": 0.380},
                "_note": "oracle-scored by dmfm.experiments.collect_table1_c0 on its complete n=1000 arrays "
                         "(H100, job 49277977); its 1000/1000 unguided sequences per target are identical to the "
                         "stored GLASS n=1000 run, so only the posterior and the guidance scale differ.",
            }},
        "_ci_method": "Percentile bootstrap of the mean over pairs: 2,000 resamples, numpy default_rng(seed=0) "
                      "re-seeded for each statistic, rng.choice with replacement, 2.5/97.5% quantiles; float32 "
                      "oracle scores (exactly dmfm.experiments.score_c0_oracle --bootstrap 2000 --seed 0, the method "
                      "behind the submitted Table 1).",
        "_oracle": "park_cnn oracle checkpoints/dna/c0/oracle/best_state.pt (= workdir/yeast_parent_c0_oracle_2026-07-26), "
                   f"reverse-complement averaged, scored on {device}; trained on b_train parents (disjoint from the guide's a_train).",
        "_sampler": ("dMFM-posterior value-gradient guidance on the parent-disjoint L=50 base flow: value gradients from "
                     f"{a['nfe_value']} composed flow-map steps of the L=50 dMFM ({Path(a['student_ckpt']).name}), "
                     f"mc {a['mc']} frozen per-sample pool, nfe_traj {a['nfe_sample']}, guidance_frac {a['guidance_frac']}, "
                     f"reward_sigma {a['reward_sigma']}, reward_scale {a['reward_scale']}, t in [{a['guide_t_start']}, "
                     f"{a['guide_t_end']}], t_max {a['t_max']}, seed {a['seed']}, batch {a['batch_size']}, noise-paired "
                     "guided/unguided samples. Identical to the GLASS Table 1 protocol except for the posterior sampler "
                     "(dmfm.experiments.sample_c0_guidance_dmfm --paired --posterior dmfm)."),
        "source_runs": {"dmfm_n1000": {"+1": f"{cli.run_root}/target_1p0/merged", "-1": f"{cli.run_root}/target_m1p0/merged",
                                       "slurm_jobs": [int(j) if str(j).isdigit() else j for j in cli.jobs],
                                       "wrapper": "scripts/dna/table1_c0_guidance_dmfm.sbatch",
                                       "results": f"{paths.rel(res)}/target_{{1p0,m1p0}}/"},
                        "glass_n1000_not_in_paper": glass.get("source_runs", {}).get("n1000")},
        "n1000": cells,
        "table1_n1000_formatted": formatted,
        "table1_n1000_latex_rows": latex,
        "matched_glass_arm": ({"_what": "GLASS-4 posterior on the base DFM under the SAME scale rule as the dMFM arm "
                                        "above: the like-for-like baseline. Not the published Table 1 numbers.",
                               "run_root": cli.glass_run_root, "n1000": glass_cells, "formatted": gfmt,
                               "sampler": ("dmfm.experiments.sample_c0_guidance_dmfm --paired --posterior glass "
                                           f"--nfe_value {a['nfe_value']}")} if glass_cells else None),
        "dmfm_vs_glass_matched": ({"_what": "Per-sample paired difference in guided oracle error, dMFM minus the "
                                            "matched GLASS arm (same initial noise per sample id). Positive means dMFM "
                                            "is worse; |mean| < ~2 SEM means the two agree.",
                                   "by_target": paired} if paired else None),
        "in_text_sec3_1": {
            "oracle_mae_unguided_to_guided": {k: [r(v["unguided_mae"]), r(v["guided_mae"])] for k, v in cells.items()},
            "oracle_mae_unguided_to_guided_3dp": {k: [r(v["unguided_mae"], 3), r(v["guided_mae"], 3)] for k, v in cells.items()},
            "oracle_own_heldout_mae": oracle_mae,
            "n_pairs_per_target": {k: v["n_pairs"] for k, v in cells.items()},
            "frac_pairs_improved": {k: v["frac_pairs_improved"] for k, v in cells.items()},
            "submitted_text": glass.get("in_text_sec3_1", {}).get("submitted_text"),
            "suggested_sentences": suggested,
        },
        "glass_n1000_provenance_not_in_paper": {
            "n1000": glass.get("n1000"), "table1_n1000_formatted": glass.get("table1_n1000_formatted"),
            "in_text_sec3_1": glass.get("in_text_sec3_1")},
        "submitted_n32": glass.get("submitted_n32"),
        "fig3_caption_and_table11": glass.get("fig3_caption_and_table11"),
    }
    out = Path(cli.summary_json)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"scale_rule": scale_rule,
                      "table1_dmfm_n1000_formatted": formatted,
                      "matched_glass_arm_formatted": gfmt or None,
                      "dmfm_vs_glass_matched": {k: {"mean": v["mean"], "sem": v["sem"]} for k, v in paired.items()} or None,
                      "latex_rows": latex,
                      "in_text": summary["in_text_sec3_1"]["oracle_mae_unguided_to_guided_3dp"],
                      "frac_pairs_improved": summary["in_text_sec3_1"]["frac_pairs_improved"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
