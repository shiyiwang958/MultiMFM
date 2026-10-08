#!/usr/bin/env python
"""Aggregate the Figure 4 shards into the small tracked files under ``results/dna/fig4``.

Inputs (gitignored, written by the Slurm arrays)
    ``outputs/dna/fig4/panelA/*.json``      one shard per block of conditioning states
    ``outputs/dna/fig4/panelB/<plan>/*.json`` one shard per (method, budget, repeat)

Outputs (tracked)
    ``results/dna/fig4/panelA_value_error.csv``   mean |V_hat - V_ref| per (method, N)
    ``results/dna/fig4/panelA_per_condition.csv.gz``  the per-condition numbers behind it
    ``results/dna/fig4/panelA_value_error_c0_reward.csv``  robustness variant (cyclizability reward)
    ``results/dna/fig4/panelB_runs.csv``          one row per (method, config, repeat)
    ``results/dna/fig4/panelB_curves.csv``        the plotted curves (mean +- SEM over repeats)
    ``results/dna/fig4/panelB_pilot.csv``         the hyperparameter pilot
    ``results/dna/fig4/summary.json``             headline numbers + what changed vs the PDF
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dmfm import paths

REPO = paths.REPO_ROOT
OUT = paths.RESULTS / "fig4"


def _load(pattern: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pattern.parent.glob(pattern.name))]


# --------------------------------------------------------------------------- panel A


def _curves_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The original producer's summary, verbatim: mean over conditions of |V_est - V_ref|
    per (method, n_samples). The reference is itself a Monte Carlo estimate."""
    sem = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0  # noqa: E731
    return (
        rows.groupby(["method", "n_samples"], as_index=False)
        .agg(
            V_err_mean=("V_err", "mean"),
            V_err_std=("V_err", "std"),
            abs_V_err_mean=("abs_V_err", "mean"),
            abs_V_err_std=("abs_V_err", "std"),
            abs_V_err_sem=("abs_V_err", sem),
            abs_V_err_median=("abs_V_err", "median"),
            n_conditions=("abs_V_err", "size"),
            ref_r_std_mean=("ref_r_std", "mean"),
            est_r_std_mean=("est_r_std", "mean"),
            ref_ess_frac_mean=("ref_ess_frac", "mean"),
            est_ess_frac_mean=("est_ess_frac", "mean"),
        )
        .sort_values(["method", "n_samples"])
    )


def collect_panel_a(shard_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    shards = [json.loads(p.read_text()) for p in sorted(shard_dir.glob("*.json"))]
    if not shards:
        return pd.DataFrame(), pd.DataFrame(), {}
    rows = pd.DataFrame([r for s in shards for r in s["rows"]])
    meta = {
        "n_shards": len(shards),
        "args": shards[0]["args"],
        "checkpoints": shards[0]["checkpoints"],
        "slurm_job_ids": sorted({s.get("slurm_job_id") for s in shards if s.get("slurm_job_id")}),
        "nfe_total": {
            k: int(sum(s["nfe"][k] for s in shards)) for k in ("base", "student", "guide")
        },
        "devices_seen": _devices_seen(shards),
    }
    if len(meta["devices_seen"]) > 1:
        print(
            f"WARNING: {shard_dir} mixes device types {sorted(meta['devices_seen'])}. A plotted "
            "panel must be produced on ONE device type (TF32 + CUDA RNG vs true fp32 + CPU RNG); "
            "rerun the minority shards."
        )
    curves = _curves_from_rows(rows)
    per_cond = rows.drop_duplicates("cond_idx")
    meta["reference_floor"] = {
        "reference": shards[0]["args"].get("reference", "sde"),
        "reference_n": shards[0]["args"].get("sde_ref_n") if shards[0]["args"].get("reference", "sde") == "sde"
        else shards[0]["args"].get("glass_ref_n"),
        "selfsplit_abs_diff_mean": float(per_cond["ref_selfsplit_abs_diff"].mean()),
        "V_ref_mean": float(per_cond["V_ref"].mean()),
        "V_ref_sd": float(per_cond["V_ref"].std(ddof=1)) if len(per_cond) > 1 else 0.0,
        "n_conditions": int(len(per_cond)),
    }
    meta["reward"] = shards[0]["args"].get("reward", "smooth")
    meta["original_producer"] = shards[0].get("original_producer")
    return curves, rows, meta


# --------------------------------------------------------------------------- panel B


def collect_panel_b(
    shard_dir: Path, *, plan: str | None = None, n_outputs: int | None = None, base_seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate one plan's shards; shards with other (plan, n_outputs, base seed) are excluded."""
    shards = [json.loads(p.read_text()) for p in sorted(shard_dir.glob("*.json"))]
    excluded = []
    keep = []
    for s in shards:
        st = s["meta"].get("settings", {})
        ok = (
            (plan is None or st.get("plan", s["meta"].get("plan")) == plan)
            and (n_outputs is None or int(s["n_outputs"]) == int(n_outputs))
            and (base_seed is None or st.get("base_seed", s["meta"].get("base_seed")) == base_seed)
        )
        (keep if ok else excluded).append(s)
    if excluded:
        print(f"WARNING: {len(excluded)} shard(s) in {shard_dir} excluded (settings differ from "
              f"plan={plan} n={n_outputs} seed={base_seed})")
    shards = keep
    if not shards:
        return pd.DataFrame(), pd.DataFrame(), {"n_excluded": len(excluded)}
    runs = pd.DataFrame(
        [
            {
                "method": s["method"],
                "config": s["config"],
                "repeat": s["repeat"],
                "seed": s["seed"],
                "n_outputs": s["n_outputs"],
                "eta": s["nfe"].get("eta", 0.0),
                **{k: v for k, v in s["metrics"].items()},
                **{k: v for k, v in s["nfe"].items() if k != "eta"},
                "elapsed_s": s["meta"].get("elapsed_s"),
                "plan": s["meta"].get("plan"),
                "base_seed": s["meta"].get("base_seed"),
                "slurm_job_id": s["meta"].get("slurm_job_id"),
                **{f"param_{k}": v for k, v in s["config_params"].items()},
            }
            for s in shards
        ]
    )
    curves = (
        runs.groupby(["method", "config"], as_index=False)
        .agg(
            nfe_gen_per_output=("nfe_gen_per_output", "mean"),
            nfe_guide_per_output=("nfe_guide_per_output", "mean"),
            nfe_backward_per_output=("nfe_backward_per_output", "mean"),
            mae_mean=("mae", "mean"),
            mae_sem=("mae", lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0),
            frac05_mean=("frac05", "mean"),
            frac10_mean=("frac10", "mean"),
            oracle_mae_mean=("oracle_mae", "mean"),
            oracle_mae_sem=("oracle_mae", lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0),
            n_repeats=("repeat", "size"),
        )
        .sort_values(["method", "nfe_gen_per_output"])
    )
    meta = {
        "n_shards": len(shards),
        "n_excluded": len(excluded),
        "slurm_job_ids": sorted({s["meta"].get("slurm_job_id") for s in shards if s["meta"].get("slurm_job_id")}),
        "checkpoints": shards[0]["meta"]["checkpoints"],
        "n_steps": shards[0]["meta"]["n_steps"],
        "target_c0": shards[0]["meta"]["target_c0"],
        # every distinct (plan, base seed, n_outputs) among the shards: more than one entry
        # means runs with different settings were mixed in one directory
        "settings_seen": runs.groupby(["plan", "base_seed", "n_outputs"], dropna=False)
        .size()
        .reset_index(name="n_shards")
        .to_dict("records"),
    }
    if len(meta["settings_seen"]) > 1:
        print(f"WARNING: {shard_dir} mixes settings: {meta['settings_seen']}")
    meta["devices_seen"] = _devices_seen(shards)
    if len(meta["devices_seen"]) > 1:
        print(
            f"WARNING: {shard_dir} mixes device types {sorted(meta['devices_seen'])}. A plotted "
            "panel must be produced on ONE device type: CUDA uses TF32 matmuls and a CUDA RNG "
            "stream, CPU uses true fp32 and a CPU RNG, and no A/B run exists to show the "
            "difference is below the reported error bars. Rerun the minority shards."
        )
    return curves, runs, meta


def _devices_seen(shards: list[dict]) -> dict[str, int]:
    """How many shards each device type produced (for the one-device-per-panel rule)."""
    out: dict[str, int] = {}
    for s in shards:
        m = s.get("meta", s)
        d = m.get("device_type") or str(m.get("device", "unknown")).split(":")[0]
        out[d] = out.get(d, 0) + 1
    return out


def _interp_at(curve: pd.DataFrame, nfe: float) -> float:
    """Log-linear interpolation of a method's MAE at a given NFE (NaN outside its range)."""
    g = curve.sort_values("nfe_gen_per_output")
    x, y = g["nfe_gen_per_output"].to_numpy(float), g["mae_mean"].to_numpy(float)
    if len(x) < 2 or nfe < x[0] or nfe > x[-1]:
        return float("nan")
    return float(np.interp(np.log(nfe), np.log(x), y))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panelA-dir", default=str(paths.OUTPUTS / "fig4" / "panelA"))
    p.add_argument("--panelA-c0-dir", default=str(paths.OUTPUTS / "fig4" / "panelA_c0"),
                   help="robustness variant: the cyclizability reward with a GLASS reference")
    p.add_argument("--panelB-dir", default=str(paths.OUTPUTS / "fig4" / "panelB" / "sweep"))
    p.add_argument("--pilot-dir", default=str(paths.OUTPUTS / "fig4" / "panelB" / "pilot"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--sweep-n", type=int, default=100)
    p.add_argument("--sweep-seed", type=int, default=0)
    p.add_argument("--pilot-n", type=int, default=32)
    p.add_argument("--pilot-seed", type=int, default=12345)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "_what": "Figure 4 rerun on the parent-disjoint L=50 models (authors' decision 3, 2026-09-29). "
        "Nothing of the submitted figure survives: see docs/provenance/dna/README.md.",
    }

    a_curves, a_rows, a_meta = collect_panel_a(Path(args.panelA_dir))
    if len(a_curves):
        a_curves.to_csv(out / "panelA_value_error.csv", index=False)
        a_rows.to_csv(out / "panelA_per_condition.csv.gz", index=False, compression="gzip")
        # The published panel's own n_conditions is unrecorded; 8 is the shared default of
        # the original script and its launcher, so any comparison against the digitized
        # curve uses conditions 0-7 only.
        if int(a_rows.cond_idx.max()) >= 8:
            _curves_from_rows(a_rows[a_rows.cond_idx < 8]).to_csv(
                out / "panelA_value_error_first8_conditions.csv", index=False
            )
        summary["panelA"] = a_meta
        summary["panelA"]["headline"] = {
            m: {int(r.n_samples): round(float(r.abs_V_err_mean), 5) for r in g.itertuples()}
            for m, g in a_curves.groupby("method")
        }
        print(f"panel A: {len(a_rows)} rows -> {out/'panelA_value_error.csv'}")

    d_curves, d_rows, d_meta = collect_panel_a(Path(args.panelA_c0_dir))
    if len(d_curves):
        d_curves.to_csv(out / "panelA_value_error_c0_reward.csv", index=False)
        summary["panelA_c0_reward"] = d_meta
        print(f"panel A (c0 reward / GLASS reference): {len(d_rows)} rows")

    b_curves, b_runs, b_meta = collect_panel_b(
        Path(args.panelB_dir), plan="sweep", n_outputs=args.sweep_n, base_seed=args.sweep_seed
    )
    if len(b_curves):
        b_curves.to_csv(out / "panelB_curves.csv", index=False)
        b_runs.drop(columns=[c for c in b_runs.columns if c.startswith("per_sample")]).to_csv(
            out / "panelB_runs.csv", index=False
        )
        summary["panelB"] = b_meta
        dmfm = b_curves[b_curves.method == "dmfm"]
        cross = []
        for _, d in dmfm.iterrows():
            row = {"dmfm_config": d["config"], "nfe": d["nfe_gen_per_output"], "dmfm_mae": d["mae_mean"]}
            for m, g in b_curves[b_curves.method != "dmfm"].groupby("method"):
                row[f"{m}_mae_at_same_nfe"] = _interp_at(g, d["nfe_gen_per_output"])
            cross.append(row)
        summary["panelB"]["matched_nfe_comparison"] = cross
        summary["panelB"]["dmfm_wins_at_lowest_nfe"] = bool(
            len(cross) and all(
                (not np.isfinite(v)) or cross[0]["dmfm_mae"] <= v
                for k, v in cross[0].items()
                if k.endswith("_mae_at_same_nfe")
            )
        )
        print(f"panel B: {len(b_runs)} runs -> {out/'panelB_curves.csv'}")

    p_curves, p_runs, p_meta = collect_panel_b(
        Path(args.pilot_dir), plan="pilot", n_outputs=args.pilot_n, base_seed=args.pilot_seed
    )
    if len(p_runs):
        p_runs.to_csv(out / "panelB_pilot.csv", index=False)
        summary["pilot"] = p_meta
        print(f"pilot: {len(p_runs)} runs -> {out/'panelB_pilot.csv'}")

    cal = Path(args.panelB_dir).parent / "calibration.json"
    if cal.exists():
        (out / "calibration.json").write_text(cal.read_text())
        payload = json.loads(cal.read_text())
        summary["calibration"] = [
            {k: v for k, v in r.items() if k != "scores"} for r in payload["rows"]
        ]

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
