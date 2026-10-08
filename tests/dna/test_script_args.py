"""Dry-run the DNA experiment wrappers and compare their arguments with the original runs.

Each ``scripts/dna/*.sbatch`` is executed with ``PY=scripts/dna/_dryrun.py``: every
``python -m dmfm.experiments.<name> ...`` call is parsed by that module's own parser (so a
typo or unknown flag fails here) and recorded instead of run. The parsed arguments are then
compared with the arguments stored by the original runs (copies under ``results/dna`` and
``checkpoints/dna``). Path-valued arguments (checkpoints, data, output dirs) are compared by
the checkpoint *file name* only, since the directory layout changed.

The training wrappers are covered by ``test_dmfm_port.test_training_wrappers_reproduce_original_args``.

    source scripts/env.sh
    python tests/dna/test_script_args.py        # ~1 min, login node is fine
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dmfm import paths

REPO = paths.REPO_ROOT
RES = REPO / "results" / "dna"
SHIM = REPO / "scripts" / "dna" / "_dryrun.py"
PATH_KEYS = {"ckpt", "c0_ckpt", "output_dir", "out_dir", "oracle_ckpt", "run_glob", "data_pt", "split_pt",
             "probe_data_dir", "teacher_ckpt", "dmfm_ckpt", "device"}


def dry_run(script: str, **env) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="dmfm_dry_") as tmp:
        out = Path(tmp) / "calls.jsonl"
        full_env = dict(os.environ, PY=f"python {SHIM}", DMFM_DRYRUN_OUT=str(out), MULTIMFM_ROOT=str(REPO),
                        OUT_DIR=str(Path(tmp) / "o"), OUT=str(Path(tmp) / "o"), OUTPUT_DIR=str(Path(tmp) / "o"),
                        OUT_ROOT=str(Path(tmp) / "o"), **{k: str(v) for k, v in env.items()})
        proc = subprocess.run(["bash", str(REPO / "scripts" / "dna" / script)], env=full_env, cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{script} {env}: rc={proc.returncode}\n{proc.stderr[-2000:]}"
        return [json.loads(line) for line in out.read_text().splitlines()]


def diff(new: dict, old: dict, *, ignore=(), only_old_keys=True) -> dict:
    keys = set(old) if only_old_keys else set(new) | set(old)
    out = {}
    for k in sorted(keys):
        if k in PATH_KEYS or k in ignore:
            continue
        a, b = new.get(k, "<missing>"), old.get(k, "<missing>")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if abs(float(a) - float(b)) <= 1e-12:
                continue
        if a != b:
            out[k] = (a, b)
    return out


def _report(name: str, d: dict) -> None:
    print(f"  {name}: {'identical' if not d else d}")
    assert not d, (name, d)


def test_table1_c0_guidance():
    """Table 1 (n=1000): sampler + oracle flags equal the stored n=1000 run args.json."""
    for target, tag in ((1.0, "1p0"), (-1.0, "m1p0")):
        calls = dry_run("table1_c0_guidance.sbatch", TARGET_C0=target)
        assert [c["module"] for c in calls] == ["dmfm.experiments.sample_c0_guidance",
                                                 "dmfm.experiments.score_c0_oracle"]
        sample, oracle = calls[0]["parsed"], calls[1]["parsed"]
        old = json.loads((RES / "c0_guidance" / "n1000" / f"target_{tag}" / "args.json").read_text())
        # keys main() adds after parsing (copied from the checkpoint's model_cfg)
        added = {"gaussian_beta_schedule", "gaussian_beta_table_path", "flow_temp"}
        _report(f"table1 target {target} sampler", diff(sample, old, ignore=added))
        assert Path(sample["ckpt"]).name == "best.pt" and "/L50/" in sample["ckpt"]
        assert Path(sample["c0_ckpt"]).parent.name == "guide"
        # the original n=1000 oracle command (scratch_dfm_recon_20260925/run_n1000.sbatch)
        want = {"oracle_model_type": "park_cnn", "batch_size": 512, "bootstrap": 2000, "seed": 0,
                "reverse_complement_average": True}
        _report(f"table1 target {target} oracle", diff(oracle, want))
        assert Path(oracle["oracle_ckpt"]).parent.name == "oracle"


def test_table1_dmfm_matches_table1_settings():
    """table1_c0_guidance_dmfm.sbatch (10-task array): every setting equals the GLASS Table 1 run;
    only the posterior sampler differs; shards tile ids 0..999 in batch-aligned blocks."""
    covered = {1.0: [], -1.0: []}
    for task in range(10):
        calls = [c for c in dry_run("table1_c0_guidance_dmfm.sbatch", SLURM_ARRAY_TASK_ID=task) if c.get("module")]
        assert [c["module"] for c in calls] == ["dmfm.experiments.sample_c0_guidance_dmfm"], calls
        p = calls[0]["parsed"]
        tag = "1p0" if p["target_c0"] == 1.0 else "m1p0"
        old = json.loads((RES / "c0_guidance" / "n1000" / f"target_{tag}" / "args.json").read_text())
        same = {k: p[k] for k in ("seed", "n_samples", "batch_size", "target_c0", "target_c0_second",
                                  "reward_sigma", "reward_scale", "mc", "mc_chunk", "nfe_value", "t_max",
                                  "guide_t_start", "guide_t_end", "guidance_frac", "coeff_cap", "grad_clip")}
        same["nfe_traj"] = p["nfe_sample"]
        _report(f"table1-dmfm task {task} vs GLASS run", diff(same, {k: old[k] for k in same}))
        assert p["paired"] and p["posterior"] == "dmfm"
        # The student must be the artifact distilled for --nfe_value composed jumps: the 4-step
        # ESD student for the 4 steps Table 1 uses (docs/dmfm_glass_parity.md). LEGACY_DIAG_STUDENT=1
        # restores the diagonal student of job 49277977, which needs --allow_gap_mismatch.
        assert Path(p["student_ckpt"]).resolve() == paths.dmfm_ckpt_for_steps(50, p["nfe_value"]).resolve()
        assert not p["allow_gap_mismatch"]
        # plan section 7 item 8 also mandates the scale-free gradient rule, on both arms.
        assert p["grad_normalize"] is True
        assert p["sample_start"] % p["batch_size"] == 0
        covered[p["target_c0"]].extend(range(p["sample_start"], p["sample_stop"]))
    for target, ids in covered.items():
        assert sorted(ids) == list(range(1000)), (target, len(ids))
    # oracle scoring happens in the collector with the Table 1 oracle flags
    import inspect
    from dmfm.experiments import collect_table1_c0
    # Whole module, not main(): the oracle call lives in collect_arm(), which main()
    # invokes once per arm. Grepping main() alone made this assert on where the call
    # sits rather than on whether it is made.
    src = inspect.getsource(collect_table1_c0)
    for flag in ('"park_cnn"', '"--bootstrap", "2000"', '"--seed", "0"', '"--reverse_complement_average"'):
        assert flag in src, flag


def test_table1_dmfm_guidance_rule_defaults_safe():
    """The superseded clip-down rule must never be reachable by omission (plan section 7 item 8)."""
    default = dry_run("table1_c0_guidance_dmfm.sbatch", SLURM_ARRAY_TASK_ID=0)[0]["parsed"]
    assert default["grad_normalize"] is True and "/dmfm_4step/" in default["student_ckpt"]
    glass = dry_run("table1_c0_guidance_dmfm.sbatch", SLURM_ARRAY_TASK_ID=0, POSTERIOR="glass")[0]["parsed"]
    assert glass["grad_normalize"] is True and glass["posterior"] == "glass"  # same rule on both arms
    # the old behaviour stays reachable, but only when asked for explicitly
    legacy = dry_run("table1_c0_guidance_dmfm.sbatch", SLURM_ARRAY_TASK_ID=0, LEGACY_DIAG_STUDENT="1")[0]["parsed"]
    assert legacy["grad_normalize"] is False and "/dmfm/L50/" in legacy["student_ckpt"]
    assert legacy["allow_gap_mismatch"] is True          # job 49277977 exactly
    optout = dry_run("table1_c0_guidance_dmfm.sbatch", SLURM_ARRAY_TASK_ID=0, GRAD_NORMALIZE="0")[0]["parsed"]
    assert optout["grad_normalize"] is False and "/dmfm_4step/" in optout["student_ckpt"]
    print("  defaults: scale-free + 4-step student; clip-down only via GRAD_NORMALIZE=0 or LEGACY_DIAG_STUDENT=1")


def test_c0_regressors():
    calls = dry_run("train_c0_regressors.sbatch")
    for call, role in zip(calls, ("guide", "oracle")):
        old = json.loads((paths.CHECKPOINTS / "c0" / role / "metadata.json").read_text())["args"]
        _report(f"c0 {role}", diff(call["parsed"], old))


def test_fig5_tables14_15():
    """Fig 5 / Tables 14-15: same flags as the stored runs (+ --glass_end_time 1.0 = pre-flag)."""
    for reward in ("gc", "motif", "conjunction"):
        (call,) = dry_run("fig5_tables14_15_gradient_mc.sbatch", REWARD=reward)
        old = json.loads((RES / "scale_free_mc" / reward / "run_metadata.json").read_text())["args"]
        d = diff(call["parsed"], old, ignore={"store_gradient_tensors"})
        _report(f"fig5 {reward}", d)
        assert call["parsed"]["glass_end_time"] == 1.0
        new_only = sorted(set(call["parsed"]) - set(old))
        print(f"    flags the original run did not have: {new_only}")


def test_table13_reward_attainment():
    for reward in ("motif", "conjunction"):
        for length in paths.LENGTHS:
            (call,) = [c for c in dry_run("table13_reward_attainment.sbatch", REWARD=reward, LENGTH=length)
                       if c.get("module")]
            meta = json.loads((RES / "exact_rewards" / reward / f"L{length}" / "metadata.json").read_text())
            _report(f"table13 {reward} L{length}", diff(call["parsed"], meta["generation"]))
            # Table 13 used the diagonal (selected) dMFM students, not the 4-step ones.
            (spec,) = call["parsed"]["dmfm_ckpt"]
            assert Path(spec.split("=", 1)[1]).name == Path(meta["dmfm_checkpoint"]).name, (spec, meta["dmfm_checkpoint"])


def test_table17_posterior_diversity():
    (call,) = dry_run("table17_posterior_diversity.sbatch")
    p = call["parsed"]
    rows = list(csv.DictReader(open(RES / "posterior_diversity" / "run_config.csv")))
    assert p["lengths"] == [int(r["length"]) for r in rows]
    r = rows[0]
    old = {"n_sources": int(r["n_sources"]), "n_futures": int(r["n_futures"]), "dmfm_sampler": r["dmfm_sampler"],
           "dmfm_steps": int(r["dmfm_steps"]), "glass_steps": int(r["glass_steps"]),
           "times": [float(t) for t in r["times"].split()], "seed": int(r["seed"])}
    _report("table17", diff(p, old))
    for row in rows:  # the frozen students are the shipped ones
        assert Path(paths.dmfm_ckpt(int(row["length"]))).name == row["student_ckpt"], row


def test_table18_dmfm4_gradient_mc():
    specs = [(r, L) for r in ("gc", "motif", "conjunction") for L in paths.LENGTHS]
    for task, (reward, length) in enumerate(specs):
        calls = [c for c in dry_run("table18_dmfm4_gradient_mc.sbatch", SLURM_ARRAY_TASK_ID=task) if c.get("module")]
        assert [c["module"] for c in calls] == ["dmfm.experiments.ablate_dmfm_one_step_gradient_mc",
                                                 "dmfm.experiments.rescore_gradient_pairs_mae"], calls
        call, rescore = calls
        # the MAE rescore (Table 18 itself) runs with the original script's defaults on this length
        assert rescore["parsed"]["bootstrap"] == 10000 and rescore["parsed"]["seed"] == 20260802
        assert rescore["parsed"]["lengths"] == [length]
        p = call["parsed"]
        assert p["reward"] == reward and p["lengths"] == [length]
        old = dict(json.loads((RES / "dmfm_vs_glass" / reward / "run_metadata.json").read_text())["args"])
        # run_metadata.json holds the args of the last array task per reward (L=400): its
        # lengths/mc_chunk are per task (MC_CHUNK 16 for L<=200, 8 for L=400 in the original launcher).
        old["lengths"] = [length]
        old["mc_chunk"] = 8 if length == 400 else 16
        _report(f"table18 {reward} L{length}", diff(p, old, ignore={"store_gradient_tensors"}))
        per = json.loads((RES / "dmfm_vs_glass" / reward / f"L{length}" / "metadata.json").read_text())
        (spec,) = p["dmfm_ckpt"]
        assert Path(spec.split("=", 1)[1]).name == Path(per["dmfm_checkpoint"]).name, (spec, per["dmfm_checkpoint"])


def test_train_dmfm_l400_diagonal():
    """train_dmfm.sbatch LENGTH=400 == the L400 diagonal+ESD run (shipped as the finetune's resume point)."""
    ref = paths.CHECKPOINTS / "dmfm_lineage" / "L400_diagonal" / "args.json"
    if not ref.exists():
        return
    (call,) = [c for c in dry_run("train_dmfm.sbatch", LENGTH=400) if c.get("module")]
    old = json.loads(ref.read_text())
    ignore = {"run_name", "yeast_data_pt", "yeast_split_pt", "mfm_teacher_ckpt", "mfm_teacher_ckpt_hparams",
              "commit", "init_ckpt", "wandb", "mfm_legacy_gap_or_default"}
    _report("train_dmfm L400 diagonal", diff(call["parsed"], old, ignore=ignore, only_old_keys=False))


def test_prepare_data():
    (call,) = dry_run("prepare_data.sbatch")
    meta = json.loads(paths.split_json().read_text())
    p = call["parsed"]
    assert p["seed"] == meta.get("seed", 0), (p["seed"], meta.get("seed"))
    assert p["lengths"] == [50, 100, 200, 400]
    print(f"  prepare_data: seed {p['seed']}, lengths {p['lengths']}, stride {p.get('stride')}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"== {name}", flush=True)
            fn()
            print("   passed", flush=True)
    print("all wrapper argument checks passed")
