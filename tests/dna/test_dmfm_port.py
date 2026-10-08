"""CPU checks that the ported DNA (dMFM) code works and matches stored results.

Run with the repo root on ``sys.path`` (or after ``pip install -e .``):

    source scripts/env.sh
    python -m pytest tests/dna -x -q          # needs pytest
    python tests/dna/test_dmfm_port.py        # plain-python fallback, no pytest

Tests that need the artifacts under ``checkpoints/dna`` / ``data/dna`` skip when
those are missing (fetch them with
``python scripts/manifest.py verify --group dna --fetch-missing``).

GPU-side checks live next to this file and run from ``scripts/dna/smoke_gpu.sbatch``:
``check_smoke_against_reference.py`` (Table 1 unguided samples vs the stored n=1000 run) and
``check_gradient_mc_reference.py`` (Fig 5 / Table 18 gradients vs stored gradient tensors).
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel, DNAMFMStudent
from dmfm.utils.model_loading import load_args_json, load_student
from dmfm.utils.torch_io import torch_load

LENGTHS = paths.LENGTHS
MODULES = [
    "dmfm.paths",
    "dmfm.api",
    "dmfm._mfm",
    "dmfm.models.dit_seq",
    "dmfm.models.dna_models",
    "dmfm.lightning.general_module",
    "dmfm.lightning.dna_module",
    "dmfm.regressors.c0",
    "dmfm.utils.dataset",
    "dmfm.utils.esm",
    "dmfm.utils.flow_utils",
    "dmfm.utils.log_utils",
    "dmfm.utils.mfm_diag_distill",
    "dmfm.utils.model_loading",
    "dmfm.utils.parsing",
    "dmfm.utils.torch_io",
    "dmfm.utils.yeast_splits",
    "dmfm.experiments.ablate_dmfm_one_step_gradient_mc",
    "dmfm.experiments.ablate_glass_gradient_mc",
    "dmfm.experiments.analyze_c0_diversity",
    "dmfm.experiments.benchmark_nfe_steering",
    "dmfm.experiments.collect_table1_c0",
    "dmfm.experiments.eval_posterior_diversity",
    "dmfm.experiments.plot_gradient_accuracy_required_mc",
    "dmfm.experiments.prepare_yeast_parent_splits",
    "dmfm.experiments.rescore_gradient_pairs_mae",
    "dmfm.experiments.rescore_gradient_pairs_scale_free",
    "dmfm.experiments.sample_c0_guidance",
    "dmfm.experiments.sample_c0_guidance_dmfm",
    "dmfm.experiments.sample_reward_attainment",
    "dmfm.experiments.score_c0_oracle",
    "dmfm.experiments.score_parent_posterior_evo2",
    "dmfm.experiments.score_parent_reward_attainment_evo2",
    "dmfm.experiments.score_yeast_c0_evo2",
    "dmfm.experiments.summarize_gradient_accuracy_metrics",
    "dmfm.experiments.train_base_dfm",
    "dmfm.experiments.train_dmfm",
    "dmfm.experiments._yeast_c0_diversity_metrics",
    "dmfm.rerun",
]
EXPERIMENT_CLIS = [m for m in MODULES if m.startswith("dmfm.experiments.") and not m.split(".")[-1].startswith("_")]


def _skip(msg):
    try:
        import pytest

        pytest.skip(msg, allow_module_level=False)
    except ImportError:
        print(f"SKIP: {msg}")
    return None


def test_import_every_module():
    for name in MODULES:
        importlib.import_module(name)


def test_base_checkpoints_forward():
    """Every base DFM loads strictly and its denoiser produces finite probabilities."""
    for length in LENGTHS:
        ckpt_path = paths.base_ckpt(length)
        if not ckpt_path.exists():
            return _skip(f"missing {ckpt_path}")
        ckpt = torch_load(ckpt_path, map_location="cpu")
        # The standalone DFM trainer stores the exact model config it built from
        # in both the checkpoint and args.json ("model_cfg").
        cfg = SimpleNamespace(**ckpt["model_cfg"])
        stored = json.loads(paths.base_args(length).read_text())["model_cfg"]
        assert stored == ckpt["model_cfg"], length
        assert int(cfg.seq_len) == length, (cfg.seq_len, length)
        model = DiTSequenceModel(cfg, alphabet_size=int(cfg.alphabet_size))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        x = torch.randn(2, length, int(cfg.alphabet_size))
        t = torch.full((2,), 0.5)
        with torch.no_grad():
            psi = model.psi_st(x, t, t)
            v = model.v(t, t, x)
        assert torch.isfinite(psi).all() and torch.isfinite(v).all()
        assert torch.allclose(psi.sum(-1), torch.ones(2, length), atol=1e-4)


def test_dmfm_checkpoints_forward():
    """dMFM and 4-step students load (no missing/unexpected keys) and map states."""
    for kind, ckpt_of, args_of in (
        ("dmfm", paths.dmfm_ckpt, paths.dmfm_args),
        ("dmfm_4step", paths.dmfm4_ckpt, paths.dmfm4_args),
    ):
        for length in LENGTHS:
            if not ckpt_of(length).exists():
                return _skip(f"missing {ckpt_of(length)}")
            student_args = load_args_json(args_of(length))
            student, incompat = load_student(
                student_args, alphabet_size=4, device=torch.device("cpu"), student_ckpt=ckpt_of(length)
            )
            assert not incompat.missing_keys, (kind, length, incompat.missing_keys[:3])
            assert not incompat.unexpected_keys, (kind, length, incompat.unexpected_keys[:3])
            x = torch.randn(2, length, 4)
            s = torch.full((2,), 0.5)
            t = torch.full((2,), 0.75)
            t_cond = torch.zeros(2)
            with torch.no_grad():
                v = student.v(s, t, x, t_cond, x)
                x_next = student.X(s, t, x, v)
            assert torch.isfinite(v).all() and torch.isfinite(x_next).all()


def test_c0_regressors_forward():
    from dmfm.regressors.c0 import load_c0_regressor

    for path in (paths.c0_guide(), paths.c0_oracle()):
        if not path.exists():
            return _skip(f"missing {path}")
        model = load_c0_regressor(path, "park_cnn")
        x = torch.rand(4, 50, 4)
        x = x / x.sum(-1, keepdim=True)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4,) and torch.isfinite(y).all()


def _rc_average(model, tokens):
    """Oracle scoring with --reverse_complement_average (score_c0_oracle)."""
    x = torch.nn.functional.one_hot(torch.as_tensor(tokens, dtype=torch.long), num_classes=4).float()
    with torch.no_grad():
        return (0.5 * (model(x) + model(x.flip(dims=(1,))[..., [3, 2, 1, 0]]))).numpy()


def test_c0_regressor_metrics_match_training_metadata():
    """Recompute the held-out C0 metrics (Table 11) from the checkpoints and data."""
    _c0_metrics()


def _c0_metrics():
    from dmfm.regressors.c0 import load_c0_regressor
    from dmfm.utils.yeast_splits import load_yeast_split_indices

    data_pt, split_pt = paths.data_pt(50), paths.split_pt(50)
    if not (data_pt.exists() and paths.c0_guide().exists()):
        return _skip("missing DNA data or C0 checkpoints")

    payload = torch_load(data_pt, map_location="cpu")
    seqs = payload["seqs"].long()
    c0 = payload["c0"] if "c0" in payload else payload.get("labels")
    assert c0 is not None, f"no C0 labels in {data_pt}: keys={sorted(payload)}"
    c0 = torch.as_tensor(c0).float().numpy()

    out = {}
    for role in ("guide", "oracle"):
        ckpt = paths.CHECKPOINTS / "c0" / role / "best_state.pt"
        meta = json.loads((paths.CHECKPOINTS / "c0" / role / "metadata.json").read_text())
        train_args = meta.get("args", {})
        model = load_c0_regressor(ckpt, train_args.get("model_type", "park_cnn"))
        idx = load_yeast_split_indices(split_pt, train_args.get("test_split", "test")).numpy()
        assert len(idx) == meta["test_metrics"]["n"], (role, len(idx), meta["test_metrics"]["n"])
        pred = _rc_average(model, seqs[idx].numpy())
        truth = c0[idx]
        mae = float(np.abs(pred - truth).mean())
        rmse = float(np.sqrt(((pred - truth) ** 2).mean()))
        pear = float(np.corrcoef(pred, truth)[0, 1])
        out[role] = {"mae": mae, "rmse": rmse, "pearson": pear}
        stored = meta["test_metrics"]
        print(
            f"{role}: recomputed test MAE={mae:.6f} (stored {stored['mae']:.6f}), "
            f"RMSE={rmse:.6f} (stored {stored['rmse']:.6f}), "
            f"Pearson={pear:.6f} (stored {stored['pearson']:.6f})"
        )
        assert math.isclose(mae, stored["mae"], rel_tol=1e-3), (role, "mae", mae, stored["mae"])
        assert math.isclose(rmse, stored["rmse"], rel_tol=1e-3), (role, "rmse", rmse, stored["rmse"])
        assert math.isclose(pear, stored["pearson"], rel_tol=1e-3), (role, "pearson", pear, stored["pearson"])
    return out


def test_mfm_distiller_one_step():
    """One dMFM training step on CPU: finite loss and a gradient on every parameter."""
    if not paths.base_ckpt(50).exists():
        return _skip("missing base checkpoint")
    from dmfm.utils.mfm_diag_distill import GaussianMfmDiagDistiller

    args = load_args_json(paths.dmfm_args(50))
    args.mfm_teacher_ckpt = str(paths.base_ckpt(50))
    args.mfm_teacher_ckpt_hparams = str(paths.base_args(50))
    student = DNAMFMStudent(args, alphabet_size=4)
    distiller = GaussianMfmDiagDistiller(args, alphabet_size=4, device=torch.device("cpu"))
    torch.manual_seed(0)
    seq = torch.randint(4, (2, 50))
    loss, logs = distiller.step_both(seq=seq, student_model=student, global_step=10_000)
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f"dMFM step loss={float(loss):.4f} diag={float(logs['mfm_diag_loss_mean'].mean()):.4f}")


def test_every_experiment_cli_has_help():
    """``python -m dmfm.experiments.<name> --help`` works for every experiment (via main(argv))."""
    import contextlib
    import io

    for name in EXPERIMENT_CLIS:
        mod = importlib.import_module(name)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                mod.main(["--help"])
        except SystemExit as exc:
            assert exc.code in (0, None), (name, exc.code)
        else:
            raise AssertionError(f"{name}.main(['--help']) did not exit")
        assert "usage" in buf.getvalue(), name


_DEAD = ("/n/netscratch", "/n/holylabs", "/n/home", "C0free_torch", "yeast_mid50", "split65k", "workdir/")


def test_no_absolute_or_dead_default_paths():
    """No argparse default / module-level path constant points at cluster paths or purged data."""
    bad = []
    for f in sorted((paths.REPO_ROOT / "src" / "dmfm").rglob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "default":
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and any(d in c.value for d in _DEAD):
                        bad.append(f"{f.name}:{c.lineno}: {c.value}")
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(getattr(node, "value", None), ast.Constant):
                v = node.value.value
                if isinstance(v, str) and any(d in v for d in _DEAD[:3]):
                    bad.append(f"{f.name}:{node.lineno}: {v}")
    for f in sorted((paths.REPO_ROOT / "scripts" / "dna").glob("*")):
        if f.suffix not in (".sh", ".sbatch"):
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.lstrip().startswith("#") and any(d in line for d in _DEAD[:3]):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)


_FAKE_PY = """#!/bin/bash
if [[ "$1" == "-c" ]]; then exec python "$@"; fi
python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@" >> "$DUMP_FILE"
"""


def _dry_run_all(script: str, env: dict) -> list[list[str]]:
    tmp = Path(tempfile.mkdtemp(prefix="dmfm_dry_"))
    fake = tmp / "fake_py.sh"
    fake.write_text(_FAKE_PY)
    fake.chmod(0o755)
    dump = tmp / "argv.jsonl"
    full_env = dict(os.environ, PY=str(fake), DUMP_FILE=str(dump), MULTIMFM_ROOT=str(paths.REPO_ROOT), **env)
    subprocess.run(["bash", str(paths.REPO_ROOT / "scripts" / "dna" / script)], env=full_env,
                   cwd=str(paths.REPO_ROOT), check=True, capture_output=True)
    lines = dump.read_text().splitlines()
    shutil.rmtree(tmp, ignore_errors=True)
    return [json.loads(line)[3:] for line in lines]  # drop "-u -m <module>"


def _dry_run(script: str, env: dict) -> list[str]:
    return _dry_run_all(script, env)[0]


def test_training_wrappers_reproduce_original_args():
    """scripts/dna/train_*.sbatch parse to exactly the args.json stored with each shipped model."""
    if shutil.which("micromamba") is None and not os.environ.get("MAMBA_EXE"):
        return _skip("micromamba not available for scripts/env.sh")
    if not paths.base_args(50).exists():
        return _skip("missing checkpoints")
    from dmfm.experiments import train_base_dfm
    from dmfm.utils.parsing import build_train_parser

    ignore = {"run_name", "output_dir", "yeast_data_pt", "yeast_split_pt", "mfm_teacher_ckpt",
              "mfm_teacher_ckpt_hparams", "commit", "init_ckpt", "ckpt", "wandb", "mfm_legacy_gap_or_default"}

    def diff(new: dict, old: dict) -> dict:
        return {k: (new.get(k), old.get(k)) for k in set(new) | set(old)
                if k not in ignore and new.get(k, "<missing>") != old.get(k, "<missing>")}

    for length in LENGTHS:
        argv = _dry_run("train_base_dfm.sbatch", {"LENGTH": str(length)})
        old = json.loads(paths.base_args(length).read_text())["args"]
        assert not diff(vars(train_base_dfm.parse_args(argv)), old), ("base", length)
        argv = _dry_run("train_dmfm_4step.sbatch", {"LENGTH": str(length)})
        old = json.loads(paths.dmfm4_args(length).read_text())
        assert not diff(vars(build_train_parser().parse_args(argv)), old), ("dmfm4", length)
        if length != 400:  # the shipped L400 dMFM is the finetune (checked below)
            argv = _dry_run("train_dmfm.sbatch", {"LENGTH": str(length)})
            old = json.loads(paths.dmfm_args(length).read_text())
            assert not diff(vars(build_train_parser().parse_args(argv)), old), ("dmfm", length)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "epoch=33-step=45000.ckpt"
        ckpt.touch()
        argv = _dry_run("train_dmfm_l400_finetune.sbatch", {"CKPT": str(ckpt)})
    old = json.loads(paths.dmfm_args(400).read_text())
    assert not diff(vars(build_train_parser().parse_args(argv)), old), ("dmfm L400 finetune",)


def _norm(value):
    """Resolve path-like strings (and ``L=PATH`` specs) so relative and absolute paths compare equal."""
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, str):
        if "=" in value and value.split("=", 1)[0].isdigit():
            length, path = value.split("=", 1)
            return f"{length}={_norm(path)}"
        cand = Path(value) if Path(value).is_absolute() else paths.REPO_ROOT / value
        if cand.exists():
            return str(cand.resolve())
    return value


def test_rerun_paper_flags_match_wrappers():
    """dmfm.rerun.PAPER_FLAGS == the flags of the full-size wrappers in scripts/dna/."""
    if shutil.which("micromamba") is None and not os.environ.get("MAMBA_EXE"):
        return _skip("micromamba not available for scripts/env.sh")
    from dmfm import rerun
    from dmfm.experiments import (ablate_dmfm_one_step_gradient_mc, ablate_glass_gradient_mc,
                                  eval_posterior_diversity, sample_c0_guidance,
                                  sample_reward_attainment, score_c0_oracle)

    out_keys = {"output_dir", "out_dir", "fig_dir", "run_glob"}

    def same(module, wrapper_argv, rerun_argv, tag):
        a = {k: _norm(v) for k, v in vars(module.parse_args(wrapper_argv)).items() if k not in out_keys}
        b = {k: _norm(v) for k, v in vars(module.parse_args(rerun_argv)).items() if k not in out_keys}
        diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
        assert not diff, (tag, diff)

    t1 = _dry_run_all("table1_c0_guidance.sbatch", {"TARGET_C0": "1.0"})
    same(sample_c0_guidance, t1[0], rerun.paper_argv("table1", output_dir="x"), "table1")
    same(score_c0_oracle, t1[1], rerun.paper_argv("table1_oracle", run_glob="x", out_dir="x"), "table1 oracle")
    same(sample_reward_attainment, _dry_run("table13_reward_attainment.sbatch", {"LENGTH": "50", "REWARD": "motif"}),
         rerun.paper_argv("table13", output_dir="x", dmfm_ckpt=f"50={paths.dmfm_ckpt(50)}"), "table13")
    same(ablate_glass_gradient_mc, _dry_run("fig5_tables14_15_gradient_mc.sbatch", {"REWARD": "gc"}),
         rerun.paper_argv("fig5", output_dir="x"), "fig5")
    same(eval_posterior_diversity, _dry_run("table17_posterior_diversity.sbatch", {}),
         rerun.paper_argv("table17", out_dir="x"), "table17")
    same(ablate_dmfm_one_step_gradient_mc, _dry_run("table18_dmfm4_gradient_mc.sbatch", {"REWARD": "gc", "LENGTH": "50"}),
         rerun.paper_argv("table18", output_dir="x", dmfm_ckpt=f"50={paths.dmfm4_ckpt(50)}"), "table18")


def test_evo2_c0_scorer_runs_with_mock_model():
    """The reconstructed Table 6 scorer runs end to end on CPU with a stand-in for Evo2."""
    import dmfm.experiments.score_yeast_c0_evo2 as evo2_c0

    runs = sorted((paths.RESULTS / "c0_guidance" / "n1000").glob("target_*/sample_scores.csv"))
    if len(runs) < 2 or not paths.data_pt(50).exists():
        return _skip("missing results/dna/c0_guidance/n1000 or data")

    class _Tok:
        pad_id, eod_id = 1, 0

        def tokenize(self, seq):
            return [ord(c) for c in seq]

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(512, 512)

        def forward(self, ids):
            return (self.emb(ids), None)

    class _Mock:
        tokenizer, model = _Tok(), _Net()

    original = evo2_c0.load_evo2
    evo2_c0.load_evo2 = lambda name: _Mock()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evo2_c0.main(["--run_glob", str(runs[0]), "--run_glob", str(runs[1]), "--out_dir", tmp,
                          "--targets", "-1", "1", "--max_per_set", "4", "--batch_size", "2",
                          "--bootstrap", "20", "--device", "cpu"])
            import pandas as pd

            summary = pd.read_csv(Path(tmp) / "evo2_summary.csv")
    finally:
        evo2_c0.load_evo2 = original
    # 2 targets x {guided, unguided} + 4 real-data baselines, all with finite NLL
    assert len(summary) >= 6, summary
    assert np.isfinite(summary.select_dtypes("number").to_numpy(dtype=float)).any()
    assert evo2_c0.reverse_complement(evo2_c0.reverse_complement("ACGTTGCA")) == "ACGTTGCA"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"== {fn.__name__}")
        fn()
        print("   passed")
    print("all CPU checks passed")
