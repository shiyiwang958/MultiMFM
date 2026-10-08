"""Small-n reruns of the DNA paper experiments with the exact paper settings.

Each function runs the same experiment module as the full-size Slurm wrapper in
``scripts/dna/`` with the same flags (``PAPER_FLAGS``, checked against the wrappers by
``tests/dna/test_dmfm_port.py``), overriding only the sample counts. Meant for the
``RERUN = True`` cells of the DNA notebooks: one GPU, minutes per call. Guided sampling
and GPU gradients are not bit-reproducible (TF32 matmuls, nondeterministic CUDA
backward), so reruns agree with the stored results within their CIs, not exactly; the
unguided Table 1 samples and all CPU post-processing are deterministic.

    from dmfm import rerun
    out = rerun.table1("outputs/dna/rerun/table1_p1", target_c0=1.0, n_samples=32)
    out = rerun.table13("outputs/dna/rerun/table13_motif", length=50, reward="motif", n_samples=16)

Every function returns the output directory (a ``Path``); the files written there are the
same as those of the full-size run (see ``docs/provenance/dna/README.md``). Fig 4 is rerun
through ``dmfm.benchmarks`` / ``python -m dmfm.experiments.benchmark_nfe_steering`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dmfm import paths

# Full-size paper settings, identical to scripts/dna/*.sbatch (and to the original runs).
PAPER_FLAGS: dict[str, dict] = {
    "table1": {  # dmfm.experiments.sample_c0_guidance; Table 1 / Fig 3 / Fig 7
        "ckpt": str(paths.base_ckpt(50)), "c0_ckpt": str(paths.c0_guide()),
        "seed": 0, "n_samples": 1000, "batch_size": 8, "target_c0": 1.0,
        "reward_sigma": 0.15, "reward_scale": 0.5, "mc": 8, "mc_chunk": 8,
        "nfe_traj": 64, "nfe_value": 4, "t_max": 0.95, "guide_t_start": 0.50,
        "guide_t_end": 0.95, "guidance_frac": 8.0, "coeff_cap": 10.0, "grad_clip": 10.0,
    },
    "table1_oracle": {  # dmfm.experiments.score_c0_oracle
        "oracle_ckpt": str(paths.c0_oracle()), "oracle_model_type": "park_cnn",
        "batch_size": 512, "bootstrap": 2000, "seed": 0,
        "reverse_complement_average": True, "device": "cuda:0",
    },
    "table13": {  # dmfm.experiments.sample_reward_attainment
        "lengths": [50], "reward": "motif", "seed": 20260731, "n_samples": 100,
        "batch_size": 8, "nfe_traj": 64, "t_end": 0.999, "guide_t_start": 0.50,
        "guide_t_end": 0.95, "mc": 8, "mc_chunk": 4, "guidance_frac": 8.0,
        "grad_clip": 10.0, "coeff_cap": 10.0, "calibration_samples": 1024,
        "nfe_calibration": 64, "calibration_batch_size": 32, "target_percentile": 0.90,
        "reward_beta": 1.0, "reward_scale": 1.0, "motif": "TTTTTC", "motif2": "AAAATT",
        "motif_tau": 0.10, "conjunction_tau": 0.10, "bootstrap": 2000, "device": "cuda:0",
    },
    "fig5": {  # dmfm.experiments.ablate_glass_gradient_mc; Fig 5, Tables 14-15
        "reward": "gc", "lengths": [50, 100, 200, 400], "t_eval": 0.50, "nfe_probe": 32,
        "nfe_calibration": 64, "calibration_samples": 1024, "sample_batch_size": 32,
        "nfe_value": 8, "glass_end_time": 1.0, "n_probes": 32, "n_repeats": 8,
        "reference_mc": 2048, "mc_values": [1, 2, 4, 8, 16, 32, 64, 128, 256], "mc_chunk": 16,
        "z_target": 1.0, "reward_beta": 1.0, "reward_scale": 1.0,
        "thresholds": [0.05, 0.10, 0.20], "bootstrap": 2000,
    },
    "table17": {  # dmfm.experiments.eval_posterior_diversity
        "lengths": [50, 100, 200, 400], "times": [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        "n_sources": 32, "n_futures": 16, "source_batch": 4, "dmfm_steps": 1,
        "dmfm_sampler": "flow_map", "glass_steps": 100, "bootstrap": 2000, "seed": 20260730,
    },
    "table18": {  # dmfm.experiments.ablate_dmfm_one_step_gradient_mc
        "lengths": [50], "reward": "gc", "reward_objective": "target", "seed": 20260802,
        "t_eval": 0.50, "nfe_calibration": 32, "calibration_samples": 128,
        "sample_batch_size": 16, "n_probes": 32, "n_repeats": 8, "reference_mc": 128,
        "reference_sampler": "glass", "reference_glass_steps": 50,
        "reference_glass_solver": "rk4", "glass_end_time": 0.999, "dmfm_sampler": "flow_map",
        "dmfm_steps": 4, "dmfm_end_time": 0.999, "mc_values": [1, 2, 4, 8],
        "nested_mc_pools": True, "mc_chunk": 16, "z_target": 1.0, "reward_beta": 1.0,
        "reward_scale": 1.0, "thresholds": [0.05, 0.10, 0.20], "bootstrap": 500,
    },
}


def to_argv(flags: dict) -> list[str]:
    """``{"n_samples": 8, "lengths": [50, 100], "flag": True}`` -> argparse argv."""
    argv: list[str] = []
    for key, value in flags.items():
        if value is None or value is False:
            continue
        opt = f"--{key}"
        if value is True:
            argv.append(opt)
        elif isinstance(value, (list, tuple)):
            argv += [opt, *[str(v) for v in value]]
        else:
            argv += [opt, str(value)]
    return argv


def paper_argv(name: str, **overrides) -> list[str]:
    """Paper flags of experiment ``name`` with ``overrides`` applied (``None`` drops a flag)."""
    flags = dict(PAPER_FLAGS[name])
    flags.update(overrides)
    return to_argv(flags)


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _out(out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def table1(out_dir, *, target_c0: float = 1.0, n_samples: int = 32, seed: int = 0,
           score_oracle: bool = True, **overrides) -> Path:
    """Table 1 / Fig 3 (and the Fig 7 histogram): C0-guided vs unguided pairs at L=50 + oracle.

    Sample ``i`` uses seed ``seed + i``, so ``n_samples=32`` reproduces the printed pilot's
    unguided half exactly and the first 32 unguided samples of the n=1000 run.
    ~10 s per 8 pairs on an A100.
    """
    from dmfm.experiments import sample_c0_guidance, score_c0_oracle

    out = _out(out_dir)
    sample_c0_guidance.main(paper_argv("table1", output_dir=str(out), target_c0=target_c0,
                                       n_samples=n_samples, seed=seed, **overrides))
    if score_oracle:
        score_c0_oracle.main(paper_argv("table1_oracle", run_glob=str(out / "sample_scores.csv"),
                                        out_dir=str(out / "oracle"), seed=seed, device=_device()))
    return out


def tables4_5(out_dir, run_dir_m1=None, run_dir_p1=None) -> Path:
    """Tables 4-5 parent-disjoint recompute (CPU, ~3 min at n=1000). Defaults: the shipped n=1000 sets."""
    from dmfm.experiments import analyze_c0_diversity

    out = _out(out_dir)
    run_dirs = [str(run_dir_m1 or paths.RESULTS / "c0_guidance" / "n1000" / "target_m1p0"),
                str(run_dir_p1 or paths.RESULTS / "c0_guidance" / "n1000" / "target_1p0")]
    analyze_c0_diversity.main(["--run_dirs", *run_dirs, "--out_dir", str(out)])
    return out


def table11(device: str = "cpu") -> dict:
    """Table 11: held-out (parent-disjoint ``test``) C0 metrics of the shipped guide and oracle,
    recomputed from the weights and data with reverse-complement averaging (as in training).
    Deterministic; ~10 s on CPU. Returns ``{role: {"n", "mae", "rmse", "pearson", "spearman",
    "stored": metadata.json["test_metrics"]}}``."""
    import json

    import numpy as np
    import torch

    from dmfm.regressors.c0 import load_c0_regressor
    from dmfm.utils.torch_io import torch_load
    from dmfm.utils.yeast_splits import load_yeast_split_indices

    payload = torch_load(paths.data_pt(50), map_location="cpu")
    seqs = payload["seqs"].long()
    c0 = torch.as_tensor(payload["c0"]).float().numpy()
    out = {}
    for role in ("guide", "oracle"):
        meta = json.loads((paths.CHECKPOINTS / "c0" / role / "metadata.json").read_text())
        train_args = meta.get("args", {})
        model = load_c0_regressor(paths.CHECKPOINTS / "c0" / role / "best_state.pt",
                                  train_args.get("model_type", "park_cnn"), device=device)
        idx = load_yeast_split_indices(paths.split_pt(50), train_args.get("test_split", "test")).numpy()
        x = torch.nn.functional.one_hot(seqs[idx], num_classes=4).float().to(device)
        with torch.no_grad():
            pred = (0.5 * (model(x) + model(x.flip(dims=(1,))[..., [3, 2, 1, 0]]))).cpu().numpy()
        truth = c0[idx]
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        out[role] = {
            "n": int(len(idx)),
            "mae": float(np.abs(pred - truth).mean()),
            "rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
            "pearson": float(np.corrcoef(pred, truth)[0, 1]),
            "spearman": float(np.corrcoef(rank(pred), rank(truth))[0, 1]),
            "stored": meta["test_metrics"],
        }
    return out


def table13(out_dir, *, length: int = 50, reward: str = "motif", n_samples: int = 16,
            calibration_samples: int = 1024, **overrides) -> Path:
    """Table 13: exact motif / conjunction reward attainment with the diagonal dMFM students."""
    from dmfm.experiments import sample_reward_attainment

    out = _out(out_dir)
    overrides.setdefault("device", _device())
    sample_reward_attainment.main(paper_argv(
        "table13", lengths=[length], reward=reward, output_dir=str(out), n_samples=n_samples,
        calibration_samples=calibration_samples, dmfm_ckpt=f"{length}={paths.dmfm_ckpt(length)}",
        **overrides))
    return out


def fig5(out_dir, *, reward: str = "gc", lengths: Iterable[int] = (50,), n_probes: int = 4,
         n_repeats: int = 2, mc_values: Iterable[int] = (1, 2, 4, 8, 16, 32),
         reference_mc: int = 2048, rescore: bool = True, **overrides) -> Path:
    """Fig 5 / Tables 14-15: finite-MC GLASS gradient error vs a GLASS reference (base DFM),
    followed by the scale-free rescore (Tables 14-15 columns) and the relative/cosine summary.
    Keep ``n_probes`` even (the probe noise is drawn in blocks). ~25 s per probe at L=50."""
    from dmfm.experiments import (ablate_glass_gradient_mc, rescore_gradient_pairs_scale_free,
                                  summarize_gradient_accuracy_metrics)

    out = _out(out_dir)
    ablate_glass_gradient_mc.main(paper_argv(
        "fig5", output_dir=str(out), reward=reward, lengths=list(lengths), n_probes=n_probes,
        n_repeats=n_repeats, mc_values=list(mc_values), reference_mc=reference_mc, **overrides))
    if rescore:
        rescore_gradient_pairs_scale_free.main(["--input-dir", str(out), "--bootstrap", "10000", "--seed", "20260802"])
        summarize_gradient_accuracy_metrics.main(["--run_dir", str(out)])
    return out


def table17(out_dir, *, lengths: Iterable[int] = (50,), n_sources: int = 4, n_futures: int = 16,
            times: Iterable[float] = (0.0, 0.25, 0.5, 0.75), **overrides) -> Path:
    """Table 17: reward-free posterior diversity, one-step dMFM map vs 100-step GLASS."""
    from dmfm.experiments import eval_posterior_diversity

    out = _out(out_dir)
    eval_posterior_diversity.main(paper_argv(
        "table17", out_dir=str(out), lengths=list(lengths), n_sources=n_sources,
        n_futures=n_futures, times=list(times), **overrides))
    return out


def table18(out_dir, *, reward: str = "gc", length: int = 50, n_probes: int = 4,
            n_repeats: int = 2, **overrides) -> Path:
    """Table 18: value-gradient MAE of the 4-step dMFM vs a GLASS-128 RK4 reference, followed by
    the per-coordinate MAE rescore that produced the table (``L<L>/gradient_mae_summary.csv``,
    column ``mean_mae``). Keep ``n_probes`` even."""
    from dmfm.experiments import ablate_dmfm_one_step_gradient_mc, rescore_gradient_pairs_mae

    out = _out(out_dir)
    ablate_dmfm_one_step_gradient_mc.main(paper_argv(
        "table18", output_dir=str(out), reward=reward, lengths=[length],
        dmfm_ckpt=f"{length}={paths.dmfm4_ckpt(length)}", mc_chunk=8 if length == 400 else 16,
        n_probes=n_probes, n_repeats=n_repeats, **overrides))
    rescore_gradient_pairs_mae.main(["--input-dir", str(out), "--lengths", str(length),
                                     "--bootstrap", "10000", "--seed", "20260802"])
    return out


__all__ = ["PAPER_FLAGS", "to_argv", "paper_argv", "table1", "tables4_5", "table11", "table13", "fig5",
           "table17", "table18"]
