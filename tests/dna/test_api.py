"""CPU checks that ``dmfm.api`` reproduces the experiment code paths.

    source scripts/env.sh
    python tests/dna/test_api.py          # or: python -m pytest tests/dna -q
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmfm import api, paths

REPO = paths.REPO_ROOT
REF_N1000 = REPO / "results" / "dna" / "c0_guidance" / "n1000" / "target_1p0"


def _have(*ps: Path) -> bool:
    return all(Path(p).exists() for p in ps)


def test_value_and_grad_matches_c0_sampler():
    """api.value_and_grad(GLASS, C0 reward) == sample_c0_guidance.estimate_v_and_grad."""
    if not _have(paths.base_ckpt(50), paths.c0_guide()):
        return
    from dmfm.experiments.sample_c0_guidance import estimate_v_and_grad

    base, cfg = api.load_base(50, "cpu")
    guide = api.load_c0("guide", "cpu")
    ids = [0, 1]
    x0 = api.paired_initial_noise(ids, 50, seed=0, device="cpu")
    x = api.sample_unguided(base, cfg, x0, nfe=32, t_max=0.5)
    pool = api.paired_eps_pool(ids, 50, mc=4, seed=0, device="cpu")
    v_ref, g_ref = estimate_v_and_grad(base, guide, x, 0.5, pool, 1.0, None, 0.15, 0.5, 4, 2)
    post = api.glass_posterior_fn(base, n_steps=4, end_time=1.0)
    logr = api.c0_log_reward_fn(guide, 1.0)
    v, g = api.value_and_grad(post, logr, x, 0.5, pool, mc_chunk=2)
    assert torch.allclose(v, v_ref, atol=1e-5, rtol=1e-5), (v, v_ref)
    assert torch.allclose(g, g_ref, atol=1e-5, rtol=1e-4), (g - g_ref).abs().max()


def test_value_and_grad_matches_dmfm_ablation():
    """api.value_and_grad(dMFM flow map, gc reward) == ablate_dmfm_one_step_gradient_mc.estimate_value_gradient."""
    if not _have(paths.dmfm4_ckpt(50)):
        return
    from dmfm.experiments import ablate_dmfm_one_step_gradient_mc as abl
    from dmfm.experiments.ablate_glass_gradient_mc import motif_tensor, target_reward

    student = api.load_dmfm(50, "cpu", kind="dmfm4")
    torch.manual_seed(0)
    x = torch.randn(2, 50, 4)
    pool = torch.randn(2, 3, 50, 4)
    motif = motif_tensor("TATAAA", torch.device("cpu"))
    kw = dict(score_center=0.4, score_std=0.05, z_target=1.0, reward_beta=1.0, reward_scale=1.0, reward="gc",
              motif=motif, motif2=motif, motif_tau=0.1, conjunction_tau=0.1)
    v_ref, g_ref = abl.estimate_value_gradient(student, x, t_eval=0.5, eps_pool=pool, mc_chunk=2,
                                               dmfm_sampler="flow_map", dmfm_steps=4, dmfm_end_time=0.999, **kw)
    post = api.dmfm_posterior_fn(student, sampler="flow_map", n_steps=4, end_time=0.999)

    def logr(endpoint):
        return 1.0 * target_reward(endpoint, score_center=0.4, score_std=0.05, z_target=1.0, beta=1.0, reward="gc",
                                   motif=motif, motif2=motif, motif_tau=0.1, conjunction_tau=0.1)

    v, g = api.value_and_grad(post, logr, x, 0.5, pool, mc_chunk=2)
    assert torch.allclose(v, v_ref, atol=1e-5), (v, v_ref)
    assert torch.allclose(g, g_ref, atol=1e-5, rtol=1e-4), (g - g_ref).abs().max()


def test_unguided_matches_stored_table1_run():
    """The deterministic unguided branch reproduces the stored n=1000 Table 1 run (CPU vs GPU:
    identical tokens expected for nearly all samples; guide scores to ~1e-4)."""
    if not _have(paths.base_ckpt(50), paths.c0_guide(), REF_N1000 / "sample_scores.csv"):
        return
    ref = pd.read_csv(REF_N1000 / "sample_scores.csv").set_index("sample_idx")
    base, cfg = api.load_base(50, "cpu")
    guide = api.load_c0("guide", "cpu")
    ids = list(range(8))
    x0 = api.paired_initial_noise(ids, 50, seed=0, device="cpu")
    x = api.sample_unguided(base, cfg, x0, nfe=64, t_max=0.95)
    seqs = api.tokens_to_strings(x)
    same = sum(a == b for a, b in zip(seqs, ref.loc[ids, "seq_unguided"]))
    with torch.no_grad():
        score = guide(api.one_hot(x.argmax(-1))).numpy()
    diff = np.abs(score - ref.loc[ids, "unguided"].to_numpy())
    print(f"unguided tokens identical {same}/{len(ids)}; guide score max|diff| {diff.max():.2e}")
    assert same >= len(ids) - 1, same
    assert np.median(diff) < 1e-3, diff


def test_oracle_score_matches_stored():
    """api.oracle_score (RC-averaged park_cnn oracle) reproduces the stored per-sample oracle scores."""
    f = REF_N1000 / "oracle" / "oracle_per_sample.csv"
    if not _have(paths.c0_oracle(), f, REF_N1000 / "sample_scores.csv"):
        return
    oracle = api.load_c0("oracle", "cpu")
    per = pd.read_csv(f).set_index("sample_idx").sort_index()
    df = pd.read_csv(REF_N1000 / "sample_scores.csv").set_index("sample_idx").sort_index()
    ids = df.index[:200]
    for label in ("unguided", "guided"):
        got = api.oracle_score(oracle, df.loc[ids, f"seq_{label}"].tolist())
        d = np.abs(got - per.loc[ids, f"oracle_{label}"].to_numpy())
        print(f"oracle {label}: max|diff| {d.max():.2e} over {len(ids)}")
        assert d.max() < 1e-3, d.max()


def test_c0_guided_pairs_runs():
    """One tiny guided batch through the Table 1 code path (CPU, 2 samples, reduced NFE)."""
    if not _have(paths.base_ckpt(50), paths.c0_guide()):
        return
    base, cfg = api.load_base(50, "cpu")
    guide = api.load_c0("guide", "cpu")
    rows = api.c0_guided_pairs(base, cfg, guide, [0, 1], target_c0=1.0, nfe_traj=8, mc=2, mc_chunk=2, nfe_value=2)
    assert len(rows) == 2 and {"guided", "unguided", "seq_guided"} <= set(rows[0])


def test_paired_glass_equals_table1_sampler():
    """sample_c0_guidance_dmfm --paired --posterior glass == sample_c0_guidance (Table 1 code path),
    and --posterior dmfm runs through the same loop (CPU, reduced NFE)."""
    if not _have(paths.base_ckpt(50), paths.c0_guide(), paths.dmfm_ckpt(50)):
        return
    from dmfm.experiments import sample_c0_guidance as t1
    from dmfm.experiments import sample_c0_guidance_dmfm as t1d

    small = ["--n_samples", "2", "--batch_size", "2", "--mc", "2", "--mc_chunk", "2", "--nfe_value", "2",
             "--t_max", "0.95", "--guide_t_start", "0.5", "--guide_t_end", "0.95", "--guidance_frac", "8.0",
             "--coeff_cap", "10.0", "--grad_clip", "10.0", "--reward_sigma", "0.15", "--reward_scale", "0.5",
             "--target_c0", "1.0", "--seed", "0"]
    a = t1.parse_args(["--ckpt", str(paths.base_ckpt(50)), "--nfe_traj", "16", *small])
    base, cfg = api.load_base(50, "cpu")
    guide = api.load_c0("guide", "cpu")
    a.gaussian_beta_schedule, a.gaussian_beta_table_path, a.flow_temp = "linear", None, 1.0
    ref = t1.sample_paired_batch([0, 1], a, base, guide, torch.device("cpu"), 50, 4)
    b = t1d.parse_args(["--paired", "--posterior", "glass", "--nfe_sample", "16", *small])
    got = t1d.paired_batch(b, [0, 1], base, cfg, None, guide, torch.device("cpu"))
    for r, g in zip(ref, got):
        assert r["seq_unguided"] == g["seq_unguided"] and r["seq_guided"] == g["seq_guided"], (r, g)
        assert abs(r["guided"] - g["guided"]) < 1e-6 and abs(r["unguided"] - g["unguided"]) < 1e-6, (r, g)
    b.posterior = "dmfm"
    rows = t1d.paired_batch(b, [0, 1], base, cfg, api.load_dmfm(50, "cpu"), guide, torch.device("cpu"))
    assert [r["seq_unguided"] for r in rows] == [r["seq_unguided"] for r in ref]  # same pairing noise
    print("  guided C0 (GLASS vs dMFM posterior, 2 samples, reduced NFE):",
          [round(r["guided"], 3) for r in got], [round(r["guided"], 3) for r in rows])


def test_data_loaders():
    if not _have(paths.data_pt(50), paths.split_pt(50)):
        return
    seqs, c0 = api.load_c0_labels("test")
    assert seqs.shape == (8294, 50) and c0.shape == (8294,)
    parents = api.load_windows(50)["parent_id"][api.split_indices(50, "test")].unique()
    assert len(parents) == 58
    for L in paths.LENGTHS:
        x = api.load_split_seqs(L, "a_train", max_n=4)
        assert x.shape == (4, L) and x.dtype == torch.long


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"== {name}", flush=True)
            fn()
            print("   passed", flush=True)
    print("all dmfm.api checks passed")
