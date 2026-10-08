#!/usr/bin/env python
"""Deterministic forward-pass parity of the port against the ORIGINAL code (CPU).

Runs the same fixed-seed computation twice -- once with the original modules (DNA-MFM@187fe7b
library from the 2026-09-25 reconstruction + the surviving July scripts, in the original
``seq_h100`` env: Python 3.9, torch 2.1) and once with ``dmfm`` (multimfm env: Python 3.11,
torch 2.5) -- on every shipped checkpoint, and compares the outputs:

* base DFM (L=50..400): denoiser logits, drift ``v``, one exact flow step, 4-step GLASS
  posterior (Euler and RK4), GLASS value + value gradient for the gc / motif / conjunction rewards;
* dMFM and 4-step dMFM (L=50..400): velocity, two-time map, one-step map, 4 composed maps,
  value + value gradient through the 4-step flow map (motif reward);
* C0 guide / oracle forward; the Table 1 value gradient (``estimate_v_and_grad``).

Usage (the original tree is assembled from the read-only sources into a work dir)::

    python tests/dna/parity_vs_original.py assemble --work outputs/dna/_parity
    PYTHONNOUSERSITE=1 PYTHONPATH=outputs/dna/_parity/orig:<recon>/pydeps \\
        ~/micromamba/envs/seq_h100/bin/python tests/dna/parity_vs_original.py run --side orig --out outputs/dna/_parity/orig.pt
    python tests/dna/parity_vs_original.py run --side port --out outputs/dna/_parity/port.pt     # multimfm env
    python tests/dna/parity_vs_original.py compare outputs/dna/_parity/orig.pt outputs/dna/_parity/port.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RECON = Path("/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925")
SRC = Path("/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dirichlet-flow-matching")
DMFM = {50: "epoch=95-step=49000.ckpt", 100: "epoch=80-step=39000.ckpt", 200: "epoch=51-step=45000.ckpt",
        400: "epoch=33-step=45750-v1.ckpt"}
DMFM4 = {50: "epoch=29-step=15000.ckpt", 100: "epoch=306-step=149000.ckpt", 200: "epoch=39-step=34000.ckpt",
         400: "epoch=96-step=129000.ckpt"}


def rnd(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def compute(G, D, H, build_c0, flow_step) -> dict:
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    out = {}
    motif, motif2 = G.motif_tensor("TTTTTC", dev), G.motif_tensor("AAAATT", dev)
    kw = dict(score_center=0.5, score_std=0.1, z_target=1.0, reward_beta=1.0, reward_scale=1.0,
              motif=motif, motif2=motif2, motif_tau=0.1, conjunction_tau=0.1)
    for L in (50, 100, 200, 400):
        model, cfg = G.load_model(str(REPO / f"checkpoints/dna/base/L{L}/best.pt"), dev)
        x = rnd((2, L, 4), L)
        t = torch.full((2,), 0.4)
        eps = rnd((6, L, 4), L + 1)
        xr, tr = x.repeat_interleave(3, 0), torch.full((6,), 0.5)
        with torch.no_grad():
            out[f"base{L}_logits"] = model.psi_st(x, t, t, return_logits=True)
            out[f"base{L}_v"] = model.v(t, t, x)
            out[f"base{L}_step"] = flow_step(cfg, model, x, t, torch.full((2,), 0.45))[0]
            out[f"glass{L}_euler"] = G.glass_integrate_diff(model, eps, xr, tr, 4, end_time=0.999, solver="euler")
            out[f"glass{L}_rk4"] = G.glass_integrate_diff(model, eps, xr, tr, 4, end_time=0.999, solver="rk4")
        pool = rnd((2, 3, L, 4), L + 2)
        for reward in ("gc", "motif", "conjunction"):
            v, g = G.estimate_value_gradient(model, x, t_eval=0.5, eps_pool=pool, reward=reward, nfe_value=4,
                                             mc_chunk=2, **kw)
            out[f"glass{L}_{reward}_V"], out[f"glass{L}_{reward}_grad"] = v, g
        for kind, files in (("dmfm", DMFM), ("dmfm_4step", DMFM4)):
            student = D.load_dmfm(str(REPO / f"checkpoints/dna/{kind}/L{L}/{files[L]}"), dev)
            with torch.no_grad():
                s0, s1 = torch.full((6,), 0.1), torch.full((6,), 0.6)
                out[f"{kind}{L}_v"] = student.v(s0, s1, eps, tr, xr)
                out[f"{kind}{L}_map"] = student(s0, s1, eps, tr, xr)
                out[f"{kind}{L}_onestep"] = D.one_step_dmfm(student, eps, xr, tr)
                out[f"{kind}{L}_flow4"] = D.flow_map_dmfm(student, eps, xr, tr, n_steps=4, end_time=0.999)
            v, g = D.estimate_value_gradient(student, x, t_eval=0.5, eps_pool=pool, reward="motif", mc_chunk=2,
                                             dmfm_sampler="flow_map", dmfm_steps=4, dmfm_end_time=0.999, **kw)
            out[f"{kind}{L}_motif_V"], out[f"{kind}{L}_motif_grad"] = v, g
    c0 = {}
    for role in ("guide", "oracle"):
        c0[role] = build_c0("park_cnn")
        c0[role].load_state_dict(torch.load(REPO / f"checkpoints/dna/c0/{role}/best_state.pt", map_location="cpu"))
        c0[role].eval()
    probs = torch.softmax(rnd((5, 50, 4), 7), -1)
    with torch.no_grad():
        out["c0_guide"], out["c0_oracle"] = c0["guide"](probs), c0["oracle"](probs)
    model, _ = G.load_model(str(REPO / "checkpoints/dna/base/L50/best.pt"), dev)
    v, g = H.estimate_v_and_grad(model, c0["guide"], rnd((2, 50, 4), 11), 0.6, rnd((2, 8, 50, 4), 12),
                                 1.0, None, 0.15, 0.5, 4, 8)
    out["table1_V"], out["table1_grad"] = v, g
    return {k: v.detach().clone() for k, v in out.items()}


def cmd_assemble(work: Path) -> None:
    orig = work / "orig"
    if orig.exists():
        shutil.rmtree(orig)
    (orig / "scripts").mkdir(parents=True)
    for d in ("model", "utils", "mfm", "lightning_modules", "dinko"):
        shutil.copytree(RECON / d, orig / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(SRC / "dinko" / "c0_regressors.py", orig / "dinko")
    for f in ("mfm_diag_distill.py", "parsing.py"):  # the July versions
        shutil.copy2(SRC / "utils" / f, orig / "utils")
    for f in ("ablate_glass_gradient_mc.py", "ablate_dmfm_one_step_gradient_mc.py", "probe_dna_mfm_glass.py"):
        shutil.copy2(SRC / "scripts" / f, orig / "scripts")
    shutil.copy2(RECON / "scripts" / "sample_yeast_c0_guidance_hist.py", orig / "scripts")
    print(f"assembled {orig}")


def cmd_run(side: str, out: Path) -> None:
    if side == "orig":
        orig = next(Path(p) for p in sys.path if p and (Path(p) / "model" / "dna_models.py").exists())
        sys.path.insert(0, str(orig / "scripts"))
        import ablate_dmfm_one_step_gradient_mc as D
        import ablate_glass_gradient_mc as G
        import sample_yeast_c0_guidance_hist as H
        from dinko.c0_regressors import build_c0_regressor
        from utils.flow_utils import gaussian_denoiser_flow_step
    else:
        from dmfm.experiments import ablate_dmfm_one_step_gradient_mc as D
        from dmfm.experiments import ablate_glass_gradient_mc as G
        from dmfm.experiments import sample_c0_guidance as H
        from dmfm.regressors.c0 import build_c0_regressor
        from dmfm.utils.flow_utils import gaussian_denoiser_flow_step
    res = compute(G, D, H, build_c0_regressor, gaussian_denoiser_flow_step)
    res["_torch"] = torch.tensor([int(x) for x in torch.__version__.split("+")[0].split(".")[:2]])
    torch.save(res, out)
    print(f"{side}: {len(res) - 1} tensors, torch {torch.__version__} -> {out}")


def cmd_compare(a: Path, b: Path, tol: float = 2e-3) -> int:
    A, B = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    keys = sorted(k for k in A if not k.startswith("_"))
    assert keys == sorted(k for k in B if not k.startswith("_")), set(A) ^ set(B)
    worst = 0.0
    print(f"{'quantity':32s} {'max|diff|':>11s} {'max|ref|':>10s} {'rel':>9s}")
    for k in keys:
        d = (A[k].double() - B[k].double()).abs().max().item()
        s = A[k].double().abs().max().item()
        rel = d / max(s, 1e-30)
        worst = max(worst, rel)
        print(f"{k:32s} {d:11.3e} {s:10.3e} {rel:9.2e}")
    print(f"worst relative max-abs difference over {len(keys)} quantities: {worst:.2e}")
    # Everything agrees to <1e-4 relative except 4-step RK4 GLASS to t=0.999 at L=400 (~1e-3: the
    # near-singular last RK4 stage amplifies torch 2.1 vs 2.5 float differences; argmax identical).
    print("PARITY_OK" if worst < tol else "PARITY_FAILED")
    return 0 if worst < tol else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--work", default=str(REPO / "outputs" / "dna" / "_parity"))
    s = sub.add_parser("run")
    s.add_argument("--side", choices=["orig", "port"], required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--tol", type=float, default=2e-3)
    args = p.parse_args(argv)
    if args.cmd == "assemble":
        cmd_assemble(Path(args.work))
    elif args.cmd == "run":
        cmd_run(args.side, Path(args.out))
    else:
        return cmd_compare(Path(args.a), Path(args.b), args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
