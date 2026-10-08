#!/usr/bin/env python
"""Check that the ported dMFM training step equals the original library's, bit for bit.

Needs a copy of the original DNA-MFM@187fe7b library (the 2026-09-25 reconstruction
``scratch_dfm_recon_20260925/`` or a checkout of GitHub ``tullebulle/DNA-MFM@187fe7b``) and,
optionally, the July ``utils/mfm_diag_distill.py`` of ``dirichlet-flow-matching`` (the version
the 4-step students were trained with; it differs from 187fe7b only in the gap-bound fix, see
``dmfm.utils.mfm_diag_distill``)::

    source scripts/env.sh
    python tests/dna/check_against_original_library.py --original <recon dir> \\
        --july_distill <dirichlet-flow-matching>/utils/mfm_diag_distill.py --kind dmfm
    python tests/dna/check_against_original_library.py --original <recon dir> \\
        --july_distill <...>/mfm_diag_distill.py --kind dmfm4

Builds the original ``lightning_modules.dna_module.DNAModule`` and the ported
``dmfm.lightning.dna_module.DNAModule`` from the shipped ``args.json`` of the L=50 dMFM
(``--kind dmfm``) or 4-step student (``--kind dmfm4``), loads the same shipped weights, and
compares on CPU: one training step at global step 10 and 5000 (loss and every parameter
gradient), one validation step (all fixed-probe metrics) and the optimizer groups.
Result on 2026-09-29: max |diff| = 0 everywhere, for both kinds.

``--kind c0_sampler`` instead runs the original ``scripts/sample_yeast_c0_guidance_hist.py``
(from ``--original``) and ``dmfm.experiments.sample_c0_guidance`` on CPU with the Table 1
settings for 2 pairs and compares ``sample_scores.csv``. On CPU the guided branch is
deterministic too; result on 2026-09-29: identical guided and unguided sequences and scores.

Run it from a directory other than the repo root (the repo's top-level ``mfm/`` would shadow
the original library's vendored copy). ``timm`` (imported by the image DiT of the vendored MFM
library, never used for DNA) needs torchvision, which the multimfm env lacks; it is stubbed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
from argparse import Namespace
from types import SimpleNamespace

import torch


def _stub_timm() -> None:
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vit = types.ModuleType("timm.models.vision_transformer")
    vit.Mlp = object
    vit.PatchEmbed = object
    data = types.ModuleType("timm.data")
    data.ImageNetInfo = object
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.vision_transformer": vit, "timm.data": data})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="Root of the original DNA-MFM@187fe7b library.")
    p.add_argument("--july_distill", default=None, help="July utils/mfm_diag_distill.py to use in the original stack.")
    p.add_argument("--kind", choices=["dmfm", "dmfm4", "c0_sampler"], default="dmfm")
    args = p.parse_args(argv)
    if args.kind == "c0_sampler":
        return check_c0_sampler(args.original)

    sys.path.insert(0, os.path.abspath(args.original))
    _stub_timm()
    if args.july_distill:
        spec = importlib.util.spec_from_file_location("utils.mfm_diag_distill", args.july_distill)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["utils.mfm_diag_distill"] = mod
        spec.loader.exec_module(mod)
    from lightning_modules.dna_module import DNAModule as OriginalModule

    from dmfm import paths
    from dmfm.lightning.dna_module import DNAModule as PortedModule
    from dmfm.utils.torch_io import torch_load

    length = 50
    args_path = paths.dmfm_args(length) if args.kind == "dmfm" else paths.dmfm4_args(length)
    ckpt_path = paths.dmfm_ckpt(length) if args.kind == "dmfm" else paths.dmfm4_ckpt(length)
    a = json.loads(args_path.read_text())
    a.update(mfm_teacher_ckpt=str(paths.base_ckpt(length)), mfm_teacher_ckpt_hparams=str(paths.base_args(length)),
             mfm_fixed_probe_batch_size=8, batch_size=8)
    margs = Namespace(**a)
    os.environ["MODEL_DIR"] = tempfile.mkdtemp(prefix="dmfm_equiv_")

    seq = torch_load(paths.data_pt(length), map_location="cpu")["seqs"][:8].long()
    cls = torch.zeros(8, dtype=torch.long)
    state = torch_load(ckpt_path, map_location="cpu")["state_dict"]
    state = {k: v for k, v in state.items() if "cls_model" not in k and "distill_model" not in k}

    modules = {}
    for name, cls_ in (("original", OriginalModule), ("port", PortedModule)):
        torch.manual_seed(123)
        m = cls_(margs, 4, 1)
        m.load_state_dict(state, strict=False)
        if hasattr(m.model, "_mfm_teacher_warm_started"):
            m.model._mfm_teacher_warm_started = True
        modules[name] = m

    worst = 0.0
    for step in (10, 5000):
        res = {}
        for name, m in modules.items():
            m._trainer = SimpleNamespace(global_step=step, current_epoch=0, optimizers=[])
            m.stage = "train"
            m.zero_grad(set_to_none=True)
            torch.manual_seed(1000 + step)
            loss = m.general_step((seq, cls), 0)
            loss.backward()
            grad = torch.cat([q.grad.flatten() for q in m.model.parameters() if q.grad is not None])
            res[name] = (float(loss), grad)
        dl = abs(res["original"][0] - res["port"][0])
        dg = float((res["original"][1] - res["port"][1]).abs().max())
        worst = max(worst, dl, dg)
        print(f"train step (global_step={step}): loss {res['port'][0]:.8f}, |dloss| {dl:.1e}, max|dgrad| {dg:.1e}")

    logs = {}
    for name, m in modules.items():
        m._trainer = SimpleNamespace(global_step=5000, current_epoch=0, optimizers=[])
        m.stage = "val"
        torch.manual_seed(7)
        with torch.no_grad():
            m.general_step((seq, cls), 0)
        logs[name] = m._log
    keys = sorted(k for k in logs["original"] if k.startswith("val_mfm"))
    dv = max(float((torch.as_tensor(logs["original"][k], dtype=torch.float64).mean()
                    - torch.as_tensor(logs["port"][k], dtype=torch.float64).mean()).abs()) for k in keys)
    worst = max(worst, dv)
    print(f"validation step: {len(keys)} probe metrics, max|diff| {dv:.1e}")

    groups = {}
    for name, m in modules.items():
        m._trainer = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[], estimated_stepping_batches=1000, max_steps=1000)
        opt = m.configure_optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else (opt["optimizer"] if isinstance(opt, dict) else opt)
        opt = opt[0] if isinstance(opt, list) else opt
        groups[name] = [(len(g["params"]), g.get("lr"), g.get("weight_decay")) for g in opt.param_groups]
    print("optimizer groups:", groups["port"], "(identical)" if groups["port"] == groups["original"] else f"!= {groups['original']}")
    ok = worst == 0.0 and groups["port"] == groups["original"]
    print("ORIGINAL_EQUIVALENCE_OK" if ok else "ORIGINAL_EQUIVALENCE_FAILED")
    return 0 if ok else 1


def check_c0_sampler(original: str) -> int:
    import runpy

    import pandas as pd

    from dmfm import paths
    from dmfm.experiments import sample_c0_guidance

    flags = ["--ckpt", str(paths.base_ckpt(50)), "--c0_ckpt", str(paths.c0_guide()), "--seed", "0",
             "--n_samples", "2", "--batch_size", "2", "--target_c0", "1.0", "--reward_sigma", "0.15",
             "--reward_scale", "0.5", "--mc", "8", "--mc_chunk", "8", "--nfe_traj", "64", "--nfe_value", "4",
             "--t_max", "0.95", "--guide_t_start", "0.50", "--guide_t_end", "0.95", "--guidance_frac", "8.0",
             "--coeff_cap", "10.0", "--grad_clip", "10.0"]
    if torch.cuda.is_available():
        print("run this on a CPU-only node: the guided branch is only deterministic on CPU")
    tmp = tempfile.mkdtemp(prefix="dmfm_c0_equiv_")
    sample_c0_guidance.main(flags + ["--output_dir", f"{tmp}/port"])
    sys.path.insert(0, os.path.abspath(original))
    _stub_timm()
    sys.argv = ["sample_yeast_c0_guidance_hist.py", *flags, "--output_dir", f"{tmp}/original"]
    runpy.run_path(os.path.join(original, "scripts", "sample_yeast_c0_guidance_hist.py"), run_name="__main__")
    a = pd.read_csv(f"{tmp}/original/sample_scores.csv")
    b = pd.read_csv(f"{tmp}/port/sample_scores.csv")
    ok = bool((a.seq_guided == b.seq_guided).all() and (a.seq_unguided == b.seq_unguided).all()
              and (a.guided - b.guided).abs().max() == 0 and (a.unguided - b.unguided).abs().max() == 0)
    print(b[["sample_idx", "unguided", "guided"]].to_string(index=False))
    print("ORIGINAL_EQUIVALENCE_OK" if ok else "ORIGINAL_EQUIVALENCE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
