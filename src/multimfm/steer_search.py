"""steer_search (SS): PB-valid best-of-lookahead selection for QM9 property guidance.

``steer_search`` is the ``return_best`` variant of MultiMFM steering. A plain
guided sampler returns the FINAL trajectory endpoint. At every step the model
exposes a lookahead endpoint x1_hat = endpoint_prediction(model, x_t, t): its
clean-sample estimate of the molecule it is heading toward. steer_search harvests
those lookahead endpoints along the trajectory (on a t-subgrid), keeps only the
ones that PASS PoseBusters, and returns, per molecule, the PB-VALID lookahead
endpoint whose GUIDE-predicted property is closest to target. Selection is by the
guide regressor; we report held-out ORACLE MAE + PoseBusters validity.

Because only PB-valid endpoints are eligible (and the final endpoint is always a
candidate), selected PB >= final-endpoint PB by construction. We also report the
UNFILTERED best-by-guide endpoint to expose the Goodhart gap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[2]

from multimfm.glass_guidance import (  # noqa: E402
    build_posebusters,
    capture_torch_rng,
    clip_guidance_rms,
    clone_state,
    load_flow_model,
    load_mfm_student,
    pb_summary_for_state,
    predicted_property,
    production_step_with_optional_guidance,
    require_linear_dfm_atoms,
    restore_torch_rng,
    seed_all,
    value_gradient,
)
from multimfm.glass_posterior import endpoint_prediction  # noqa: E402
from multimfm.qm9_data import choose_device, load_tfg_regressor  # noqa: E402
from multimfm.property_targets import (  # noqa: E402
    build_or_load_histograms,
    sample_property_target,
)
from tabasco.chem.convert import MoleculeConverter  # noqa: E402
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com  # noqa: E402
from multimfm.train_base_flow import prepare_pb_molecules  # noqa: E402


def _pb_bust_chunk(payload):
    """Worker: PoseBusters-validate a chunk of RDKit mols (None allowed). Returns list[bool].
    Each worker builds its own PoseBusters from the config path (picklable inputs only)."""
    config_path, mols = payload
    import yaml
    from rdkit import Chem
    from posebusters import PoseBusters
    from multimfm.posebusters_compat import install_posebusters_compat
    install_posebusters_compat()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    pb = PoseBusters(config=cfg)
    keep, pb_mols = [], []
    for j, m in enumerate(mols):
        if m is None:
            continue
        try:
            mm = Chem.Mol(m)
        except Exception:
            continue
        if mm is None or mm.GetNumAtoms() == 0:
            continue
        pb_mols.append(mm)
        keep.append(j)
    out = [False] * len(mols)
    if pb_mols:
        try:
            res = pb.bust(mol_pred=pb_mols)
            passes = ~res.isin([False]).any(axis=1)
            for j, ok in zip(keep, passes.tolist()):
                out[j] = bool(ok)
        except RuntimeError:
            pass
    return out


def parallel_pb_valid(all_mols, config_path, workers):
    """all_mols: list length C of lists length B (RDKit mol or None). Returns valid[C,B] bool tensor.
    Flattens every candidate x molecule and spreads PoseBusters across `workers` CPU processes."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    C = len(all_mols)
    B = len(all_mols[0]) if C else 0
    flat_idx = [(c, i) for c in range(C) for i in range(B)]
    flat_mols = [all_mols[c][i] for (c, i) in flat_idx]
    W = max(1, int(workers))
    chunks_mols = [flat_mols[k::W] for k in range(W)]
    chunks_idx = [flat_idx[k::W] for k in range(W)]
    valid = torch.zeros(C, B, dtype=torch.bool)
    payloads = [(str(config_path), ch) for ch in chunks_mols]
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
            results = list(ex.map(_pb_bust_chunk, payloads))
    except Exception as exc:  # robust fallback: sequential
        print(f"[pb] parallel pool failed ({exc}); falling back to sequential", flush=True)
        results = [_pb_bust_chunk(p) for p in payloads]
    for w in range(W):
        for (c, i), ok in zip(chunks_idx[w], results[w]):
            valid[c, i] = ok
    return valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "base_flow" / "model_step_100000.pt",
                   help="Base QM9 flow-matching checkpoint.")
    p.add_argument("--mfm-checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "mfm_student" / "student_step_1000.pt",
                   help="Meta flow map (MFM) student checkpoint used as the value sampler.")
    p.add_argument("--mfm-key", default="student")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--target-seed", type=int, default=2026)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=128)
    p.add_argument("--mu", type=float, default=0.45)
    p.add_argument("--reward-scale", type=float, default=0.3)
    p.add_argument("--value-samples", type=int, default=32)
    p.add_argument("--value-batch-size", type=int, default=0)
    p.add_argument("--value-glass-steps", type=int, default=4)
    p.add_argument("--mfm-diagonal", action="store_true")
    p.add_argument("--guide-every", type=int, default=1)
    p.add_argument("--guide-min-t", type=float, default=0.05)
    p.add_argument("--guide-max-t", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--soft-atom-temperature", type=float, default=0.25)
    p.add_argument("--deterministic", action="store_true",
                   help=("disable TF32 and enable deterministic cuDNN/cuBLAS kernels; makes runs "
                         "reproducible across GPUs at some cost in speed"))
    p.add_argument("--guidance-max-coord-rms", type=float, default=0.0,
                   help="per-molecule RMS cap on the coordinate guidance gradient (0 = off)")
    p.add_argument("--guidance-max-atom-rms", type=float, default=0.0,
                   help="per-molecule RMS cap on the atom guidance gradient (0 = off)")
    p.add_argument("--property-name", default="alpha",
                   choices=["alpha", "cv", "gap", "homo", "lumo", "mu"],
                   help="QM9 property to steer toward.")
    p.add_argument("--partition", default="train_flow")
    p.add_argument("--hist-bins", type=int, default=100)
    # steer_search knobs
    p.add_argument("--select-min-t", type=float, default=0.5,
                   help="Only harvest lookahead endpoints with t >= this value.")
    p.add_argument("--select-max-t", type=float, default=1.01)
    p.add_argument("--pb-candidates", type=int, default=0,
                   help="Harvest subgrid size; <=0 means harvest & PB-check EVERY in-window step.")
    p.add_argument("--pb-workers", type=int, default=16,
                   help="CPU processes to spread the PoseBusters validity check across.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "steer_search")
    p.add_argument("--hist-cache", type=Path, default=None)
    p.add_argument("--force-rebuild-hist", action="store_true")
    p.add_argument("--no-sanitize", action="store_true")
    p.add_argument("--posebusters-config", type=Path, default=None)
    return p.parse_args()


def make_state(coords, atomics, mask):
    return TensorDict(
        {"coords": mask_and_zero_com(coords, mask),
         "atomics": apply_mask(atomics, mask),
         "padding_mask": mask},
        batch_size=coords.shape[0],
    ).to(coords.device)


@torch.no_grad()
def pb_valid_mask(state, *, data_stats, posebusters, sanitize) -> torch.Tensor:
    """Per-molecule PoseBusters validity (True = passes all checks). Shape [B]."""
    B = state["padding_mask"].shape[0]
    mask = torch.zeros(B, dtype=torch.bool)
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    batch_mols = converter.from_batch(state.detach().cpu(), sanitize=sanitize, rescale_coords=True)
    mols, mol_indices = [], []
    for i, m in enumerate(batch_mols):
        if m is not None:
            mols.append(m); mol_indices.append(i)
    if not mols:
        return mask
    pb_mols, pb_indices, _, _ = prepare_pb_molecules(mols, mol_indices)
    if not pb_mols:
        return mask
    try:
        results = posebusters.bust(mol_pred=pb_mols)
    except RuntimeError:
        return mask
    passes = ~results.isin([False]).any(axis=1)
    for idx, ok in zip(pb_indices, passes.tolist()):
        mask[idx] = bool(ok)
    return mask


@torch.no_grad()
def evaluate(state, *, targets, property_name, guide_reg, oracle_reg, data_stats,
             posebusters, sanitize, run):
    guide = predicted_property(state, property_name=property_name, property_regressor=guide_reg)
    oracle = predicted_property(state, property_name=property_name, property_regressor=oracle_reg)
    tgt = targets.to(guide.device)
    pb = pb_summary_for_state(state, data_stats=data_stats, posebusters=posebusters, sanitize=sanitize)
    # The paper's Table 2 reports MAE over PoseBusters-valid molecules only.
    valid = pb_valid_mask(state, data_stats=data_stats, posebusters=posebusters,
                          sanitize=sanitize).to(guide.device)
    oracle_err = (oracle - tgt).abs()
    return {
        "run": run,
        "guide_mae": float((guide - tgt).abs().mean().cpu()),
        "oracle_mae": float(oracle_err.mean().cpu()),
        "oracle_median_abs_error": float(oracle_err.median().cpu()),
        "oracle_mae_pbvalid": float(oracle_err[valid].mean().cpu()) if valid.any() else float("nan"),
        "n_pbvalid": int(valid.sum().cpu()),
        "pb_pb_valid": pb.get("pb_valid"),
        # uniqueness over molecules with a SMILES (U) and over all generated (U_all)
        "unique_smiles_rate": pb.get("unique_smiles_rate"),
        "unique_smiles_rate_total": pb.get("unique_smiles_rate_total"),
    }


def guided_sample_collect(model, mfm_student, init_state, *, args, targets, guide_reg):
    """Guided trajectory; snapshot lookahead endpoints on a t-subgrid in the window.

    Returns (final_state, cand_coords[C,B,N,3], cand_atomics[C,B,N,A], cand_reward[C,B],
    cand_t[C]) where the last candidate (index C-1) is the final trajectory endpoint.
    """
    state = clone_state(init_state)
    device = state["coords"].device
    schedule = model._get_sample_schedule(args.sample_steps).to(device)
    tgt = targets.to(device)
    B = state["coords"].shape[0]
    mask = init_state["padding_mask"]

    # choose harvest steps: t in window, evenly spaced, <= pb_candidates of them
    in_window = [i for i in range(1, len(schedule))
                 if args.select_min_t <= float(schedule[i - 1]) <= args.select_max_t]
    if args.pb_candidates and args.pb_candidates > 0 and len(in_window) > args.pb_candidates:
        sel = np.linspace(0, len(in_window) - 1, args.pb_candidates).round().astype(int)
        harvest = sorted({in_window[j] for j in sel})
    else:
        harvest = in_window  # harvest EVERY in-window step

    def score_reward(cand_state):
        pred = predicted_property(cand_state, property_name=args.property_name,
                                  property_regressor=guide_reg)
        return -(pred - tgt).square()

    cand_coords, cand_atomics, cand_reward, cand_t = [], [], [], []
    grad_rms_log = []
    for i in range(1, len(schedule)):
        t_value = schedule[i - 1]
        dt_value = schedule[i] - schedule[i - 1]
        t = t_value.expand(B)
        should_guide = (
            args.mu != 0.0 and args.value_samples > 0 and args.guide_every > 0
            and (i - 1) % args.guide_every == 0
            and float(t_value) >= args.guide_min_t and float(t_value) <= args.guide_max_t
        )
        grad_coords = grad_atomics = None
        if should_guide:
            rng = capture_torch_rng(device)
            with torch.enable_grad():
                grad_coords, grad_atomics, vg_stats = value_gradient(
                    model, state, t, value_sampler="mfm", mfm_student=mfm_student,
                    mfm_diagonal=args.mfm_diagonal, posterior_samples=args.value_samples,
                    n_steps=args.value_glass_steps, reward_name="target_property",
                    reward_scale=args.reward_scale, target_x=0.0,
                    property_name=args.property_name, property_target=tgt,
                    property_regressor=guide_reg, guide_atomics=True, eps=args.eps,
                    value_batch_size=args.value_batch_size,
                )
            grad_coords, _ = clip_guidance_rms(grad_coords, state["padding_mask"],
                                               args.guidance_max_coord_rms)
            grad_atomics, _ = clip_guidance_rms(grad_atomics, state["padding_mask"],
                                                args.guidance_max_atom_rms)
            grad_rms_log.append((vg_stats["grad_coords_rms"], vg_stats.get("grad_atomics_rms", 0.0)))
            restore_torch_rng(rng, device)

        if i in harvest:
            with torch.no_grad():
                cand = endpoint_prediction(model, state, t)
                cand_coords.append(cand["coords"].clone())
                cand_atomics.append(cand["atomics"].clone())
                cand_reward.append(score_reward(cand))
                cand_t.append(float(t_value))

        with torch.no_grad():
            dt = dt_value.expand(B)
            state = production_step_with_optional_guidance(
                model, state, t, dt, mu=args.mu, grad_coords=grad_coords, grad_atomics=grad_atomics)

    # append final trajectory endpoint as the last candidate
    cand_coords.append(state["coords"].clone())
    cand_atomics.append(state["atomics"].clone())
    cand_reward.append(score_reward(state))
    cand_t.append(1.0)
    grad_rms = (
        {"grad_coords_rms_mean": float(np.mean([g[0] for g in grad_rms_log])),
         "grad_coords_rms_max": float(np.max([g[0] for g in grad_rms_log])),
         "grad_atomics_rms_mean": float(np.mean([g[1] for g in grad_rms_log]))}
        if grad_rms_log else {}
    )
    return (state,
            torch.stack(cand_coords), torch.stack(cand_atomics),
            torch.stack(cand_reward), torch.tensor(cand_t), grad_rms)


def gather_by_idx(cand, idx):  # cand [C,B,...], idx [B] -> [B,...]
    B = idx.shape[0]
    return cand[idx, torch.arange(B, device=idx.device)]


def main() -> None:
    args = parse_args()
    if args.hist_cache is None:
        args.hist_cache = (REPO_ROOT / "cache" / "histograms" /
                           f"hist_{args.property_name}_{args.partition}_{args.hist_bins}.json")
    if args.posebusters_config is None:
        import tabasco
        args.posebusters_config = (Path(tabasco.__file__).parent /
                                   "utils" / "posebusters_no_strain.yaml")
    seed_all(args.seed)
    if args.deterministic:
        # TF32 (10-bit mantissa) is implemented differently on A100 vs H100; over a
        # 128-step ODE those differences amplify until molecules cross PoseBusters
        # thresholds, giving up to ~12% run-to-run spread in reported MAE.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.deterministic only covers convolutions; this model is a transformer,
        # so the run-to-run drift comes from atomicAdd reductions in the autograd
        # backward (scatter/index_add, SDPA backward). This is what catches those.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, data_stats, model_args = load_flow_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mfm_student = load_mfm_student(args.mfm_checkpoint, data_stats=data_stats,
                                   teacher_args=model_args, device=device, key=args.mfm_key)
    guide_reg = load_tfg_regressor("guide", args.property_name, device)
    oracle_reg = load_tfg_regressor("oracle", args.property_name, device)
    for reg in (guide_reg, oracle_reg):
        reg.soft_atom_temperature = args.soft_atom_temperature
        reg.eval()
        for prm in reg.parameters():
            prm.requires_grad_(False)

    max_len = int(data_stats["max_num_atoms"])
    hist = build_or_load_histograms(args, max_len=max_len)

    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_samples).to(device)
    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
    trng = np.random.default_rng(args.target_seed)
    target_values = [sample_property_target(hist, int(l), trng) for l in lengths]
    targets = torch.tensor(target_values, dtype=torch.float32, device=device)

    seed_all(args.seed)
    final_state, c_coords, c_atomics, c_reward, c_t, grad_rms = guided_sample_collect(
        model, mfm_student, init_state, args=args, targets=targets, guide_reg=guide_reg)

    mask = init_state["padding_mask"]
    posebusters = build_posebusters(args.posebusters_config)
    sanitize = not args.no_sanitize
    C, B = c_reward.shape

    # decode EVERY candidate endpoint to RDKit mols (CPU), then PB-check ALL of them in parallel
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    all_mols = []
    for c in range(C):
        cs = make_state(c_coords[c], c_atomics[c], mask).detach().cpu()
        all_mols.append(list(converter.from_batch(cs, sanitize=sanitize, rescale_coords=True)))
    valid = parallel_pb_valid(all_mols, args.posebusters_config, args.pb_workers).to(device)

    # unfiltered best-by-guide (Goodhart reference)
    idx_raw = c_reward.argmax(dim=0)
    raw_state = make_state(gather_by_idx(c_coords, idx_raw), gather_by_idx(c_atomics, idx_raw), mask)

    # PB-valid: highest-guide-reward endpoint (over ALL lookaheads) that passes PB; else final
    masked = torch.where(valid, c_reward, torch.full_like(c_reward, -float("inf")))
    idx_valid = masked.argmax(dim=0)
    has_valid = valid.any(dim=0)
    sel_idx = torch.where(has_valid, idx_valid, torch.full_like(idx_valid, C - 1))
    valid_state = make_state(gather_by_idx(c_coords, sel_idx), gather_by_idx(c_atomics, sel_idx), mask)

    ev = lambda st, run: evaluate(st, targets=targets, property_name=args.property_name,
                                  guide_reg=guide_reg, oracle_reg=oracle_reg, data_stats=data_stats,
                                  posebusters=posebusters, sanitize=sanitize, run=run)
    final_eval = ev(final_state, "final_endpoint")
    raw_eval = ev(raw_state, "best_lookahead_raw")
    valid_eval = ev(valid_state, "best_lookahead_pbvalid")
    valid_eval["frac_with_valid_candidate"] = float(has_valid.float().mean().cpu())
    valid_eval["selected_t_mean"] = float(c_t.to(device)[sel_idx].float().mean().cpu())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": [final_eval, raw_eval, valid_eval],
        "grad_rms": grad_rms,
        "num_candidates": int(C),
        "candidate_t": [float(x) for x in c_t.tolist()],
        "lengths": [int(x) for x in lengths],
        "targets": [float(x) for x in target_values],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    for e in (final_eval, raw_eval, valid_eval):
        print(e)


if __name__ == "__main__":
    main()
