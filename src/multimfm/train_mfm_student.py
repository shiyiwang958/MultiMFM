"""Stage-1 training for the TABASCO MFM student: diagonal GLASS distillation.

Trains ``TabascoFlowMap`` to match the teacher's GLASS posterior velocity on the
diagonal (s=u), over the full t_cond range, on real QM9 molecules. Periodically
evaluates the few-step consistency posterior sampler against the many-step GLASS
posterior (PoseBusters validity), and checkpoints the student.

Run:
  python -m multimfm.train_mfm_student --device cuda --steps 5000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multimfm.glass_guidance import (  # noqa: E402
    build_posebusters,
    clip_guidance_rms,
    is_tfg_property,
    load_flow_model,
    pb_summary_for_state,
    production_step_with_optional_guidance,
    value_gradient,
)
from multimfm.glass_endpoints import (  # noqa: E402
    make_condition_state_from_endpoint,
    sample_glass_posterior,
)
from multimfm.qm9_data import load_qm9_tensors, load_tfg_regressor  # noqa: E402
from multimfm.property_targets import (  # noqa: E402
    build_or_load_histograms,
    sample_property_target,
)
from tabasco.data.utils import TensorDictCollator  # noqa: E402
from tabasco.flow.mfm_losses import (  # noqa: E402
    diagonal_distill_loss,
    esd_consistency_loss,
    glass_posterior_diagonal_distill_loss,
    psd_consistency_loss,
    glass_rollout_distill_loss,
    glass_transition_distill_loss,
)
from tabasco.models.components.flow_map_transformer import TabascoFlowMap  # noqa: E402
from tabasco.sample.flow_map import _prior_like, consistency_posterior_sample  # noqa: E402
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com  # noqa: E402

CKPT = "checkpoints/base_flow/model_step_100000.pt"


def build_student(teacher, margs, hidden_dim=None, num_layers=None, num_heads=None,
                  block_conditioning="additive",
                  velocity_parametrization="endpoint",
                  encoder_depth=None, time_encoding_max_len=200) -> TabascoFlowMap:
    return TabascoFlowMap(
        spatial_dim=teacher.data_stats["spatial_dim"],
        atom_dim=teacher.data_stats["atom_dim"],
        num_heads=num_heads or margs.num_heads,
        num_layers=num_layers or margs.num_layers,
        hidden_dim=hidden_dim or margs.hidden_dim,
        cross_attention=not getattr(margs, "no_cross_attention", False),
        add_sinusoid_posenc=True,
        max_num_atoms=teacher.data_stats["max_num_atoms"],
        block_conditioning=block_conditioning,
        velocity_parametrization=velocity_parametrization,
        encoder_depth=encoder_depth,
        time_encoding_max_len=time_encoding_max_len,
    )


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def infinite_batches(tensors, batch_size):
    loader = DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TensorDictCollator(),
        drop_last=True,
    )
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def sample_production_condition_state(
    model,
    *,
    batch_size: int,
    t_cond: torch.Tensor,
    num_steps: int,
) -> TensorDict:
    """Sample an unguided production-trajectory state at a scalar condition time."""
    if t_cond.ndim != 1 or t_cond.numel() == 0:
        raise ValueError("t_cond must be a non-empty 1D tensor")
    t_target = float(t_cond[0].detach().cpu())
    if not torch.allclose(t_cond, torch.full_like(t_cond, t_target), atol=1e-6):
        raise ValueError("generated condition source requires a scalar t_cond per batch")

    state = model._sample_noise_like_batch(batch_size=batch_size).to(t_cond.device)
    schedule = model._get_sample_schedule(num_steps).to(t_cond.device)
    for i in range(1, len(schedule)):
        t0 = float(schedule[i - 1])
        t1 = float(schedule[i])
        if t0 >= t_target:
            break
        dt_val = min(t1, t_target) - t0
        if dt_val <= 0.0:
            continue
        t = torch.full((batch_size,), t0, device=t_cond.device)
        dt = torch.full((batch_size,), dt_val, device=t_cond.device)
        pred = model._call_net(state, t)
        coords = model.coords_interpolant.step(state, pred, t, dt)
        atomics = model.atomics_interpolant.step(state, pred, t, dt)
        state = TensorDict(
            {
                "coords": mask_and_zero_com(coords, state["padding_mask"]),
                "atomics": apply_mask(atomics, state["padding_mask"]),
                "padding_mask": state["padding_mask"],
            },
            batch_size=state["padding_mask"].shape[0],
        ).to(t_cond.device)
    return state


def sample_guided_production_condition_state(
    model,
    init_state: TensorDict,
    *,
    t_cond: torch.Tensor,
    num_steps: int,
    mu: float,
    reward_scale: float,
    value_samples: int,
    value_glass_steps: int,
    property_name: str,
    property_target: torch.Tensor,
    property_regressor,
    guide_atomics: bool,
    guide_min_t: float,
    guide_max_t: float,
    guide_every: int,
    guidance_max_coord_rms: float,
    guidance_max_atom_rms: float,
    eps: float,
) -> TensorDict:
    """Sample a strongly guided production-trajectory state at scalar ``t_cond``."""
    if t_cond.ndim != 1 or t_cond.numel() == 0:
        raise ValueError("t_cond must be a non-empty 1D tensor")
    t_target = float(t_cond[0].detach().cpu())
    if not torch.allclose(t_cond, torch.full_like(t_cond, t_target), atol=1e-6):
        raise ValueError("guided condition source requires a scalar t_cond per batch")

    state = TensorDict(
        {
            "coords": init_state["coords"].clone(),
            "atomics": init_state["atomics"].clone(),
            "padding_mask": init_state["padding_mask"].clone(),
        },
        batch_size=init_state["padding_mask"].shape[0],
    ).to(t_cond.device)
    schedule = model._get_sample_schedule(num_steps).to(t_cond.device)
    for i in range(1, len(schedule)):
        t0 = float(schedule[i - 1])
        t1 = float(schedule[i])
        if t0 >= t_target:
            break
        dt_val = min(t1, t_target) - t0
        if dt_val <= 0.0:
            continue
        t = torch.full((state["coords"].shape[0],), t0, device=t_cond.device)
        dt = torch.full((state["coords"].shape[0],), dt_val, device=t_cond.device)
        should_guide = (
            mu != 0.0
            and value_samples > 0
            and guide_every > 0
            and (i - 1) % guide_every == 0
            and t0 >= guide_min_t
            and t0 <= guide_max_t
        )
        grad_coords = None
        grad_atomics = None
        if should_guide:
            with torch.enable_grad():
                grad_coords, grad_atomics, _ = value_gradient(
                    model,
                    state,
                    t,
                    value_sampler="glass",
                    posterior_samples=value_samples,
                    n_steps=value_glass_steps,
                    reward_name="target_property",
                    reward_scale=reward_scale,
                    target_x=0.0,
                    property_name=property_name,
                    property_target=property_target,
                    property_regressor=property_regressor,
                    guide_atomics=guide_atomics,
                    eps=eps,
                )
            grad_coords, _ = clip_guidance_rms(
                grad_coords,
                state["padding_mask"],
                guidance_max_coord_rms,
            )
            if grad_atomics is not None:
                grad_atomics, _ = clip_guidance_rms(
                    grad_atomics,
                    state["padding_mask"],
                    guidance_max_atom_rms,
                )
        with torch.no_grad():
            state = production_step_with_optional_guidance(
                model,
                state,
                t,
                dt,
                mu=mu,
                grad_coords=grad_coords,
                grad_atomics=grad_atomics if guide_atomics else None,
            )
    return state


@torch.no_grad()
def parse_eval_configs(spec: str, legacy_diagonal: bool, legacy_steps: int):
    """Parse ``"diag:4,jump:2"`` into ``[("diag", 4), ("jump", 2)]``."""
    if not spec.strip():
        return [("diag" if legacy_diagonal else "jump", legacy_steps)]
    configs = []
    for item in spec.split(","):
        mode, _, n = item.strip().partition(":")
        if mode not in ("diag", "jump") or not n.isdigit():
            raise ValueError(f"bad --eval-configs entry {item!r}; expected diag:N or jump:N")
        configs.append((mode, int(n)))
    return configs


@torch.no_grad()
def build_eval_conditions(teacher, eval_x1, device, t_conds, seed):
    """Seeded noised conditions and prior draws, shared by every evaluation.

    Fixing them makes eval curves comparable across steps and across runs that
    use the same ``--eval-seed`` (the samplers are deterministic given the prior).
    """
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        conds = []
        for t_cond_val in t_conds:
            endpoint = eval_x1.clone().to(device)
            cond, t_cond = make_condition_state_from_endpoint(
                teacher, endpoint, t_cond=t_cond_val
            )
            conds.append((t_cond_val, cond, t_cond, _prior_like(cond)))
    return conds


@torch.no_grad()
def source_fidelity(sample, source):
    """Agreement of posterior samples with the clean molecules that produced x_t.

    Returns (atom-type accuracy over real atoms, mean per-molecule coordinate RMSD
    after zeroing the COM). The noised condition keeps the source frame and atom
    order, so no alignment is needed; at t=0 these are chance-level by design.
    """
    mask = source["padding_mask"]
    real = (~mask).float()
    n = real.sum(dim=1).clamp_min(1.0)
    acc = ((sample["atomics"].argmax(-1) == source["atomics"].argmax(-1)).float() * real).sum(1) / n
    dc = mask_and_zero_com(sample["coords"], mask) - mask_and_zero_com(source["coords"], mask)
    rmsd = ((dc.square().sum(-1) * real).sum(1) / n).sqrt()
    return float(acc.mean()), float(rmsd.mean())


@torch.no_grad()
def glass_reference(teacher, eval_conds, data_stats, posebusters, glass_steps, seed, device,
                    source):
    """PB validity of the many-step GLASS teacher posterior on the eval conditions."""
    devices = [device] if device.type == "cuda" else []
    rows = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for t_cond_val, cond, t_cond, _ in eval_conds:
            glass = sample_glass_posterior(
                teacher, cond, t_cond, samples_per_condition=1, num_steps=glass_steps
            )
            pb = pb_summary_for_state(
                glass, data_stats=data_stats, posebusters=posebusters, sanitize=True
            )
            acc, rmsd = source_fidelity(glass, source)
            rows.append({"t_cond": t_cond_val, "pb_valid": pb.get("pb_valid", 0.0),
                         "conv": pb.get("conversion_rate", 0.0), "acc": acc, "rmsd": rmsd})
    return rows


@torch.no_grad()
def evaluate(student, eval_conds, data_stats, posebusters, eval_configs, source):
    """PB validity of student posterior samples for each (t_cond, sampler config).

    ``diag:N`` Euler-integrates the diagonal velocity ``v(s, s)`` (the meta
    denoiser turned into the inner GLASS velocity) with N steps; ``jump:N`` takes
    N flow-map jumps ``v(s, u)``.
    """
    rows = []
    for t_cond_val, cond, t_cond, prior in eval_conds:
        for mode, n_steps in eval_configs:
            sample = consistency_posterior_sample(
                student, cond, t_cond, n_steps=n_steps, noise=prior.clone(),
                diagonal=(mode == "diag"),
            )
            pb = pb_summary_for_state(
                sample, data_stats=data_stats, posebusters=posebusters, sanitize=True
            )
            acc, rmsd = source_fidelity(sample, source)
            rows.append({"t_cond": t_cond_val, "mode": mode, "steps": n_steps,
                         "pb_valid": pb.get("pb_valid", 0.0),
                         "conv": pb.get("conversion_rate", 0.0), "acc": acc, "rmsd": rmsd})
    return rows


def eval_log_dict(rows, prefix):
    out = {}
    by_cfg = defaultdict(list)
    for r in rows:
        cfg = f"{r['mode']}{r['steps']}" if "mode" in r else prefix
        key = f"eval/{cfg}/t{r['t_cond']:.1f}"
        out[f"{key}/pb_valid"] = r["pb_valid"]
        out[f"{key}/conv"] = r["conv"]
        out[f"{key}/source_atom_acc"] = r["acc"]
        out[f"{key}/source_coord_rmsd"] = r["rmsd"]
        by_cfg[cfg].append(r["pb_valid"])
    for cfg, vals in by_cfg.items():
        out[f"eval/{cfg}/mean_pb_valid"] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--partition", default="train_flow")
    ap.add_argument("--limit-data", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--t-max", type=float, default=0.99, help="clamp s and t_cond below 1")
    ap.add_argument("--endpoint-space-loss", action="store_true",
                    help="weight loss by (1-s)^2 (endpoint-space; use with endpoint param)")
    ap.add_argument("--diag-loss-type", choices=["mse", "adaptive"], default="mse",
                    help="diagonal target loss; adaptive matches the MFM v2 recipe")
    ap.add_argument("--diag-adaptive-p", type=float, default=0.5)
    ap.add_argument("--diag-adaptive-c", type=float, default=0.01)
    ap.add_argument("--diag-direction-weight", type=float, default=0.0,
                    help="extra masked 1-cosine loss on diagonal GLASS velocities")
    ap.add_argument("--diag-condition-jvp-weight", type=float, default=0.0,
                    help="finite-difference loss matching d v_diag / d x_cond")
    ap.add_argument("--diag-condition-jvp-direction-weight", type=float, default=0.0,
                    help="masked 1-cosine loss for finite-difference d v_diag / d x_cond")
    ap.add_argument("--diag-condition-jvp-eps", type=float, default=0.03,
                    help="per-sample RMS condition perturbation for diagonal sensitivity loss")
    ap.add_argument("--diag-condition-jvp-mode",
                    choices=["both", "coords", "atomics"],
                    default="both",
                    help="which conditioning modality to perturb for the diagonal sensitivity loss")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help=("1=diagonal distill; 2=ESD/JVP; "
                          "3=direct GLASS transition distill; "
                          "4=GLASS rollout distill"))
    ap.add_argument("--init-ckpt", default=None,
                    help="student checkpoint to initialize from (e.g. a Stage-1 run)")
    ap.add_argument("--skip-init-coord-head", action="store_true",
                    help="when loading --init-ckpt, leave the coordinate output head at initialization")
    ap.add_argument("--init-from-teacher", action="store_true",
                    help="initialize the student from the teacher's endpoint predictor")
    ap.add_argument("--esd-weight", type=float, default=1.0)
    ap.add_argument("--diag-weight", type=float, default=1.0)
    ap.add_argument("--coord-loss-weight", type=float, default=1.0,
                    help="weight for coordinate loss terms in diagonal/ESD/transition objectives")
    ap.add_argument("--atom-loss-weight", type=float, default=1.0,
                    help="weight for atom-state loss terms in diagonal/ESD/transition objectives")
    ap.add_argument("--esd-target-clip", type=float, default=0.0,
                    help="clamp the GLASS velocity target to +/- this (0=off)")
    ap.add_argument("--esd-delta-clip", type=float, default=0.0,
                    help="per-sample L2 cap for the ESD delta (u-s)*jvp before adding vss")
    ap.add_argument("--esd-loss-type", choices=["mse", "adaptive"], default="mse")
    ap.add_argument("--esd-adaptive-p", type=float, default=0.5)
    ap.add_argument("--esd-adaptive-c", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500,
                    help="LR warmup steps (then cosine decay)")
    ap.add_argument("--anneal-frac", type=float, default=0.5,
                    help="fraction of steps over which the s->u jump grows to full")
    ap.add_argument("--max-jump", type=float, default=0.5,
                    help="cap on the inner-time jump (u - s) for the ESD term")
    ap.add_argument("--t-cond-power", type=float, default=1.0,
                    help="sample t_cond as rand()**power * t_max")
    ap.add_argument("--t-cond-zero-rate", type=float, default=0.0,
                    help="probability of forcing t_cond=0, matching MFM-style sampling")
    ap.add_argument("--t-cond-grid", default="",
                    help="optional comma-separated t_cond values sampled uniformly, e.g. 0.3,0.6,0.9")
    ap.add_argument("--t-cond-jitter", type=float, default=0.0,
                    help="uniform jitter radius for --t-cond-grid samples")
    ap.add_argument("--s-grid", default="",
                    help="optional comma-separated inner-time s values sampled uniformly")
    ap.add_argument("--s-grid-jitter", type=float, default=0.0,
                    help="uniform jitter radius for --s-grid samples")
    ap.add_argument("--s-power", type=float, default=1.0,
                    help="sample unconstrained s as rand()**power * t_max when --s-grid is unset")
    ap.add_argument("--transition-steps", default="1,2,4",
                    help="comma-separated coarse sampler grids for Stage 3, e.g. 1,2,4")
    ap.add_argument("--transition-teacher-steps", type=int, default=64,
                    help="fine GLASS steps used to build Stage-3 transition targets")
    ap.add_argument("--rollout-steps", type=int, default=4,
                    help="student coarse steps for Stage-4 rollout distillation")
    ap.add_argument("--rollout-final-weight", type=float, default=2.0,
                    help="extra weight for the final coarse node in Stage-4 rollout loss")
    ap.add_argument("--rollout-velocity-weight", type=float, default=0.0,
                    help="Stage-4 weight for per-step GLASS coarse-transition velocity targets")
    ap.add_argument("--rollout-teacher-forced-velocity-weight", type=float, default=0.0,
                    help="Stage-4 weight for velocity targets queried on GLASS coarse nodes")
    ap.add_argument("--rollout-spread-weight", type=float, default=0.0,
                    help="Stage-4 weight for coordinate posterior-spread matching")
    ap.add_argument("--rollout-spread-samples", type=int, default=1,
                    help="posterior samples per condition for Stage-4 spread matching")
    ap.add_argument("--rollout-diagonal", action="store_true",
                    help="Stage-4 trains diagonal Euler rollout v(s,s,.) instead of off-diagonal jumps")
    ap.add_argument("--rollout-condition-source",
                    choices=["endpoint", "generated", "guided"],
                    default="endpoint",
                    help=("Stage-4 conditioning source: endpoint uses noised real QM9 "
                          "endpoints; generated uses unguided production states at t_cond; "
                          "guided uses GLASS-guided production states at t_cond"))
    ap.add_argument("--generated-condition-steps", type=int, default=64,
                    help="production steps used when --rollout-condition-source is generated/guided")
    ap.add_argument("--guided-condition-property-name", default="alpha",
                    help="TFG property used to guide generated conditioning states")
    ap.add_argument("--guided-condition-mu", type=float, default=0.45)
    ap.add_argument("--guided-condition-reward-scale", type=float, default=0.3)
    ap.add_argument("--guided-condition-value-samples", type=int, default=16)
    ap.add_argument("--guided-condition-value-glass-steps", type=int, default=4)
    ap.add_argument("--guided-condition-guide-every", type=int, default=1)
    ap.add_argument("--guided-condition-guide-min-t", type=float, default=0.05)
    ap.add_argument("--guided-condition-guide-max-t", type=float, default=0.85)
    ap.add_argument("--guided-condition-guide-atomics", action="store_true")
    ap.add_argument("--guided-condition-max-coord-rms", type=float, default=0.0)
    ap.add_argument("--guided-condition-max-atom-rms", type=float, default=0.0)
    ap.add_argument("--guided-condition-soft-atom-temperature", type=float, default=0.25)
    ap.add_argument("--guided-condition-eps", type=float, default=1e-6)
    ap.add_argument("--target-seed", type=int, default=2026)
    ap.add_argument("--hist-bins", type=int, default=100)
    ap.add_argument("--hist-cache", type=Path,
                    default=Path("cache/qm9_property_histograms/alpha_train_flow_bins100.json"))
    ap.add_argument("--force-rebuild-hist", action="store_true")
    ap.add_argument("--eval-diagonal", action="store_true",
                    help="evaluate checkpoints with diagonal Euler posterior sampling")
    ap.add_argument("--eval-configs", default="",
                    help=("comma-separated sampler configs, e.g. diag:4,diag:32,jump:1,jump:4; "
                          "empty = legacy single config from --eval-diagonal/--mfm-steps"))
    ap.add_argument("--eval-t-conds", default="0.3,0.6,0.9",
                    help="comma-separated outer times t of the noised eval conditions")
    ap.add_argument("--eval-seed", type=int, default=1234,
                    help="seed for the fixed eval conditions and priors")
    ap.add_argument("--eval-offset", type=int, default=0,
                    help=("index of the first eval molecule; with --steps 0 an offset past "
                          "the training range gives an unseen confirmation set"))
    ap.add_argument("--eval-at-start", action="store_true",
                    help="also evaluate the initialized student before step 1")
    ap.add_argument("--atom-loss-type", choices=["mse", "ce"], default="mse",
                    help=("atom term of the diagonal and ESD losses: mse matches atom "
                          "velocities; ce matches the softmax meta denoiser with soft-label "
                          "cross entropy (diagonal: teacher psi_t*; ESD: logit-space target)"))
    ap.add_argument("--time-encoding-max-len", type=int, default=200,
                    help="highest Fourier frequency of the student's s and (u-s) time features")
    ap.add_argument("--jump-t-max", type=float, default=None,
                    help=("upper bound of the stage-2 jump end time u (default t_max); "
                          "s stays <= t_max. Set 1.0 to train jumps that land at u=1"))
    ap.add_argument("--consistency-loss", choices=["esd", "psd"], default="esd",
                    help=("stage-2 off-diagonal loss: esd = Eulerian self-distillation (JVP); "
                          "psd = progressive self-distillation X_{s,w} = X_{u,w}(X_{s,u})"))
    ap.add_argument("--esd-ce-shift-clip", type=float, default=5.0,
                    help="clip on the ESD logit shift -log(1 - kinv*delta) for --atom-loss-type ce")
    ap.add_argument("--wandb-project", default="",
                    help="log to this W&B project (disabled if empty)")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--mfm-steps", type=int, default=4)
    ap.add_argument("--glass-steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/mfm_student")
    ap.add_argument("--hidden-dim", type=int, default=None, help="override student width")
    ap.add_argument("--num-layers", type=int, default=None, help="override student depth")
    ap.add_argument("--num-heads", type=int, default=None, help="override student heads")
    ap.add_argument("--block-conditioning", choices=["additive", "adaln", "gated", "gated_delta", "split_gated"], default="additive",
                    help="additive=time added to tokens; gated=additive plus zero-init per-block gates; adaln=per-block adaLN-Zero modulation")
    ap.add_argument("--encoder-depth", type=int, default=None,
                    help="encoder depth for split_gated conditioning; defaults to about 2/3 of layers")
    ap.add_argument("--velocity-parametrization",
                    choices=[
                        "endpoint",
                        "direct",
                        "endpoint_residual",
                        "coord_direct_atom_endpoint",
                        "coord_endpoint_residual_atom_endpoint",
                    ],
                    default="endpoint",
                    help="student output parametrization for diagonal velocity")
    ap.add_argument("--train-residual-only", action="store_true",
                    help="freeze non-residual parameters; requires a residual parametrization")
    ap.add_argument("--train-coord-output-only", action="store_true",
                    help="freeze all parameters except out_coord_linear")
    ap.add_argument("--train-conditioning-adapter-only", action="store_true",
                    help="freeze all parameters except zero-init gated/delta conditioning adapters")
    args = ap.parse_args()
    if args.atom_loss_type == "ce" and args.stage in (3, 4):
        raise ValueError("--atom-loss-type ce is implemented for stages 1 and 2 only")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir) / time.strftime(f"stage{args.stage}_%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out_dir}")
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, group=args.wandb_group,
            config={**vars(args), "out_dir": str(out_dir)}, dir=str(out_dir),
        )

    teacher, data_stats, margs = load_flow_model(Path(CKPT), device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = build_student(teacher, margs, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers, num_heads=args.num_heads,
                            block_conditioning=args.block_conditioning,
                            velocity_parametrization=args.velocity_parametrization,
                            encoder_depth=args.encoder_depth,
                            time_encoding_max_len=args.time_encoding_max_len).to(device)
    same_size = (args.hidden_dim in (None, margs.hidden_dim)
                 and args.num_layers in (None, margs.num_layers))
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        init_state = init["student"]
        if args.skip_init_coord_head:
            init_state = {
                k: v
                for k, v in init_state.items()
                if not k.startswith("out_coord_linear.")
            }
            print("skipping coordinate output head from init checkpoint")
        incompatible = student.load_state_dict(init_state, strict=False)
        print(f"initialized student from {args.init_ckpt} (step {init.get('step')})")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "  non-strict load: "
                f"missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
    elif args.init_from_teacher:
        if same_size:
            student.init_from_base(
                teacher.net,
                copy_output_heads=(args.velocity_parametrization != "direct"),
            )
        else:
            print("WARNING: student size != teacher; skipping init_from_base (training from scratch)")
    if args.train_residual_only:
        if args.velocity_parametrization not in (
            "endpoint_residual",
            "coord_endpoint_residual_atom_endpoint",
        ):
            raise ValueError(
                "--train-residual-only requires a residual velocity parametrization"
            )
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith("residual_"))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training residual heads only: trainable_params={trainable:,}")
    if args.train_coord_output_only:
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith("out_coord_linear."))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training coordinate output head only: trainable_params={trainable:,}")
    if args.train_conditioning_adapter_only:
        adapter_prefixes = ("block_cond_gates.", "cond_delta_coord_embed.", "cond_delta_atom_embed.")
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith(adapter_prefixes))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training conditioning adapters only: trainable_params={trainable:,}")
    ema = {k: v.detach().clone() for k, v in student.state_dict().items()}
    print(f"teacher net={sum(p.numel() for p in teacher.net.parameters()):,} "
          f"student={sum(p.numel() for p in student.parameters()):,}")

    data_args = SimpleNamespace(
        partition=args.partition, limit_data=args.limit_data,
        max_len=args.max_len, include_hydrogens=False,
    )
    tensors, _ = load_qm9_tensors(data_args)
    print(f"loaded {len(tensors)} QM9 molecules")
    collate = TensorDictCollator()
    eval_x1 = collate(tensors[args.eval_offset : args.eval_offset + args.n_eval])
    if args.eval_offset and args.steps > 0:
        raise ValueError("--eval-offset is for eval-only runs (--steps 0)")
    if eval_x1["padding_mask"].shape[0] < args.n_eval:
        raise ValueError(f"only {eval_x1['padding_mask'].shape[0]} eval molecules at offset "
                         f"{args.eval_offset}; raise --limit-data")
    batches = infinite_batches(tensors[args.n_eval :], args.batch_size)

    guided_regressor = None
    guided_hist = None
    guided_target_rng = None
    if args.rollout_condition_source == "guided":
        if not is_tfg_property(args.guided_condition_property_name):
            raise ValueError(
                "--guided-condition-property-name must be a TFG property, got "
                f"{args.guided_condition_property_name!r}"
            )
        guided_regressor = load_tfg_regressor(
            "guide",
            args.guided_condition_property_name,
            device,
        )
        guided_regressor.soft_atom_temperature = args.guided_condition_soft_atom_temperature
        guided_regressor.eval()
        for param in guided_regressor.parameters():
            param.requires_grad_(False)
        hist_args = argparse.Namespace(**vars(args))
        hist_args.property_name = args.guided_condition_property_name
        guided_hist = build_or_load_histograms(
            hist_args,
            max_len=int(data_stats["max_num_atoms"]),
        )
        guided_target_rng = np.random.default_rng(args.target_seed)

    posebusters = build_posebusters(Path("src/tabasco/utils/posebusters_no_strain.yaml"))
    eval_configs = parse_eval_configs(args.eval_configs, args.eval_diagonal, args.mfm_steps)
    eval_t_conds = [float(x) for x in args.eval_t_conds.split(",") if x.strip()]
    eval_conds = build_eval_conditions(teacher, eval_x1, device, eval_t_conds, args.eval_seed)
    eval_source = eval_x1.clone().to(device)
    glass_rows = glass_reference(teacher, eval_conds, data_stats, posebusters,
                                 args.glass_steps, args.eval_seed, device, eval_source)
    glass_log = eval_log_dict(glass_rows, f"glass{args.glass_steps}")
    print(f"  [GLASS {args.glass_steps}-step reference] " + "  ".join(
        f"t {r['t_cond']:.2f}: pb {r['pb_valid']:.3f} acc {r['acc']:.3f} rmsd {r['rmsd']:.3f}"
        for r in glass_rows))

    def run_eval(step):
        backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
        student.load_state_dict(ema)
        student.eval()
        rows = evaluate(student, eval_conds, data_stats, posebusters, eval_configs, eval_source)
        student.load_state_dict(backup)
        student.train()
        print(f"  [eval @ {step}] EMA student vs GLASS {args.glass_steps}-step")
        glass_by_t = {r["t_cond"]: r for r in glass_rows}
        for r in rows:
            g = glass_by_t[r["t_cond"]]
            print(f"    t_cond {r['t_cond']:.2f}  {r['mode']}:{r['steps']:<3d} "
                  f"pb_valid {r['pb_valid']:.3f} conv {r['conv']:.3f} "
                  f"acc {r['acc']:.3f} rmsd {r['rmsd']:.3f}  | "
                  f"GLASS pb_valid {g['pb_valid']:.3f} acc {g['acc']:.3f} rmsd {g['rmsd']:.3f}",
                  flush=True)
        if wandb_run is not None:
            wandb_run.log({**eval_log_dict(rows, ""), **glass_log}, step=step)

    if args.eval_at_start:
        run_eval(0)
    opt = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    t_cond_grid = None
    if args.t_cond_grid.strip():
        values = [float(x.strip()) for x in args.t_cond_grid.split(",") if x.strip()]
        if not values:
            raise ValueError("--t-cond-grid was set but no numeric values were parsed")
        t_cond_grid = torch.tensor(values, dtype=torch.float32, device=device)
        if torch.any(t_cond_grid < 0) or torch.any(t_cond_grid > args.t_max):
            raise ValueError(
                f"--t-cond-grid values must lie in [0, t_max={args.t_max}]"
            )
    s_grid = None
    if args.s_grid.strip():
        values = [float(x.strip()) for x in args.s_grid.split(",") if x.strip()]
        if not values:
            raise ValueError("--s-grid was set but no numeric values were parsed")
        s_grid = torch.tensor(values, dtype=torch.float32, device=device)
        if torch.any(s_grid < 0) or torch.any(s_grid > args.t_max):
            raise ValueError(f"--s-grid values must lie in [0, t_max={args.t_max}]")

    def sample_inner_s(shape):
        if s_grid is not None:
            idx = torch.randint(s_grid.numel(), shape, device=device)
            s_val = s_grid[idx]
            if args.s_grid_jitter > 0:
                jitter = (2.0 * torch.rand_like(s_val) - 1.0) * args.s_grid_jitter
                s_val = (s_val + jitter).clamp(0.0, args.t_max)
            return s_val
        return torch.rand(shape, device=device).pow(args.s_power) * args.t_max
    transition_steps = tuple(
        int(x.strip()) for x in args.transition_steps.split(",") if x.strip()
    )
    if args.stage in (3, 4):
        if not transition_steps:
            raise ValueError("--transition-steps must contain at least one integer")
        for n in transition_steps:
            if args.transition_teacher_steps % n != 0:
                raise ValueError(
                    "--transition-teacher-steps must be divisible by every "
                    f"transition grid; got teacher_steps={args.transition_teacher_steps}, "
                    f"transition_steps={transition_steps}"
                )
    if args.stage == 4 and args.transition_teacher_steps % args.rollout_steps != 0:
        raise ValueError(
            "--transition-teacher-steps must be divisible by --rollout-steps; "
            f"got teacher_steps={args.transition_teacher_steps}, "
            f"rollout_steps={args.rollout_steps}"
        )

    bins = np.linspace(0, 1, 6)
    window = defaultdict(list)
    running = {"loss": [], "coord": [], "atom": []}
    bin_l2 = {i: [] for i in range(len(bins) - 1)}

    for step in range(1, args.steps + 1):
        cur_lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        batch = next(batches).to(device)
        B = batch["padding_mask"].shape[0]
        scalar_t_cond_batch = (
            args.rollout_condition_source in ("generated", "guided")
            and args.stage in (1, 4)
        )
        if t_cond_grid is not None:
            shape = (1,) if scalar_t_cond_batch else (B,)
            idx = torch.randint(t_cond_grid.numel(), shape, device=device)
            t_cond = t_cond_grid[idx]
            if args.t_cond_jitter > 0:
                jitter = (2.0 * torch.rand_like(t_cond) - 1.0) * args.t_cond_jitter
                t_cond = (t_cond + jitter).clamp(0.0, args.t_max)
        else:
            shape = (1,) if scalar_t_cond_batch else (B,)
            t_cond_base = torch.rand(shape, device=device).pow(args.t_cond_power)
            t_cond = t_cond_base * args.t_max
        if args.t_cond_zero_rate > 0:
            keep_noisy = torch.rand_like(t_cond) >= args.t_cond_zero_rate
            t_cond = t_cond * keep_noisy.to(t_cond.dtype)
        if scalar_t_cond_batch:
            t_cond = t_cond.expand(B)

        opt.zero_grad()
        if args.stage == 1:
            if args.rollout_condition_source in ("generated", "guided"):
                s = sample_inner_s((1,)).expand(B)
                if args.rollout_condition_source == "generated":
                    cond_state = sample_production_condition_state(
                        teacher,
                        batch_size=B,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                    )
                else:
                    if guided_hist is None or guided_target_rng is None:
                        raise RuntimeError("guided condition source was not initialized")
                    init_state = teacher._sample_noise_like_batch(batch_size=B).to(device)
                    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
                    target_values = [
                        sample_property_target(guided_hist, int(length), guided_target_rng)
                        for length in lengths
                    ]
                    property_targets = torch.tensor(
                        target_values,
                        dtype=torch.float32,
                        device=device,
                    )
                    cond_state = sample_guided_production_condition_state(
                        teacher,
                        init_state,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                        mu=args.guided_condition_mu,
                        reward_scale=args.guided_condition_reward_scale,
                        value_samples=args.guided_condition_value_samples,
                        value_glass_steps=args.guided_condition_value_glass_steps,
                        property_name=args.guided_condition_property_name,
                        property_target=property_targets,
                        property_regressor=guided_regressor,
                        guide_atomics=args.guided_condition_guide_atomics,
                        guide_min_t=args.guided_condition_guide_min_t,
                        guide_max_t=args.guided_condition_guide_max_t,
                        guide_every=args.guided_condition_guide_every,
                        guidance_max_coord_rms=args.guided_condition_max_coord_rms,
                        guidance_max_atom_rms=args.guided_condition_max_atom_rms,
                        eps=args.guided_condition_eps,
                    )
                loss, comp = glass_posterior_diagonal_distill_loss(
                    student,
                    teacher,
                    cond_state,
                    t_cond,
                    s,
                    posterior_steps=args.transition_teacher_steps,
                    endpoint_space_loss=args.endpoint_space_loss,
                    loss_type=args.diag_loss_type,
                    adaptive_p=args.diag_adaptive_p,
                    adaptive_c=args.diag_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    direction_weight=args.diag_direction_weight,
                    condition_jvp_weight=args.diag_condition_jvp_weight,
                    condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                    condition_jvp_eps=args.diag_condition_jvp_eps,
                    condition_jvp_mode=args.diag_condition_jvp_mode,
                    atom_loss_type=args.atom_loss_type,
                )
            else:
                s = sample_inner_s((B,))
                loss, comp = diagonal_distill_loss(
                    student, teacher, batch, s, t_cond,
                    endpoint_space_loss=args.endpoint_space_loss,
                    loss_type=args.diag_loss_type,
                    adaptive_p=args.diag_adaptive_p,
                    adaptive_c=args.diag_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    direction_weight=args.diag_direction_weight,
                    condition_jvp_weight=args.diag_condition_jvp_weight,
                    condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                    condition_jvp_eps=args.diag_condition_jvp_eps,
                    condition_jvp_mode=args.diag_condition_jvp_mode,
                    atom_loss_type=args.atom_loss_type,
                )
        elif args.stage == 2:
            # Stage 2: off-diagonal ESD consistency + diagonal anchor.
            jump_t_max = args.t_max if args.jump_t_max is None else args.jump_t_max
            two = torch.rand(B, 2, device=device) * jump_t_max
            s_lo, u_hi = two.min(dim=1).values, two.max(dim=1).values
            # anneal the jump: shrink toward the diagonal early in training.
            prog = min(1.0, step / max(1, int(args.anneal_frac * args.steps)))
            mid = 0.5 * (s_lo + u_hi)
            jump = (u_hi - s_lo).clamp(max=args.max_jump)  # cap the jump size
            half = 0.5 * jump * prog
            s, u = (mid - half).clamp(max=args.t_max), mid + half
            if args.consistency_loss == "esd":
                esd_loss, comp = esd_consistency_loss(
                    student, teacher, batch, s, u, t_cond,
                    target_clip=args.esd_target_clip,
                    delta_clip=args.esd_delta_clip,
                    loss_type=args.esd_loss_type,
                    adaptive_p=args.esd_adaptive_p,
                    adaptive_c=args.esd_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    atom_loss_type=args.atom_loss_type,
                    ce_shift_clip=args.esd_ce_shift_clip,
                )
            else:
                # PSD: the annealed (s, u) above become the outer jump (s, w); the
                # intermediate time is uniform in between.
                s, w = s, u
                u = s + torch.rand_like(s) * (w - s)
                esd_loss, comp = psd_consistency_loss(
                    student, teacher, batch, s, u, w, t_cond,
                    loss_type=args.esd_loss_type,
                    adaptive_p=args.esd_adaptive_p,
                    adaptive_c=args.esd_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    atom_loss_type=args.atom_loss_type,
                )
            comp["jump_cap"] = args.max_jump * prog
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
                atom_loss_type=args.atom_loss_type,
            )
            loss = args.esd_weight * esd_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    "atom_ce": dcomp["atom_ce"], "atom_kl": dcomp["atom_kl"],
                    "consistency_loss": float(esd_loss.detach()),
                    "diag_loss": float(diag_loss.detach()), **comp}
        elif args.stage == 3:
            # Stage 3: direct GLASS transition distillation on the exact coarse
            # grids used by one-/two-/four-step posterior samplers.
            n_flow_steps = int(transition_steps[torch.randint(
                len(transition_steps), (), device=device
            ).item()])
            interval_idx = int(torch.randint(n_flow_steps, (), device=device).item())
            trans_loss, comp = glass_transition_distill_loss(
                student,
                teacher,
                batch,
                t_cond,
                n_flow_steps=n_flow_steps,
                interval_idx=interval_idx,
                teacher_steps=args.transition_teacher_steps,
                loss_type=args.esd_loss_type,
                adaptive_p=args.esd_adaptive_p,
                adaptive_c=args.esd_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
            )
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
            )
            loss = args.esd_weight * trans_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    **comp}
        else:
            # Stage 4: train the realized few-step rollout against GLASS coarse
            # nodes, so gradients see compounding sampler error.
            cond_state = None
            if args.rollout_condition_source in ("generated", "guided"):
                if args.rollout_condition_source == "generated":
                    cond_state = sample_production_condition_state(
                        teacher,
                        batch_size=B,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                    )
                else:
                    if guided_hist is None or guided_target_rng is None:
                        raise RuntimeError("guided condition source was not initialized")
                    init_state = teacher._sample_noise_like_batch(batch_size=B).to(device)
                    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
                    target_values = [
                        sample_property_target(guided_hist, int(length), guided_target_rng)
                        for length in lengths
                    ]
                    property_targets = torch.tensor(
                        target_values,
                        dtype=torch.float32,
                        device=device,
                    )
                    cond_state = sample_guided_production_condition_state(
                        teacher,
                        init_state,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                        mu=args.guided_condition_mu,
                        reward_scale=args.guided_condition_reward_scale,
                        value_samples=args.guided_condition_value_samples,
                        value_glass_steps=args.guided_condition_value_glass_steps,
                        property_name=args.guided_condition_property_name,
                        property_target=property_targets,
                        property_regressor=guided_regressor,
                        guide_atomics=args.guided_condition_guide_atomics,
                        guide_min_t=args.guided_condition_guide_min_t,
                        guide_max_t=args.guided_condition_guide_max_t,
                        guide_every=args.guided_condition_guide_every,
                        guidance_max_coord_rms=args.guided_condition_max_coord_rms,
                        guidance_max_atom_rms=args.guided_condition_max_atom_rms,
                        eps=args.guided_condition_eps,
                    )
            rollout_loss, comp = glass_rollout_distill_loss(
                student,
                teacher,
                batch,
                t_cond,
                n_flow_steps=args.rollout_steps,
                teacher_steps=args.transition_teacher_steps,
                loss_type=args.esd_loss_type,
                adaptive_p=args.esd_adaptive_p,
                adaptive_c=args.esd_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                final_weight=args.rollout_final_weight,
                velocity_weight=args.rollout_velocity_weight,
                teacher_forced_velocity_weight=args.rollout_teacher_forced_velocity_weight,
                spread_weight=args.rollout_spread_weight,
                spread_samples=args.rollout_spread_samples,
                diagonal=args.rollout_diagonal,
                cond_state=cond_state,
            )
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
            )
            loss = args.esd_weight * rollout_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    **comp}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in student.state_dict().items():
                if ema[k].dtype.is_floating_point:
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)

        window["loss"].append(float(loss))
        for k, v in comp.items():
            if isinstance(v, (int, float)):
                window[k].append(float(v))
        running["loss"].append(float(loss))
        running["coord"].append(comp["coord_l2"])
        running["atom"].append(comp["atom_l2"])
        bidx = min(int(t_cond.mean().item() * (len(bins) - 1)), len(bins) - 2)
        bin_l2[bidx].append(float(loss))

        if step % args.log_every == 0:
            n = args.log_every
            extra = ""
            if args.stage == 1:
                extra = (f"  coordCos {comp.get('coord_cosine', 0):.3f}  "
                         f"atomCos {comp.get('atom_cosine', 0):.3f}  "
                         f"dir {comp.get('direction_loss', 0):.4f}  "
                         f"cjvpC {comp.get('condition_jvp_coord_l2', 0):.4f}  "
                         f"cjvpCos {comp.get('condition_jvp_coord_cosine', 0):.3f}")
            elif args.stage == 2 and args.consistency_loss == "psd":
                extra = (f"  psd_coord {np.mean(window['psd_coord_l2']):.4f}  "
                         f"psd_atom {np.mean(window['psd_atom_l2']):.4f}  "
                         f"psd_atom_kl {np.mean(window['psd_atom_kl']):.4f}  "
                         f"jump_cap {comp['jump_cap']:.3f}")
            elif args.stage == 2:
                extra = (f"  jump_cap {comp['jump_cap']:.3f}  "
                         f"esd_coord {comp.get('esd_coord_l2', 0):.4f}  "
                         f"esd_atom {comp.get('esd_atom_l2', 0):.4f}  "
                         f"delta_c {comp.get('delta_coord_norm_clipped', 0):.2f}  "
                         f"jvp_c {comp.get('jvp_coord_norm', 0):.2f}  "
                         f"tgt_c {comp.get('target_coord_norm', 0):.2f}")
            elif args.stage == 3:
                extra = (f"  trans_coord {comp.get('transition_coord_l2', 0):.4f}  "
                         f"trans_atom {comp.get('transition_atom_l2', 0):.4f}  "
                         f"grid {int(comp.get('transition_n_flow_steps', 0))}  "
                         f"idx {int(comp.get('transition_interval_idx', 0))}  "
                         f"tgt_c {comp.get('transition_target_coord_norm', 0):.2f}")
            elif args.stage == 4:
                extra = (f"  roll_coord {comp.get('rollout_coord_l2', 0):.4f}  "
                         f"roll_atom {comp.get('rollout_atom_l2', 0):.4f}  "
                         f"final_c {comp.get('rollout_final_coord_l2', 0):.4f}  "
                         f"final_a {comp.get('rollout_final_atom_l2', 0):.4f}  "
                         f"vel_c {comp.get('rollout_velocity_coord_l2', 0):.4f}  "
                         f"tfv_c {comp.get('rollout_teacher_forced_velocity_coord_l2', 0):.4f}  "
                         f"spr {comp.get('rollout_coord_spread_l2', 0):.4f}  "
                         f"sprR {comp.get('rollout_final_coord_spread_ratio', 0):.2f}  "
                         f"diagR {int(comp.get('rollout_diagonal', 0))}  "
                         f"steps {int(comp.get('rollout_steps', 0))}")
            if args.stage in (1, 2):
                extra += (f"  atom_kl {np.mean(window['atom_kl']):.4f}"
                          + (f"  esd_atom_kl {np.mean(window['esd_atom_kl']):.4f}"
                             if args.stage == 2 and args.consistency_loss == "esd" else ""))
            print(f"step {step:5d}  loss {np.mean(running['loss'][-n:]):.4f}  "
                  f"diag_coord {np.mean(running['coord'][-n:]):.4f}  "
                  f"diag_atom {np.mean(running['atom'][-n:]):.4f}{extra}", flush=True)
            if wandb_run is not None:
                wandb_run.log({**{f"train/{k}": float(np.mean(v)) for k, v in window.items()},
                               "train/lr": cur_lr}, step=step)
            window.clear()

        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step)
            torch.save(
                {"student": ema, "margs": vars(margs), "data_stats": data_stats,
                 "step": step, "args": vars(args),
                 "student_cfg": {
                     "hidden_dim": student.hidden_dim,
                     "num_layers": student.num_layers,
                     "num_heads": args.num_heads or margs.num_heads,
                     "block_conditioning": args.block_conditioning,
                     "velocity_parametrization": args.velocity_parametrization,
                     "encoder_depth": student.encoder_depth,
                     "time_encoding_max_len": args.time_encoding_max_len,
                 }},
                out_dir / f"student_step_{step}.pt",
            )

    print(f"\nDone. Checkpoints in {out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
