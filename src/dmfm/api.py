"""Stable, documented entry points into the DNA (dMFM) code.

This is a thin facade over the ported experiment modules: every function here
delegates to (or reproduces line for line) the code path that produced the paper
numbers, so building on it gives the same numerics as the ``dmfm.experiments``
scripts. See ``docs/provenance/dna/API.md`` for a cheat sheet with examples.

Conventions
-----------
* Time runs from 0 (Gaussian noise) to 1 (data). The interpolant is
  ``x_t = beta(t) * onehot(x1) + (1 - beta(t)) * eps`` with ``beta(t) = t``.
* States are float tensors ``[B, L, 4]`` over the alphabet A, C, G, T
  (channel order 0..3). Tokens are ``LongTensor[B, L]``; ``argmax(-1)`` decodes.
* A *posterior sampler* maps terminal noise ``eps [B, L, 4]`` and a condition
  ``(x_cond [B, L, 4], t_cond [B])`` to an endpoint ``[B, L, 4]`` (approximately
  on the simplex); it is a sample of ``x_1 | x_t = x_cond``.
* All loaders return frozen models in ``eval()`` mode.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel, DNAMFMStudent
from dmfm.utils.flow_utils import gaussian_beta, gaussian_denoiser_flow_step
from dmfm.utils.torch_io import torch_load

DNA_ALPHABET = "ACGT"
DNA_TO_INT = {b: i for i, b in enumerate(DNA_ALPHABET)}
SPLITS = ("a_train", "a_val", "b_train", "b_val", "test")

PosteriorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
LogRewardFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "DNA_ALPHABET",
    "SPLITS",
    # loading
    "load_base",
    "load_dmfm",
    "load_c0",
    # data
    "load_windows",
    "split_indices",
    "load_split_seqs",
    "load_c0_labels",
    # encoding
    "tokens_to_strings",
    "strings_to_tokens",
    "one_hot",
    "reverse_complement",
    # noise / interpolant
    "seeded_randn",
    "paired_initial_noise",
    "paired_eps_pool",
    "forward_noise",
    # samplers
    "sample_unguided",
    "glass_posterior",
    "dmfm_posterior",
    "glass_posterior_fn",
    "dmfm_posterior_fn",
    # values
    "c0_log_reward_fn",
    "value_and_grad",
    "guided_sample",
    "c0_guided_pairs",
    "TABLE1_SETTINGS",
    # scoring
    "oracle_score",
]


# ---------------------------------------------------------------- loading

def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_base(length: int = 50, device=None, *, ckpt: str | Path | None = None) -> tuple[DiTSequenceModel, SimpleNamespace]:
    """Base DFM DiT (``checkpoints/dna/base/L{length}/best.pt``).

    Returns ``(model, cfg)``. ``cfg`` is the checkpoint's ``model_cfg`` (seq_len,
    alphabet_size, beta schedule, flow_temp, ...) and is what the samplers take as
    ``args``. Loaded exactly like ``ablate_glass_gradient_mc.load_model`` /
    ``sample_c0_guidance`` (strict state dict, no flash attention).
    """
    from dmfm.experiments.ablate_glass_gradient_mc import load_model

    device = _resolve_device(device)
    path = Path(ckpt) if ckpt is not None else paths.base_ckpt(length)
    model, cfg = load_model(str(path), device)
    cfg.use_flash_attn = bool(getattr(cfg, "use_flash_attn", False))
    return model, cfg


def load_dmfm(length: int = 50, device=None, *, kind: str = "dmfm", ckpt: str | Path | None = None) -> DNAMFMStudent:
    """dMFM student. ``kind="dmfm"``: the diagonal+ESD students (Tables 9, 13, 17, and
    the dMFM used for steering); ``kind="dmfm4"``: the 4-step ESD students (Table 18).

    ``ckpt`` overrides the file (its directory must contain ``args.json``). Raises if
    any key is missing/unexpected.
    """
    from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import load_dmfm as _load

    device = _resolve_device(device)
    if ckpt is None:
        if kind == "dmfm":
            ckpt = paths.dmfm_ckpt(length)
        elif kind in ("dmfm4", "dmfm_4step", "4step"):
            ckpt = paths.dmfm4_ckpt(length)
        else:
            raise ValueError(f"kind must be 'dmfm' or 'dmfm4', got {kind!r}")
    return _load(str(ckpt), device)


def load_c0(role: str = "guide", device=None, *, ckpt: str | Path | None = None) -> torch.nn.Module:
    """C0 regressor (``park_cnn``): ``role="guide"`` (reward model, trained on parent
    set a) or ``role="oracle"`` (independent evaluator, parent set b). Input is
    ``[B, 50, 4]`` A/C/G/T probabilities, output ``[B]``."""
    from dmfm.regressors.c0 import load_c0_regressor

    if role not in ("guide", "oracle"):
        raise ValueError(f"role must be 'guide' or 'oracle', got {role!r}")
    path = Path(ckpt) if ckpt is not None else (paths.c0_guide() if role == "guide" else paths.c0_oracle())
    return load_c0_regressor(path, "park_cnn", device=_resolve_device(device))


# ------------------------------------------------------------------- data

def load_windows(length: int = 50) -> dict:
    """The full window table for one length: ``seqs LongTensor[N, L]``, ``parent_id``,
    ``window_start`` (+ ``c0`` float labels at L=50)."""
    return torch_load(paths.data_pt(length), map_location="cpu")


def split_indices(length: int, split: str) -> torch.Tensor:
    """Row indices of a parent-disjoint split: one of ``a_train, a_val, b_train, b_val,
    test``. Set a trains the generators and the C0 guide, set b the C0 oracle; ``test``
    is the 58 held-out parents (8,294 windows at L=50)."""
    from dmfm.utils.yeast_splits import load_yeast_split_indices

    return load_yeast_split_indices(paths.split_pt(length), split)


def load_split_seqs(length: int, split: str = "test", max_n: int | None = None) -> torch.Tensor:
    """``LongTensor[N, L]`` token windows of one split (``split="full"`` = all)."""
    from dmfm.utils.model_loading import load_data_seqs

    return load_data_seqs(paths.data_pt(length), max_n=max_n, split_pt=paths.split_pt(length), split=split)


def load_c0_labels(split: str = "test") -> tuple[torch.Tensor, torch.Tensor]:
    """L=50 windows and their measured C0 labels for one split: ``(seqs [N, 50], c0 [N])``."""
    payload = load_windows(50)
    idx = split_indices(50, split)
    return payload["seqs"].long()[idx], torch.as_tensor(payload["c0"]).float()[idx]


# --------------------------------------------------------------- encoding

def tokens_to_strings(tokens: torch.Tensor) -> list[str]:
    """``LongTensor[B, L]`` (or a ``[B, L, 4]`` state, decoded by argmax) -> A/C/G/T strings."""
    if tokens.ndim == 3:
        tokens = tokens.argmax(-1)
    table = np.asarray(list(DNA_ALPHABET))
    return ["".join(table[row].tolist()) for row in tokens.detach().cpu().numpy()]


def strings_to_tokens(seqs: Sequence[str]) -> torch.Tensor:
    return torch.tensor([[DNA_TO_INT[b] for b in s.upper()] for s in seqs], dtype=torch.long)


def one_hot(tokens: torch.Tensor) -> torch.Tensor:
    return F.one_hot(tokens.long(), num_classes=4).float()


def reverse_complement(x_acgt: torch.Tensor) -> torch.Tensor:
    """Reverse complement of a ``[B, L, 4]`` A/C/G/T state."""
    return x_acgt.flip(dims=(1,))[..., [3, 2, 1, 0]]


# ------------------------------------------------------ noise/interpolant

def seeded_randn(shape, seed: int, device=None) -> torch.Tensor:
    """``torch.randn`` from a CPU generator seeded with ``seed`` (the scripts' RNG convention)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=g, dtype=torch.float32).to(_resolve_device(device))


def paired_initial_noise(sample_ids: Iterable[int], length: int, seed: int = 0, device=None) -> torch.Tensor:
    """Initial noise of the Table 1 sampler: sample ``i`` uses ``seeded_randn((1, L, 4), seed + i)``.
    Identical ids give identical unguided samples regardless of batch composition."""
    return torch.cat([seeded_randn((1, length, 4), seed + int(i), device) for i in sample_ids], dim=0)


def paired_eps_pool(sample_ids: Iterable[int], length: int, mc: int, seed: int = 0, device=None) -> torch.Tensor:
    """Frozen per-sample MC terminal-noise pool of the Table 1 sampler: ``[B, mc, L, 4]``,
    sample ``i`` uses ``seeded_randn((mc, L, 4), seed + 1_000_000 + i)``."""
    return torch.stack([seeded_randn((mc, length, 4), seed + 1_000_000 + int(i), device) for i in sample_ids], dim=0)


def forward_noise(tokens_or_x1: torch.Tensor, t, cfg, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Noise clean data to time ``t``: ``beta(t) * onehot(x1) + (1 - beta(t)) * noise``
    (as in ``eval_posterior_diversity`` / ``make_heldout_probe_states``)."""
    x1 = one_hot(tokens_or_x1) if tokens_or_x1.dtype == torch.long else tokens_or_x1.float()
    if noise is None:
        noise = torch.randn_like(x1)
    t = torch.as_tensor(t, dtype=x1.dtype, device=x1.device)
    if t.ndim == 0:
        t = t.expand(x1.shape[0])
    beta = gaussian_beta(cfg, t).reshape(-1, 1, 1)
    return beta * x1 + (1.0 - beta) * noise.to(x1.device)


# ---------------------------------------------------------------- samplers

@torch.no_grad()
def sample_unguided(
    base: DiTSequenceModel,
    cfg,
    x0: torch.Tensor,
    *,
    nfe: int = 64,
    t_max: float = 0.95,
    t_start: float = 0.0,
) -> torch.Tensor:
    """Integrate the base DFM probability flow from ``x0`` at ``t_start`` to ``t_max`` with
    ``nfe`` exact denoiser-flow steps on a uniform grid; returns the state (decode with
    ``argmax(-1)``). The Table 1 unguided branch is ``nfe=64, t_max=0.95`` from
    :func:`paired_initial_noise`; the calibration / probe code uses ``t_max=0.999``."""
    x = x0
    grid = torch.linspace(float(t_start), float(t_max), int(nfe) + 1, device=x.device)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        x, _, _ = gaussian_denoiser_flow_step(cfg, base, x, s0.expand(x.shape[0]), s1.expand(x.shape[0]))
    return x


def glass_posterior(
    base: DiTSequenceModel,
    eps: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    *,
    n_steps: int,
    end_time: float = 1.0,
    solver: str = "euler",
) -> torch.Tensor:
    """GLASS reference posterior sampler on the base DFM (the "teacher" posterior).

    ``end_time=1.0, solver="euler"`` is the Table 1 / Fig 5 / Tables 14-15 / Table 17
    integrator (``linspace(0, 1, n_steps+1)``); Table 18 used ``end_time=0.999`` with
    ``solver="rk4"``. Differentiable w.r.t. ``x_cond``.
    """
    from dmfm.experiments.ablate_glass_gradient_mc import glass_integrate_diff

    return glass_integrate_diff(base, eps, x_cond, t_cond, int(n_steps), end_time=end_time, solver=solver)


def dmfm_posterior(
    student: DNAMFMStudent,
    eps: torch.Tensor,
    x_cond: torch.Tensor,
    t_cond: torch.Tensor,
    *,
    sampler: str = "flow_map",
    n_steps: int = 1,
    end_time: float = 1.0,
) -> torch.Tensor:
    """dMFM posterior sampler ``X_{0->1}(eps | t_cond, x_cond)``.

    * ``"flow_map"``: compose ``n_steps`` learned two-time maps on ``linspace(0, end_time)``
      (``n_steps=1, end_time=1`` is the one-step map; Table 17 uses ``n_steps=1``).
    * ``"one_step"``: ``student(0, 1, eps, t_cond, x_cond)`` (== flow_map, 1 step).
    * ``"diagonal_rk4"``: RK4 on the diagonal velocity, needs ``end_time < 1``.
    * ``"diagonal_euler"``: Euler on the diagonal velocity to 1.
    Differentiable w.r.t. ``x_cond``.
    """
    from dmfm.experiments.ablate_dmfm_one_step_gradient_mc import sample_dmfm

    if sampler == "diagonal_euler":
        x = eps
        grid = torch.linspace(0.0, float(end_time), int(n_steps) + 1, device=x.device, dtype=x.dtype)
        for r0, r1 in zip(grid[:-1], grid[1:]):
            b = x.shape[0]
            x = x + (r1 - r0) * student.v(r0.expand(b), r0.expand(b), x, t_cond, x_cond)
        return x
    return sample_dmfm(student, eps, x_cond, t_cond, sampler=sampler, n_steps=int(n_steps), end_time=float(end_time))


def glass_posterior_fn(base: DiTSequenceModel, *, n_steps: int, end_time: float = 1.0, solver: str = "euler") -> PosteriorFn:
    """Bind :func:`glass_posterior` into a ``(eps, x_cond, t_cond) -> endpoint`` callable."""
    return lambda eps, x_cond, t_cond: glass_posterior(base, eps, x_cond, t_cond, n_steps=n_steps, end_time=end_time, solver=solver)


def dmfm_posterior_fn(student: DNAMFMStudent, *, sampler: str = "flow_map", n_steps: int = 1, end_time: float = 1.0) -> PosteriorFn:
    """Bind :func:`dmfm_posterior` into a ``(eps, x_cond, t_cond) -> endpoint`` callable."""
    return lambda eps, x_cond, t_cond: dmfm_posterior(student, eps, x_cond, t_cond, sampler=sampler, n_steps=n_steps, end_time=end_time)


# ------------------------------------------------------------------ values

def c0_log_reward_fn(
    guide: torch.nn.Module,
    target_c0: float,
    *,
    reward_sigma: float = 0.15,
    reward_scale: float = 0.5,
    target_c0_second: float | None = None,
) -> LogRewardFn:
    """Scaled Table 1 log-reward ``reward_scale * -0.5 ((guide(x1) - target) / sigma)^2``
    evaluated on the *soft* endpoint (``= -11.1 (f - y*)^2`` at the defaults)."""
    from dmfm.experiments.sample_c0_guidance import target_reward

    def fn(endpoint: torch.Tensor) -> torch.Tensor:
        return float(reward_scale) * target_reward(guide(endpoint), target_c0, target_c0_second, reward_sigma)

    return fn


def value_and_grad(
    posterior: PosteriorFn,
    log_reward: LogRewardFn,
    x: torch.Tensor,
    t: float,
    eps_pool: torch.Tensor,
    *,
    mc_chunk: int = 16,
    need_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finite-MC value ``V(x) = log mean_j exp(log_reward(posterior(eps_j | t, x)))`` and its
    exact autodiff gradient w.r.t. ``x``.

    ``eps_pool`` is ``[B, MC, L, 4]``. Two-pass chunked estimator identical to
    ``sample_c0_guidance.estimate_v_and_grad`` and ``ablate_*_gradient_mc.estimate_value_gradient``:
    pass 1 computes all log-rewards without grad (softmax weights), pass 2 back-propagates
    ``sum_j w_j * log_reward_j`` per chunk. Returns ``(V [B], grad [B, L, 4])`` (``grad`` is
    ``None`` when ``need_grad=False``).
    """
    batch, mc = eps_pool.shape[:2]
    chunk = max(1, min(int(mc_chunk), mc))
    with torch.no_grad():
        parts = []
        for j0 in range(0, mc, chunk):
            j1 = min(mc, j0 + chunk)
            w = j1 - j0
            eps = eps_pool[:, j0:j1].reshape(batch * w, *x.shape[1:])
            x_rep = x.detach().repeat_interleave(w, 0)
            t_rep = torch.full((batch * w,), float(t), device=x.device, dtype=x.dtype)
            parts.append(log_reward(posterior(eps, x_rep, t_rep)).view(batch, w))
        log_r = torch.cat(parts, dim=1)
        value = torch.logsumexp(log_r, dim=1) - math.log(mc)
        weights = torch.softmax(log_r, dim=1)
    if not need_grad:
        return value.detach(), None

    x_leaf = x.detach().clone().requires_grad_(True)
    grad = torch.zeros_like(x_leaf)
    for j0 in range(0, mc, chunk):
        j1 = min(mc, j0 + chunk)
        w = j1 - j0
        eps = eps_pool[:, j0:j1].reshape(batch * w, *x.shape[1:])
        x_rep = x_leaf.repeat_interleave(w, 0)
        t_rep = torch.full((batch * w,), float(t), device=x.device, dtype=x.dtype)
        r = log_reward(posterior(eps, x_rep, t_rep)).view(batch, w)
        obj = (weights[:, j0:j1].detach() * r).sum()
        grad = grad + torch.autograd.grad(obj, x_leaf)[0]
    return value.detach(), grad.detach()


def guided_sample(
    base: DiTSequenceModel,
    cfg,
    x0: torch.Tensor,
    value_grad: Callable[[torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]],
    *,
    nfe_traj: int = 64,
    t_max: float = 0.95,
    guide_t_start: float = 0.50,
    guide_t_end: float = 0.95,
    guidance_frac: float = 8.0,
    coeff_cap: float | None = 10.0,
    grad_clip: float | None = 10.0,
    taper: bool = False,
    return_unguided: bool = False,
    grad_normalize: bool = False,
):
    """Value-gradient guided sampling on the base DFM (the loop of ``sample_c0_guidance``).

    At each of ``nfe_traj`` steps on ``linspace(0, t_max)``: take the base flow step; if
    ``guide_t_start <= t <= guide_t_end`` add ``dt * min(guidance_frac * sigma_t^2, coeff_cap)
    * clip(grad V(x_t), grad_clip)`` with ``sigma_t^2 = 2 (1/t - 1)``. ``value_grad(x, t)`` must
    return ``(V, grad)``, e.g. ``lambda x, t: value_and_grad(post, logr, x, t, eps_pool)``.
    With ``return_unguided=True`` also returns the paired unguided endpoint from the same ``x0``.

    ``grad_normalize=True`` (default ``False``, i.e. every stored result is unaffected) rescales
    the gradient to *exactly* ``grad_clip`` instead of only clipping it down, making the guidance
    step scale-free in the gradient. Use it when comparing posterior samplers whose gradient
    magnitudes differ: the Table 1 settings (``guidance_frac 8``, ``coeff_cap 10``,
    ``grad_clip 10``) were tuned to the GLASS-4 gradient scale (mean norm ~3 at L=50, clipped at
    ~6% of guided steps), while a dMFM flow-map posterior has a mean norm of 10-80 there and is
    clipped at 13-52% of steps, so at the stock settings the dMFM arm is driven much harder than
    the GLASS arm. See ``docs/dmfm_glass_parity.md``.
    """
    x_base = x0.clone()
    x_guided = x0.clone()
    grid = torch.linspace(0, float(t_max), int(nfe_traj) + 1, device=x0.device)
    batch = x0.shape[0]
    for s0, s1 in zip(grid[:-1], grid[1:]):
        t_now = float(s0)
        s = s0.expand(batch)
        t_next = s1.expand(batch)
        dt = float(s1 - s0)
        with torch.no_grad():
            if return_unguided:
                x_base, _, _ = gaussian_denoiser_flow_step(cfg, base, x_base, s, t_next)
            x_guided_base, _, _ = gaussian_denoiser_flow_step(cfg, base, x_guided, s, t_next)
        if guide_t_start <= t_now <= guide_t_end:
            _, g = value_grad(x_guided, t_now)
            gnorm = g.flatten(1).norm(dim=1).clamp_min(1e-8)
            if grad_clip is not None:
                scale = grad_clip / gnorm
                g = g * (scale if grad_normalize else scale.clamp(max=1.0))[:, None, None]
            elif grad_normalize:
                raise ValueError("grad_normalize=True needs a grad_clip to normalise to")
            coeff = guidance_frac * base.sde_sigma_sq(s)
            if coeff_cap is not None:
                coeff = coeff.clamp(max=coeff_cap)
            if taper:
                width = max(guide_t_end - guide_t_start, 1e-8)
                coeff = coeff * ((guide_t_end - s) / width).clamp(min=0.0, max=1.0)
            x_guided = x_guided_base + dt * coeff[:, None, None] * g
        else:
            x_guided = x_guided_base
    if return_unguided:
        return x_guided.detach(), x_base.detach()
    return x_guided.detach()


#: Exact Table 1 / Fig 3 settings (``scripts/dna/table1_c0_guidance.sbatch``).
TABLE1_SETTINGS = dict(
    seed=0,
    batch_size=8,
    reward_sigma=0.15,
    reward_scale=0.5,
    mc=8,
    mc_chunk=8,
    nfe_traj=64,
    nfe_value=4,
    t_max=0.95,
    guide_t_start=0.50,
    guide_t_end=0.95,
    guidance_frac=8.0,
    coeff_cap=10.0,
    grad_clip=10.0,
    taper=False,
    target_c0_second=None,
)


def c0_guided_pairs(
    base: DiTSequenceModel,
    cfg,
    guide: torch.nn.Module,
    sample_ids: Sequence[int],
    *,
    target_c0: float,
    **overrides,
) -> list[dict]:
    """Paired unguided/guided C0 samples exactly as ``dmfm.experiments.sample_c0_guidance``
    (Table 1 / Fig 3): GLASS value gradients on the base DFM, MC=8, 4-step Euler GLASS.

    ``sample_ids`` is one batch (Table 1 used batches of 8: ``[0..7], [8..15], ...``; the
    batch composition is part of the guided RNG stream only through cuDNN nondeterminism,
    the noise itself is per-id). Returns one dict per sample with ``unguided``/``guided``
    guide scores (hard one-hot), distances to target and ``seq_unguided``/``seq_guided``.
    Keyword overrides replace entries of :data:`TABLE1_SETTINGS`.
    """
    from dmfm.experiments.sample_c0_guidance import sample_paired_batch

    settings = dict(TABLE1_SETTINGS)
    unknown = set(overrides) - set(settings)
    if unknown:
        raise TypeError(f"unknown settings {sorted(unknown)}")
    settings.update(overrides)
    ns = argparse.Namespace(
        target_c0=float(target_c0),
        gaussian_beta_schedule=getattr(cfg, "gaussian_beta_schedule", "linear"),
        gaussian_beta_table_path=getattr(cfg, "gaussian_beta_table_path", None),
        flow_temp=float(getattr(cfg, "flow_temp", 1.0)),
        **settings,
    )
    device = next(base.parameters()).device
    return sample_paired_batch(
        sample_ids=list(sample_ids),
        args=ns,
        gen_model=base,
        c0_model=guide,
        device=device,
        seq_len=int(cfg.seq_len),
        alphabet_size=int(cfg.alphabet_size),
    )


# ----------------------------------------------------------------- scoring

@torch.no_grad()
def oracle_score(
    model: torch.nn.Module,
    seqs,
    *,
    rc_average: bool = True,
    batch_size: int = 512,
) -> np.ndarray:
    """Score 50 bp sequences with a C0 regressor (oracle or guide) as ``score_c0_oracle``:
    one-hot input, averaged with the reverse complement when ``rc_average`` (Table 1 and
    Table 11 both use it). ``seqs``: list of strings, ``LongTensor[N, 50]`` tokens, or a
    ``[N, 50, 4]`` state (decoded by argmax first)."""
    if isinstance(seqs, torch.Tensor):
        tokens = seqs.argmax(-1) if seqs.ndim == 3 else seqs
    else:
        tokens = strings_to_tokens(list(seqs))
    x = one_hot(tokens.cpu())
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(x), batch_size):
        b = x[start : start + batch_size].to(device)
        p = model(b)
        if rc_average:
            p = 0.5 * (p + model(reverse_complement(b)))
        out.append(p.detach().cpu())
    return torch.cat(out).numpy()
