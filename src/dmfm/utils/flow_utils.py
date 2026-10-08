"""Gaussian-interpolant helpers for the DNA flow models.

Ported from DNA-MFM@187fe7b ``utils/flow_utils.py``. Only the functions used by
the Gaussian (``--mode gaussian``) DiT path are kept: ``gaussian_beta``,
``gaussian_denoiser_flow_step``, ``update_ema`` and ``get_wasserstein_dist``.
The Dirichlet-FM / Riemannian / diffusion-schedule utilities of the original
Dirichlet-flow-matching code base are not used by any paper experiment and were
dropped.
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm


_GAUSSIAN_BETA_TABLE_CACHE: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def gaussian_beta(args, t_in: torch.Tensor, *, return_deriv: bool = False):
    """
    Global helper for gaussian-mode interpolant scheduling.

    - Uses args.gaussian_beta_schedule in {"linear","table"}.
    - If "table", loads args.gaussian_beta_table_path (torch-saved dict with 1D 't' and 'beta').
    - Returns beta(t); if return_deriv=True also returns beta'(t) (piecewise-constant slopes).
    """
    schedule = getattr(args, "gaussian_beta_schedule", "linear")
    if schedule == "linear":
        beta = t_in
        if return_deriv:
            return beta, torch.ones_like(beta)
        return beta

    if schedule != "table":
        raise ValueError(f"Unknown gaussian_beta_schedule={schedule!r}")

    path = getattr(args, "gaussian_beta_table_path", None)
    if path is None:
        raise ValueError("--gaussian_beta_schedule table requires --gaussian_beta_table_path")

    key = (path, str(t_in.device))
    if key not in _GAUSSIAN_BETA_TABLE_CACHE:
        table = torch.load(path, map_location="cpu")
        if not (isinstance(table, dict) and "t" in table and "beta" in table):
            raise ValueError("beta table must be a dict with keys 't' and 'beta'")
        tt = table["t"].float().flatten()
        bb = table["beta"].float().flatten()
        if tt.numel() < 2 or bb.numel() != tt.numel():
            raise ValueError("beta table tensors must have same length >= 2")
        if not bool(torch.all(tt[1:] >= tt[:-1])):
            order = torch.argsort(tt)
            tt = tt[order]
            bb = bb[order]
        _GAUSSIAN_BETA_TABLE_CACHE[key] = (tt.to(t_in.device), bb.to(t_in.device))

    tt, bb = _GAUSSIAN_BETA_TABLE_CACHE[key]
    t = t_in.clamp(float(tt[0].item()), float(tt[-1].item()))
    idx = torch.bucketize(t, tt).clamp(1, tt.numel() - 1)
    t0 = tt[idx - 1]
    t1 = tt[idx]
    b0 = bb[idx - 1]
    b1 = bb[idx]
    w = (t - t0) / (t1 - t0 + 1e-12)
    beta = (1.0 - w) * b0 + w * b1
    if not return_deriv:
        return beta
    deriv = (b1 - b0) / (t1 - t0 + 1e-12)
    return beta, deriv


def gaussian_denoiser_flow_step(
    args,
    model,
    x: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    *,
    cls: Optional[torch.Tensor] = None,
    x_sc: Optional[torch.Tensor] = None,
    flow_temp: Optional[float] = None,
    eps: float = 1e-8,
    legacy_double_eval: bool = False,
):
    """One exact Gaussian denoiser-flow step.

    The model predicts denoiser logits for E[x1 | x_s]. The state update is the
    closed-form linear interpolant step
        x_t = gamma x_s + delta softmax(logits),
    with gamma=(1-beta_t)/(1-beta_s) and delta=(beta_t-beta_s)/(1-beta_s).
    For beta(t)=t this equals the Euler update used previously, but this form
    also stays exact for table/nonlinear beta schedules.

    Port change (2026-09-29): the original (DNA-MFM@187fe7b) evaluated the network
    twice per step, once for the returned ``logits`` and once more inside
    ``psi_st(..., flow_temp=temp)`` for the denoiser. The denoiser is now derived
    from the same logits (``log_softmax(logits / temp).exp()``, exactly what
    ``psi_st`` computes), so each step costs one network evaluation; outputs are
    identical. ``legacy_double_eval=True`` restores the original two-call path.
    The paper's NFE counts count one evaluation per step, i.e. they describe the
    single-evaluation cost; the original code spent 2x the wall-clock on these steps.
    """
    if s.ndim == 0:
        s = s.expand(x.shape[0])
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    s = s.to(device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)

    kwargs = {}
    if cls is not None:
        kwargs["cls"] = cls
    if x_sc is not None:
        kwargs["x_sc"] = x_sc
    temp = float(flow_temp if flow_temp is not None else getattr(args, "flow_temp", 1.0))
    
    logits = model.psi_st(x, s, s, return_logits=True, **kwargs)
    if legacy_double_eval:
        denoiser = model.psi_st(x, s, s, flow_temp=temp, **kwargs)
    else:
        denoiser = F.log_softmax(logits / temp, dim=-1).exp()

    beta_s = gaussian_beta(args, s)
    beta_t = gaussian_beta(args, t)
    denom = (1.0 - beta_s).clamp_min(float(eps))
    gamma = (1.0 - beta_t) / denom
    delta = (beta_t - beta_s) / denom
    x_next = model._X_st_psi(x, s, t, denoiser)
    
    return x_next, logits, denoiser


def update_ema(current_dict, prev_ema, gamma = 0.9):
    ema = copy.deepcopy(prev_ema)
    current_dict = copy.deepcopy(current_dict)
    for key, current_value in current_dict.items():
        ema_key  = 'ema_' + key
        if not np.isnan(current_value):
            if ema_key in prev_ema:
                ema[ema_key] = (1 - gamma) * current_value + gamma * prev_ema[ema_key]
            else:
                ema[ema_key] = current_value
    return ema

def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist
