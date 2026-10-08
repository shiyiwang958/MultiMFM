"""DPS steering for DNA: the denoiser approximation as a guidance baseline.

The QM9 comparison includes DPS (\\cite{chung2024diffusionposteriorsamplinggeneral}) and
the DNA one did not, for no reason beyond which sweep each was written in. DPS already
appears in the DNA *value-estimation* benchmark (`value_error.dps_endpoint`), where it is
the N-independent denoiser point estimate; this turns that same estimator into a steering
baseline so both domains are compared against the same methods.

DPS replaces the Monte-Carlo value

    V_t(x) = log E[exp r(X_1) | X_t = x]        (what the dMFM estimates with N samples)

by its denoiser approximation

    V_t(x) ~ r( E[X_1 | X_t = x] ) = r( psi(x, t) ),

so one network call per guided step and one backward, with no sampling. Everything else --
the base trajectory, the guide, the reward, the gradient clipping, the guidance coefficient
and the step -- is shared with `dmfm_steering.sample_dmfm_guided`, so the only difference
between the two arms is the value estimator.

Budget. DPS has no sampling knob: at a fixed number of outer steps its cost is fixed. Its
budget axis is therefore the number of integration steps, exactly as in the QM9 sweep
(`scripts/qm9/baseline_sweep_extend.sbatch`), plus the guidance window.
"""

from __future__ import annotations

import torch

from dmfm.benchmarks.core import BaseFlow, C0Guide, NFECounter, SamplerConfig
from dmfm.benchmarks.search import RewardSpec
from dmfm.benchmarks.dmfm_steering import GuidanceConfig, _step_base


def dps_value_and_grad(
    flow: BaseFlow,
    student,
    guide: C0Guide,
    reward: RewardSpec,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    nfe: NFECounter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of the denoiser-approximation value, r(psi(x_t, t)).

    Uses the student's diagonal denoiser when one is loaded -- the estimator
    ``value_error.dps_endpoint`` scores -- and the base model's denoiser otherwise, so the
    baseline never gets a network the other arms do not have.
    """
    x_leaf = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        if student is not None:
            x1 = student.Psi_st(x_leaf, t, t, t_cond=t, x_cond=x_leaf)
            nfe.student += int(x.shape[0])
        else:
            x1 = flow.denoiser(x_leaf, t)   # BaseFlow.denoiser already counts its own NFE
        score = guide.model(x1)
        nfe.guide += int(x1.shape[0])
        V = -0.5 * float(reward.scale) * ((score - reward.target) / reward.sigma).pow(2)
        grad = torch.autograd.grad(V.sum(), x_leaf)[0]
    nfe.backward += int(x.shape[0])
    return V.detach().reshape(-1), grad.detach()


def sample_dps_guided(
    flow: BaseFlow,
    student,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    sampler: SamplerConfig,
    guidance: GuidanceConfig,
    batch_size: int = 25,
    nfe: NFECounter,
) -> dict:
    """``sample_dmfm_guided`` with the denoiser approximation in place of the MC value."""
    scores, tokens = [], []
    done = 0
    while done < n_outputs:
        b = min(batch_size, n_outputs - done)
        x = flow.prior(b)
        grid = sampler.grid(x.device, x.dtype)
        for k in range(int(sampler.n_steps)):
            s0, s1 = grid[k], grid[k + 1]
            s = s0.expand(x.shape[0])
            dt = float(s1 - s0)
            x_base = _step_base(flow, x, s, dt, sampler)
            if guidance.t_start <= float(s0) <= guidance.t_end:
                _, g = dps_value_and_grad(flow, student, guide, reward, x, s, nfe=nfe)
                gnorm = g.flatten(1).norm(dim=1).clamp_min(1e-8)
                if guidance.grad_clip is not None:
                    sc = float(guidance.grad_clip) / gnorm
                    g = g * (sc if guidance.grad_normalize else sc.clamp(max=1.0))[:, None, None]
                coeff = float(guidance.guidance_frac) * flow.model.sde_sigma_sq(s)
                coeff = coeff.clamp(max=float(guidance.coeff_cap))
                x = x_base + dt * coeff[:, None, None] * g
            else:
                x = x_base
        with torch.no_grad():
            _, s_hat = reward(guide, x.detach())
        scores.append(s_hat)
        tokens.append(x.detach().argmax(dim=-1))
        done += b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"score": torch.cat(scores), "tokens": torch.cat(tokens)}
