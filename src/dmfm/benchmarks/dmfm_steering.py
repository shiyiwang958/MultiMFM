"""dMFM value-gradient steering and the posterior-sample value estimators.

The dMFM curve of Figure 4 (right) and the dMFM curve of Figure 4 (left) both
rest on the same object: N samples of the posterior p_{1|t}(.|x_t), from which

    V_hat_t(x) = logsumexp_i r(Phi(eps_i; t, x)) - log N

and its gradient in x are formed. Three ways of drawing those samples are used:

``dmfm``   the distilled Meta Flow Map, ``n_steps`` map evaluations per sample
           (``n_steps = 1`` is the NFE=1 setting of the paper);
``glass``  the GLASS posterior ODE run on the *base* model, ``n_steps``
           evaluations per sample -- the expensive but model-exact estimator used
           as the reference in Fig 5 / Tables 14-15 and here in Fig 4 (left);
``denoiser`` / ``fmap``  the two N-independent approximations V ~ r(x_1_hat) that
           Fig 4 (left) compares against.

The steering loop is a port of ``scripts/benchmark_yeast_split_dmfm_steering.py``
(the producer of the submitted right-hand panel), with the same defaults:
96 interpolant steps, guidance on t in [0.01, 0.95], coefficient
``guidance_frac * sigma_t^2`` capped at 10, gradient-norm clip 10,
``reward_sigma = 0.15``, ``reward_scale = 1``, target C0 = 0.30.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dmfm import api
from dmfm.benchmarks.core import BaseFlow, C0Guide, NFECounter, SamplerConfig
from dmfm.benchmarks.search import RewardSpec


@dataclass
class GuidanceConfig:
    """Defaults are the argparse defaults of the original benchmark script."""

    guidance_frac: float = 8.0
    coeff_cap: float = 10.0
    grad_clip: float = 10.0
    t_start: float = 0.01
    t_end: float = 0.95
    n_mc: int = 1
    nfe_value: int = 1
    grad_normalize: bool = False

    def guided_steps(self, sampler: SamplerConfig) -> int:
        grid = sampler.grid(torch.device("cpu"))
        return int(sum(1 for s0 in grid[:-1] if self.t_start <= float(s0) <= self.t_end))


class PosteriorSampler:
    """Draw N posterior samples of p_{1|t}(.|x_t) and count the NFE they cost."""

    def __init__(self, flow: BaseFlow, student=None, nfe: NFECounter | None = None):
        self.flow = flow
        self.student = student
        self.nfe = nfe if nfe is not None else flow.nfe

    def dmfm(self, eps: torch.Tensor, x_cond: torch.Tensor, t_cond: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Meta Flow Map (``api.dmfm_posterior``): ``n_steps`` student evaluations per sample."""
        if self.student is None:
            raise RuntimeError("dMFM posterior sampling needs a student checkpoint")
        out = api.dmfm_posterior(self.student, eps, x_cond, t_cond, sampler="flow_map", n_steps=int(n_steps))
        self.nfe.student += int(eps.shape[0]) * int(n_steps)
        return out

    def glass(
        self,
        eps: torch.Tensor,
        x_cond: torch.Tensor,
        t_cond: torch.Tensor,
        n_steps: int,
        *,
        end_time: float = 1.0,
        solver: str = "euler",
    ) -> torch.Tensor:
        """GLASS posterior ODE on the base model: ``n_steps`` base evaluations per sample."""
        out = api.glass_posterior(
            self.flow.model, eps, x_cond, t_cond, n_steps=int(n_steps), end_time=end_time, solver=solver
        )
        self.nfe.base += int(eps.shape[0]) * int(n_steps) * (4 if solver == "rk4" else 1)
        return out


def value_from_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """V_hat = logsumexp_i r_i - log N over the last dimension."""
    return torch.logsumexp(rewards, dim=-1) - math.log(rewards.shape[-1])


def dmfm_value_and_grad(
    sampler_post: PosteriorSampler,
    guide: C0Guide,
    reward: RewardSpec,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    n_mc: int,
    n_steps: int,
    nfe: NFECounter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact autodiff gradient of the finite-N log-mean-exp value (Eq. 3)."""
    B = x.shape[0]
    L, K = x.shape[1], x.shape[2]
    x_leaf = x.detach().clone().requires_grad_(True)
    eps = torch.randn(B * n_mc, L, K, device=x.device, dtype=x.dtype)
    x_rep = x_leaf.repeat_interleave(n_mc, 0)
    t_rep = t.detach().to(x.device, x.dtype).repeat_interleave(n_mc)
    with torch.enable_grad():
        x1 = sampler_post.dmfm(eps, x_rep, t_rep, n_steps)
        score = guide.model(x1)
        nfe.guide += int(x1.shape[0])
        r = (-0.5 * float(reward.scale) * ((score - reward.target) / reward.sigma).pow(2)).reshape(B, n_mc)
        V = value_from_rewards(r)
        grad = torch.autograd.grad(V.sum(), x_leaf)[0]
    nfe.backward += B
    return V.detach(), grad.detach()


@torch.no_grad()
def _step_base(flow: BaseFlow, x, t, dt, sampler):
    return flow.step(x, t, dt, sampler)[0]


def sample_dmfm_guided(
    flow: BaseFlow,
    sampler_post: PosteriorSampler,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    sampler: SamplerConfig,
    guidance: GuidanceConfig,
    batch_size: int = 25,
    nfe: NFECounter,
) -> dict:
    """The dMFM steering sweep of the original benchmark script, one MC setting."""
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
                _, g = dmfm_value_and_grad(
                    sampler_post, guide, reward, x, s,
                    n_mc=guidance.n_mc, n_steps=guidance.nfe_value, nfe=nfe,
                )
                gnorm = g.flatten(1).norm(dim=1).clamp_min(1e-8)
    # Scale-free gradient rule. ``grad_normalize`` rescales the gradient to EXACTLY
    # ``grad_clip`` instead of only clipping it down, so the step no longer depends on
    # ||grad V||. Every other DNA result in this paper uses it (Table 1, Figs 3/7/8,
    # Table 13, where it is worth +0.109 -> +0.124 paired); this benchmark was the one
    # place still clipping only downward, which under-doses whenever the estimate is
    # small -- exactly the low-MC regime this figure is about.
                if guidance.grad_clip is not None:
                    sc = float(guidance.grad_clip) / gnorm
                    g = g * (sc if guidance.grad_normalize else sc.clamp(max=1.0))[:, None, None]
                coeff = float(guidance.guidance_frac) * flow.model.sde_sigma_sq(s)
                coeff = coeff.clamp(max=float(guidance.coeff_cap))
                x = x_base + dt * coeff[:, None, None] * g
            else:
                x = x_base
        with torch.no_grad():
            r, s_hat = reward(guide, x.detach())
        scores.append(s_hat)
        tokens.append(x.detach().argmax(dim=-1))
        done += b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"score": torch.cat(scores), "tokens": torch.cat(tokens)}
