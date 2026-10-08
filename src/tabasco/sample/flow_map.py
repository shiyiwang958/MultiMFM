"""Few-step consistency posterior sampler for the TABASCO MFM student.

Mirrors MFM's ``consistency_sampler_fn`` but in TABASCO TensorDict space: starting
from the Gaussian prior, step the inner time ``0 -> 1`` in ``n_steps`` flow-map
jumps, each ``x <- x + (u - s) * v(s, u, x, t_cond, x_cond)``, keeping coords
zero-COM and atomics masked. Produces a posterior sample of ``p(x1 | x_cond, t_cond)``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


def _prior_like(cond_state: TensorDict) -> TensorDict:
    coords = torch.randn_like(cond_state["coords"])
    atomics = torch.randn_like(cond_state["atomics"])
    mask = cond_state["padding_mask"]
    return TensorDict(
        {
            "coords": mask_and_zero_com(coords, mask),
            "atomics": apply_mask(atomics, mask),
            "padding_mask": mask,
        },
        batch_size=cond_state.batch_size,
    )


@torch.no_grad()
def consistency_posterior_sample(
    student,
    cond_state: TensorDict,
    t_cond: Tensor,
    *,
    n_steps: int = 4,
    noise: TensorDict | None = None,
    diagonal: bool = False,
) -> TensorDict:
    """Draw a posterior sample with an ``n_steps`` consistency rollout.

    Args:
        student: ``TabascoFlowMap``.
        cond_state: conditioning observation ``Y_{t_cond}`` ``{coords, atomics, mask}``.
        t_cond: conditioning level, shape ``(B,)``.
        n_steps: number of inner-time steps over ``[0, 1]``.
        noise: optional starting prior sample (else standard Gaussian, zero-COM).
        diagonal: if True, query the *instantaneous* velocity ``v(s, s, .)`` and
            Euler-step by ``ds`` (the Stage-1 / diagonal-only capability). If False,
            use flow-map jumps ``v(s, u, .)`` (needs the Stage-2 off-diagonal training).
    """
    mask = cond_state["padding_mask"]
    state = noise if noise is not None else _prior_like(cond_state)
    flow_t = torch.linspace(0.0, 1.0, n_steps + 1, device=t_cond.device)

    for i in range(n_steps):
        s = torch.full_like(t_cond, float(flow_t[i]))
        u = torch.full_like(t_cond, float(flow_t[i + 1]))
        if diagonal:
            u = s  # query instantaneous velocity; still advance state by ds below
        coords_v, atomics_v = student(
            state["coords"],
            state["atomics"],
            cond_state["coords"],
            cond_state["atomics"],
            mask,
            s,
            u,
            t_cond,
        )
        step = float(flow_t[i + 1] - flow_t[i])  # advance by dt even when u==s
        ds = torch.full_like(s, step).view(-1, *([1] * (state["coords"].ndim - 1)))
        state["coords"] = mask_and_zero_com(state["coords"] + ds * coords_v, mask)
        state["atomics"] = apply_mask(state["atomics"] + ds * atomics_v, mask)

    return state
