"""Best-of-N, Feynman-Kac steering, beam search and MCTS for the DNA flow.

These are the four baselines of Figure 4 (right), implemented as described in
Didi et al. (2026), *Scaling Atomistic Protein Binder Design with Generative
Pretraining and Test-Time Compute* (arXiv:2603.27950), Sec. 3 / App. H:

* **Best-of-N** -- draw N independent samples, return the best by the reward.
* **Beam search** -- maintain W parallel trajectories; every K denoising steps
  launch L new stochastic trajectories from each beam element, run each of the
  W*L candidates to completion, score them, and keep the top W. Didi et al.
  explicitly do *not* use Tweedie's formula for the candidate scores and instead
  roll out full trajectories, which is what is done here.
* **Feynman-Kac steering** -- the same branch-and-score loop, but the W survivors
  are *resampled* from the W*L candidates with probability proportional to
  exp(beta * R). Implemented in its cheap sequential-Monte-Carlo form: K
  particles are propagated once and reweighted by the look-ahead reward, which
  the sampler step already provides for free (the denoiser prediction
  E[X_1 | X_t]), so FK pays no generative NFE beyond plain sampling.
* **MCTS** -- the trajectory prefixes form a tree whose edges are K-step
  stochastic segments; a node is selected by UCT, expanded with one new child,
  and the child is rolled out to completion to give the backed-up value.

Branching requires a *stochastic* sampler: with the deterministic interpolant
sampler two children of one state are identical, resampling collapses the
population and beam search/MCTS/FK degenerate to best-of-N. All four therefore
run at ``SamplerConfig.eta > 0`` (Eq. 2 of the paper, which has the same
marginals as the ODE sampler); best-of-N is unaffected and is reported at both
eta = 0 and the same eta as the search methods.

Every method returns the *best sequence it scored* (not the last one it held), so
none of them is handicapped relative to best-of-N, and every reward evaluation is
counted in ``NFECounter.guide``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dmfm.benchmarks.core import BaseFlow, C0Guide, SamplerConfig, cyclizability_reward, harden


@dataclass
class RewardSpec:
    """r(x) = -0.5 * scale * ((f_cyc(x) - target) / sigma)^2 on hardened sequences."""

    target: float = 0.30
    sigma: float = 0.15
    scale: float = 1.0
    hard: bool = True

    def __call__(self, guide: C0Guide, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(reward, raw C0 score)``."""
        probe = harden(x, x.shape[-1]) if self.hard else x
        score = guide(probe)
        return cyclizability_reward(score, target=self.target, sigma=self.sigma, scale=self.scale), score


def _group_argmax(values: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Index (into the flat batch) of the best row of each group of equal size."""
    per = values.shape[0] // n_groups
    idx = values.view(n_groups, per).argmax(dim=1)
    return torch.arange(n_groups, device=values.device) * per + idx


def _integrate_ragged(
    flow: BaseFlow,
    x: torch.Tensor,
    start_step: torch.Tensor,
    sampler: SamplerConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate every row from its own starting grid index to ``n_steps``."""
    grid = sampler.grid(x.device, x.dtype)
    x = x.clone()
    first = int(start_step.min().item())
    for k in range(first, int(sampler.n_steps)):
        mask = start_step <= k
        if not bool(mask.any()):
            continue
        sub = x[mask]
        t = grid[k].expand(sub.shape[0])
        sub, _ = flow.step(sub, t, float(grid[k + 1] - grid[k]), sampler, generator=generator)
        x[mask] = sub
    return x


# --------------------------------------------------------------------------- best-of-N


@torch.no_grad()
def best_of_n(
    flow: BaseFlow,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    n_candidates: int,
    sampler: SamplerConfig,
    generator: torch.Generator,
    batch_rows: int = 1024,
) -> dict:
    """N independent trajectories per output; return the best-scoring endpoint."""
    n_candidates = int(n_candidates)
    best_score = torch.full((n_outputs,), float("nan"), device=flow.device)
    best_reward = torch.full((n_outputs,), -float("inf"), device=flow.device)
    best_tokens = torch.zeros((n_outputs, flow.L), dtype=torch.long, device=flow.device)

    rows_per_chunk = max(1, batch_rows // max(1, n_candidates)) * n_candidates
    total = n_outputs * n_candidates
    for start in range(0, total, rows_per_chunk):
        width = min(rows_per_chunk, total - start)
        x = flow.prior(width, generator=generator)
        x = flow.integrate(x, sampler, generator=generator)
        r, s = reward(guide, x)
        groups = width // n_candidates
        offset = start // n_candidates
        sel = _group_argmax(r, groups)
        cand_r, cand_s = r[sel], s[sel]
        cand_tok = x[sel].argmax(dim=-1)
        better = cand_r > best_reward[offset : offset + groups]
        idx = torch.arange(offset, offset + groups, device=flow.device)[better]
        best_reward[idx] = cand_r[better]
        best_score[idx] = cand_s[better]
        best_tokens[idx] = cand_tok[better]
    return {"score": best_score, "reward": best_reward, "tokens": best_tokens}


# --------------------------------------------------------------------------- Feynman-Kac


@torch.no_grad()
def feynman_kac(
    flow: BaseFlow,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    n_particles: int,
    resample_every: int,
    beta: float,
    sampler: SamplerConfig,
    generator: torch.Generator,
    weight: str = "delta",
    t_min: float = 0.01,
    t_max: float = 0.95,
    ess_threshold: float = 0.5,
) -> dict:
    """SMC steering: K particles reweighted by exp(beta * dR) on the look-ahead endpoint.

    The look-ahead reward uses the denoiser prediction ``psi_t(x) = E[X_1|X_t]``,
    which the sampler step returns from the forward pass it already made, so FK
    costs exactly the same generative NFE as drawing K plain samples.
    """
    K = int(n_particles)
    N = n_outputs * K
    x = flow.prior(N, generator=generator)
    grid = sampler.grid(x.device, x.dtype)
    prev_r = None
    for k in range(int(sampler.n_steps)):
        t = grid[k].expand(N)
        x, psi = flow.step(x, t, float(grid[k + 1] - grid[k]), sampler, generator=generator)
        do_resample = (
            K > 1
            and int(resample_every) > 0
            and (k + 1) % int(resample_every) == 0
            and (k + 1) < int(sampler.n_steps)
            and t_min <= float(grid[k]) <= t_max
        )
        if not do_resample:
            continue
        r, _ = reward(guide, psi)
        if weight == "delta" and prev_r is None:
            # No increment exists yet; weighting by the absolute reward here would set the
            # spread from the reward's scale rather than its change. Record and skip.
            logw = torch.zeros_like(r)
        else:
            logw = float(beta) * (r - prev_r if weight == "delta" else r)
        prev_r = r
        logw = logw.view(n_outputs, K)
        logw = logw - logw.max(dim=1, keepdim=True).values
        probs = torch.softmax(logw, dim=1)
        # Adaptive resampling; see the note in multimfm.baselines.sample_fk. Resampling with
        # replacement loses diversity even under uniform weights, so it is done only once the
        # effective sample size drops below a fraction of K. ess_threshold = 0 restores the
        # old unconditional behaviour that the stored results were produced with.
        ess = 1.0 / probs.square().sum(dim=1)
        need = ess < float(ess_threshold) * K
        pick = torch.multinomial(probs, num_samples=K, replacement=True, generator=generator)
        keep = torch.arange(K, device=pick.device).unsqueeze(0).expand_as(pick)
        pick = torch.where(need.unsqueeze(1), pick, keep)
        flat = (torch.arange(n_outputs, device=x.device).unsqueeze(1) * K + pick).reshape(-1)
        x = x[flat]
        prev_r = prev_r[flat]
    r, s = reward(guide, x)
    sel = _group_argmax(r, n_outputs)
    return {"score": s[sel], "reward": r[sel], "tokens": x[sel].argmax(dim=-1)}


# --------------------------------------------------------------------------- beam search


@torch.no_grad()
def beam_search(
    flow: BaseFlow,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    width: int,
    branch: int,
    checkpoint_every: int,
    sampler: SamplerConfig,
    generator: torch.Generator,
) -> dict:
    """W beams, L stochastic children per beam every K steps, scored by full rollout."""
    W, L, K = int(width), int(branch), int(checkpoint_every)
    n_steps = int(sampler.n_steps)
    if n_steps % K != 0:
        raise ValueError(f"checkpoint_every={K} must divide n_steps={n_steps}")
    M = n_steps // K

    beams = flow.prior(n_outputs * W, generator=generator)  # states at grid step 0
    best_reward = torch.full((n_outputs,), -float("inf"), device=flow.device)
    best_score = torch.full((n_outputs,), float("nan"), device=flow.device)
    best_tokens = torch.zeros((n_outputs, flow.L), dtype=torch.long, device=flow.device)

    for m in range(M):
        children = beams.repeat_interleave(L, dim=0)  # (n_outputs*W*L, ...)
        # advance K steps to grid step (m+1)*K
        grid = sampler.grid(children.device, children.dtype)
        for k in range(m * K, (m + 1) * K):
            t = grid[k].expand(children.shape[0])
            children, _ = flow.step(children, t, float(grid[k + 1] - grid[k]), sampler, generator=generator)
        # roll each candidate out to completion and score it
        remaining = n_steps - (m + 1) * K
        if remaining > 0:
            start = torch.full((children.shape[0],), (m + 1) * K, device=children.device, dtype=torch.long)
            rollouts = _integrate_ragged(flow, children, start, sampler, generator=generator)
        else:
            rollouts = children
        r, s = reward(guide, rollouts)
        sel = _group_argmax(r, n_outputs)
        better = r[sel] > best_reward
        idx = torch.arange(n_outputs, device=flow.device)[better]
        best_reward[idx] = r[sel][better]
        best_score[idx] = s[sel][better]
        best_tokens[idx] = rollouts[sel].argmax(dim=-1)[better]
        if m == M - 1:
            break
        keep = r.view(n_outputs, W * L).topk(W, dim=1).indices
        flat = (torch.arange(n_outputs, device=flow.device).unsqueeze(1) * (W * L) + keep).reshape(-1)
        beams = children[flat]
    return {"score": best_score, "reward": best_reward, "tokens": best_tokens}


# --------------------------------------------------------------------------- MCTS


@torch.no_grad()
def mcts(
    flow: BaseFlow,
    guide: C0Guide,
    reward: RewardSpec,
    *,
    n_outputs: int,
    iterations: int,
    children: int,
    checkpoint_every: int,
    c_uct: float,
    sampler: SamplerConfig,
    generator: torch.Generator,
) -> dict:
    """UCT over trajectory prefixes; every expansion is scored by a full rollout.

    One independent tree per output, all trees advanced in lockstep so the K-step
    expansions and the rollouts of an iteration are one batched forward pass.
    Backed-up values are ``exp(r)`` in (0, 1], which puts them on the same scale as
    the UCT exploration term.
    """
    K = int(checkpoint_every)
    n_steps = int(sampler.n_steps)
    if n_steps % K != 0:
        raise ValueError(f"checkpoint_every={K} must divide n_steps={n_steps}")
    M = n_steps // K
    n_children = int(children)

    # Per output: a list of nodes. node = dict(state, depth, parent, kids, visits, value_sum)
    trees: list[list[dict]] = []
    roots = flow.prior(n_outputs, generator=generator)
    for j in range(n_outputs):
        trees.append([
            {"state": roots[j], "depth": 0, "parent": -1, "kids": [], "visits": 0, "value_sum": 0.0}
        ])

    best_reward = torch.full((n_outputs,), -float("inf"), device=flow.device)
    best_score = torch.full((n_outputs,), float("nan"), device=flow.device)
    best_tokens = torch.zeros((n_outputs, flow.L), dtype=torch.long, device=flow.device)

    for _ in range(int(iterations)):
        chosen: list[int] = []
        for j in range(n_outputs):
            nodes = trees[j]
            node = 0
            while nodes[node]["depth"] < M and len(nodes[node]["kids"]) >= n_children:
                parent_visits = max(1, nodes[node]["visits"])
                best_u, best_k = -float("inf"), nodes[node]["kids"][0]
                for kid in nodes[node]["kids"]:
                    v = nodes[kid]["visits"]
                    q = nodes[kid]["value_sum"] / v if v > 0 else 0.0
                    u = q + float(c_uct) * math.sqrt(math.log(parent_visits + 1.0) / max(1, v))
                    if u > best_u:
                        best_u, best_k = u, kid
                node = best_k
            chosen.append(node)

        parent_states = torch.stack([trees[j][chosen[j]]["state"] for j in range(n_outputs)])
        depths = torch.tensor([trees[j][chosen[j]]["depth"] for j in range(n_outputs)], device=flow.device)
        # expand: one new stochastic child, K steps (or 0 steps if already terminal)
        start = (depths * K).clamp(max=n_steps)
        child_end = ((depths + 1) * K).clamp(max=n_steps)
        states = parent_states.clone()
        grid = sampler.grid(states.device, states.dtype)
        for k in range(int(start.min().item()), int(child_end.max().item())):
            mask = (start <= k) & (k < child_end)
            if not bool(mask.any()):
                continue
            sub = states[mask]
            t = grid[k].expand(sub.shape[0])
            sub, _ = flow.step(sub, t, float(grid[k + 1] - grid[k]), sampler, generator=generator)
            states[mask] = sub
        child_states = states.clone()
        # rollout to completion
        rollouts = _integrate_ragged(flow, states, child_end, sampler, generator=generator)
        r, s = reward(guide, rollouts)
        value = r.exp()

        better = r > best_reward
        best_reward = torch.where(better, r, best_reward)
        best_score = torch.where(better, s, best_score)
        best_tokens = torch.where(better.unsqueeze(1), rollouts.argmax(dim=-1), best_tokens)

        for j in range(n_outputs):
            nodes = trees[j]
            parent = chosen[j]
            if nodes[parent]["depth"] < M:
                nodes.append(
                    {
                        "state": child_states[j],
                        "depth": nodes[parent]["depth"] + 1,
                        "parent": parent,
                        "kids": [],
                        "visits": 0,
                        "value_sum": 0.0,
                    }
                )
                nodes[parent]["kids"].append(len(nodes) - 1)
                leaf = len(nodes) - 1
            else:
                leaf = parent
            v = float(value[j])
            node = leaf
            while node != -1:
                nodes[node]["visits"] += 1
                nodes[node]["value_sum"] += v
                node = nodes[node]["parent"]
    return {"score": best_score, "reward": best_reward, "tokens": best_tokens}
