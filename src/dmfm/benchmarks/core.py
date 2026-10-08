"""Shared machinery for the Figure 4 benchmarks: models, sampler, reward, NFE counting.

Every method in Figure 4 -- the four search baselines and the dMFM -- uses the
*same* base flow, the *same* C0 guide, the *same* target and the *same* terminal
scoring, so the only thing that differs between the curves is how the sampling
budget is spent.

Sampler
-------
The base generator is the parent-disjoint L=50 Gaussian-interpolant DFM DiT.
:meth:`BaseFlow.step` is one Euler-Maruyama step of

    dX = [ b_t(X) + (sigma_eff^2 / 2) * score_t(X) ] dt + sigma_eff dB,
    sigma_eff^2(t) = min(eta^2 * sigma_t^2, cap),   sigma_t^2 = 2(1-t)/t,

which is equation (2) of the paper with V = 0 and a stochasticity level ``eta``.
The Fokker-Planck identity holds for any (time-dependent) sigma_eff, so every
``eta`` and every cap leaves the marginals p_t unchanged in the continuum limit.

* ``eta = 0`` is *exactly* the deterministic sampler used everywhere else in the
  DNA code: for the linear schedule beta(t) = t, an Euler step of the
  probability-flow ODE equals the closed-form interpolant update
  ``x_t = Gamma_st x_s + Delta_st psi`` of
  :func:`dmfm.utils.flow_utils.gaussian_denoiser_flow_step` (checked by
  ``scripts/dna/fig4_selftest.py``). This is the sampler for best-of-N and for
  the dMFM curve.
* ``eta > 0`` makes trajectories branch: two children of one state differ by their
  Brownian increments. Feynman-Kac steering, beam search and MCTS all need this,
  because with a deterministic sampler a resampled/duplicated particle would
  evolve identically to its parent and the population would collapse.

One step costs exactly one forward pass of the base network (unlike
``gaussian_denoiser_flow_step``, which evaluates the network twice: once with
``return_logits=True`` and once for the denoiser). The denoiser prediction
``psi_t(x) = E[X_1 | X_t = x]`` comes out of the same pass, so a look-ahead
potential built from it is free -- which is the best case for Feynman-Kac.

NFE accounting
--------------
:class:`NFECounter` keeps the networks apart because they have very different
sizes. ``nfe_gen`` (base + student forward passes) is the quantity on the x-axis
of Figure 4 right, matching the original script's ``gen_nfe_per_output``. Nothing
here assumes parallel execution, and NFEs are not wall-clock: the dMFM
additionally needs ``nfe_backward`` backward passes, which the counter reports
separately.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel

DNA_ALPHABET = "ACGT"


# --------------------------------------------------------------------------- NFE


@dataclass
class NFECounter:
    """Forward/backward passes, kept separate per network.

    ``base``      forward passes of the base DFM DiT (33 M params)
    ``student``   forward passes of the dMFM student (ditto, one per posterior sample)
    ``guide``     forward passes of the C0 CNN (0.09 M params)
    ``backward``  backward passes through student+guide (``torch.autograd.grad``)
    """

    base: int = 0
    student: int = 0
    guide: int = 0
    backward: int = 0

    def __iadd__(self, other: "NFECounter") -> "NFECounter":
        self.base += other.base
        self.student += other.student
        self.guide += other.guide
        self.backward += other.backward
        return self

    @property
    def gen(self) -> int:
        """Generative-model NFE: the x-axis of Fig 4 (right)."""
        return self.base + self.student

    def per_output(self, n_outputs: int) -> dict[str, float]:
        n = max(1, int(n_outputs))
        return {
            "nfe_gen_per_output": self.gen / n,
            "nfe_base_per_output": self.base / n,
            "nfe_student_per_output": self.student / n,
            "nfe_guide_per_output": self.guide / n,
            "nfe_backward_per_output": self.backward / n,
        }


# --------------------------------------------------------------------------- models


def _as_cfg(raw) -> SimpleNamespace:
    cfg = raw if isinstance(raw, SimpleNamespace) else SimpleNamespace(**raw)
    cfg.use_flash_attn = bool(getattr(cfg, "use_flash_attn", False))
    cfg.flow_temp = float(getattr(cfg, "flow_temp", 1.0))
    cfg.gaussian_beta_schedule = getattr(cfg, "gaussian_beta_schedule", "linear")
    cfg.gaussian_beta_table_path = getattr(cfg, "gaussian_beta_table_path", None)
    return cfg


@dataclass
class SamplerConfig:
    """Everything that defines a trajectory of the base flow."""

    n_steps: int = 96
    t_max: float = 1.0
    eta: float = 0.0
    sigma_sq_cap: float = 10.0
    flow_temp: float | None = None

    def grid(self, device, dtype=torch.float32) -> torch.Tensor:
        return torch.linspace(0.0, float(self.t_max), int(self.n_steps) + 1, device=device, dtype=dtype)


class BaseFlow:
    """The parent-disjoint base DFM, with a single-forward-pass sampler step."""

    def __init__(self, model: DiTSequenceModel, cfg: SimpleNamespace, nfe: NFECounter | None = None):
        self.model = model
        self.cfg = cfg
        self.L = int(cfg.seq_len)
        self.K = int(cfg.alphabet_size)
        self.device = next(model.parameters()).device
        self.nfe = nfe if nfe is not None else NFECounter()

    # -- basic quantities -------------------------------------------------

    def denoiser(self, x: torch.Tensor, t: torch.Tensor, *, flow_temp: float | None = None) -> torch.Tensor:
        """psi_t(x) = E[X_1 | X_t = x] as a simplex point. Costs one base NFE per row."""
        temp = float(self.cfg.flow_temp if flow_temp is None else flow_temp)
        logits = self.model.forward_logits(x, t, t)
        self.nfe.base += int(x.shape[0])
        return F.log_softmax(logits / temp, dim=-1).exp()

    def sigma_sq(self, t: torch.Tensor, sampler: SamplerConfig) -> torch.Tensor:
        """Effective diffusion coefficient min(eta^2 sigma_t^2, cap); 0 at t = 0."""
        eta = float(sampler.eta)
        if eta <= 0.0:
            return torch.zeros_like(t)
        raw = self.model.sde_sigma_sq(t.clamp_min(1e-6))
        out = (eta * eta) * raw
        out = out.clamp(max=float(sampler.sigma_sq_cap))
        return torch.where(t > 0, out, torch.zeros_like(out))

    # -- one step ---------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        sampler: SamplerConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Euler(-Maruyama) step. Returns ``(x_next, psi)``; ``psi`` is free."""
        psi = self.denoiser(x, t, flow_temp=sampler.flow_temp)
        drift = self.model.b_t(x, psi, t)
        sig_sq = self.sigma_sq(t, sampler)
        if float(sig_sq.max()) > 0.0:
            score = self.model.score(x, t.clamp_min(1e-6), u=drift)
            drift = drift + 0.5 * _bcast(sig_sq, x) * score
        x_next = x + dt * drift
        if float(sig_sq.max()) > 0.0:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x_next = x_next + _bcast((sig_sq * dt).clamp_min(0.0).sqrt(), x) * noise
        return x_next, psi

    @torch.no_grad()
    def integrate(
        self,
        x: torch.Tensor,
        sampler: SamplerConfig,
        *,
        k_start: int = 0,
        k_end: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run grid steps ``k_start .. k_end-1`` (``k_end`` defaults to ``n_steps``)."""
        k_end = int(sampler.n_steps if k_end is None else k_end)
        grid = sampler.grid(x.device, x.dtype)
        for k in range(int(k_start), k_end):
            t = grid[k].expand(x.shape[0])
            x, _ = self.step(x, t, float(grid[k + 1] - grid[k]), sampler, generator=generator)
        return x

    def prior(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(n, self.L, self.K, device=self.device, generator=generator)


def _bcast(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while c.ndim < x.ndim:
        c = c[..., None]
    return c.to(device=x.device, dtype=x.dtype)


def load_base_flow(
    length: int = 50,
    device: torch.device | None = None,
    nfe: NFECounter | None = None,
    *,
    ckpt: str | Path | None = None,
) -> BaseFlow:
    """``dmfm.api.load_base`` wrapped in the NFE-counting sampler of this package."""
    from dmfm import api

    model, cfg = api.load_base(length, device, ckpt=ckpt)
    return BaseFlow(model, _as_cfg(cfg), nfe=nfe)


def load_dmfm_student(length: int = 50, device=None, *, ckpt: str | Path | None = None):
    """``dmfm.api.load_dmfm`` (diagonal + ESD student), frozen and in eval mode."""
    from dmfm import api

    student = api.load_dmfm(length, device, ckpt=ckpt)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


# --------------------------------------------------------------------------- guide


class C0Guide:
    """The Park-CNN cyclizability predictor (``dmfm.api.load_c0``), with NFE accounting.

    ``guide`` (trained on the a-parents, the same split as the base flow and the
    dMFM) drives every method and defines the reported target error, exactly as in
    the original benchmark script. ``oracle`` (b-parents, disjoint) is carried
    along as an independent check, reverse-complement averaged as in Table 1.
    """

    def __init__(self, model: torch.nn.Module, nfe: NFECounter, *, rc_average: bool = False):
        self.model = model
        self.nfe = nfe
        self.rc_average = bool(rc_average)

    def __call__(self, x_acgt: torch.Tensor) -> torch.Tensor:
        out = self.model(x_acgt)
        self.nfe.guide += int(x_acgt.shape[0])
        if self.rc_average:
            out = 0.5 * (out + self.model(x_acgt.flip(dims=(1,))[..., [3, 2, 1, 0]]))
            self.nfe.guide += int(x_acgt.shape[0])
        return out


def load_c0(role: str, device, nfe: NFECounter, *, rc_average: bool = False, ckpt=None) -> C0Guide:
    """``dmfm.api.load_c0`` wrapped so its evaluations land in ``nfe.guide``."""
    from dmfm import api

    model = api.load_c0(role, device, ckpt=ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    return C0Guide(model, nfe, rc_average=rc_average)


# --------------------------------------------------------------------------- reward


def cyclizability_reward(score: torch.Tensor, *, target: float, sigma: float, scale: float = 1.0) -> torch.Tensor:
    """``r(x) = -0.5 lambda ((f_cyc(x) - y*) / sigma)^2``.

    The paper writes the reward as ``r(x) = -(f_cyc(x) - y*)^2``; the DNA code has
    always carried the ``1/(2 sigma^2)`` scale (``reward_sigma = 0.15``), which is
    what puts ``exp(r)`` on a useful scale for the log-mean-exp value function.
    Only ``sigma`` and ``scale`` set the temperature; the argmax over candidates --
    and therefore best-of-N and beam search -- is unchanged by them.
    """
    return -0.5 * float(scale) * ((score - float(target)) / float(sigma)).pow(2)


def harden(x: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    """argmax -> one-hot: the sequence that is actually returned to the user."""
    return F.one_hot(x.argmax(dim=-1), alphabet_size).to(x.dtype)


def tokens_to_strings(tokens: torch.Tensor) -> list[str]:
    return ["".join(DNA_ALPHABET[i] for i in row) for row in tokens.tolist()]


# --------------------------------------------------------------------------- io


@dataclass
class ShardResult:
    """One (method, config, repeat) cell of Fig 4 right."""

    method: str
    config: str
    config_params: dict = field(default_factory=dict)
    repeat: int = 0
    seed: int = 0
    n_outputs: int = 0
    metrics: dict = field(default_factory=dict)
    nfe: dict = field(default_factory=dict)
    per_sample: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=float) + "\n")
        tmp.replace(path)  # atomic: a reader never sees a partial shard
        return path


def summarize_errors(scores, target: float) -> dict[str, float]:
    """The summary the original benchmark script reported, per repeat."""
    import numpy as np

    d = np.abs(np.asarray(scores, dtype=np.float64) - float(target))
    return {
        "mae": float(d.mean()),
        "median_ae": float(np.median(d)),
        "frac05": float((d <= 0.05).mean()),
        "frac10": float((d <= 0.10).mean()),
        "mean_score": float(np.asarray(scores, dtype=np.float64).mean()),
        "std_score": float(np.asarray(scores, dtype=np.float64).std(ddof=1)) if len(d) > 1 else 0.0,
    }


def default_paths(length: int = 50) -> dict[str, str]:
    return {
        "base_ckpt": str(paths.base_ckpt(length)),
        "dmfm_ckpt": str(paths.dmfm_ckpt(length)),
        "dmfm_args": str(paths.dmfm_args(length)),
        "c0_guide": str(paths.c0_guide()),
        "c0_oracle": str(paths.c0_oracle()),
    }


def make_generator(device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def log_mean_exp(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])
