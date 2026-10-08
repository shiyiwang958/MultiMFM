#!/usr/bin/env python
"""Figure 4 (left): value-function estimation error vs the number of posterior samples N.

    V_t(x) = log E[e^{r(X_1)} | X_t = x],
    V_hat_t(x; N) = logsumexp_{i<=N} r(x_1^{(i)}) - log N.

**This is a port of the original producer**, recovered on 2026-09-29 from
``tullebulle/DNA-MFM@187fe7b`` and kept verbatim at
``results/dna/original_scripts/benchmark_yeast_split_value_mc.py``, run on the
parent-disjoint L=50 models instead of the purged ``split65k`` ones. Its protocol
(all defaults of the original, which are what its Slurm launcher used):

* **Reward** -- *not* the cyclizability guide. A synthetic smooth reward
  ``SmoothDNAReward(L, 4, seed=1234, gain=4.0)``: a random linear term, a random
  nearest-neighbour pair term, a fixed 8-base motif bonus, a GC penalty and an
  entropy bonus, squashed and scaled by ``gain``. Reproduced here exactly
  (the reward's own seed is 1234, independent of ``--seed``).
* **Conditioning states** -- held-out *test* sequences, each noised forward to its own
  time ``t ~ U(0.05, 0.35)``: ``x_t = beta_t x_1 + (1 - beta_t) eps``. ``n_conditions = 8``
  is the shared default of the script and its Slurm launcher; no log of the plotted run
  survives, so **8 is an assumption, not evidence**.
* **Reference V_t** -- teacher **SDE** sampling: ``--sde_ref_n 512`` samples,
  ``--sde_steps 1024`` Euler-Maruyama steps from ``t`` to 1. It is itself a Monte
  Carlo estimate; the ``sde_mc`` series below measures its own finite-N error, and
  the published panel omits that series.
* **Estimators** -- ``dmfm`` (``--dmfm_nfe 1``: one flow-map evaluation per sample),
  ``dps`` (the *student's* denoiser ``Psi_st(x_t, t, t | t, x_t)``, N-independent),
  ``fmap`` (teacher ODE integration from ``t`` to 1 with ``--fmap_nfe 256``, the
  paper's "best case ... integrating along its flow matching b_t(x) trajectory";
  also N-independent), and ``sde_mc``.
* N: the script's default ``--n_list`` stops at 256, but the launcher takes ``N_LIST``
  from the environment and the published panel has exactly the 10 points N = 1..512, so
  the plotted run overrode it. The default here is 1..512.
* ``dps`` and ``fmap`` are computed **once per condition**, outside the N loop -- that is
  why they are exactly flat in the published panel. Every other draw is seeded
  ``seed + 10000*cond_idx + {1, 100+n, 200+n}`` for the reference, the ``sde_mc`` floor and
  the dMFM, so the budgets are independent draws, not nested prefixes.
* Sampling is chunked only to bound memory; at the default ``--chunk 512`` every draw
  (at most 512 samples) is a single chunk, so the seeding is the original's exactly.

``--n_conditions`` is raised above 8 only to add error bars; conditions are drawn
in the original's order from one seeded stream, so conditions 0..7 are the
original protocol and any extra ones are additional draws.

Robustness variants kept behind flags (not the submitted experiment):
``--reward c0`` (the cyclizability reward of Fig 4 right) and ``--reference glass``
(the GLASS-2048 posterior reference of Fig 5 / Tables 14-15).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dmfm import api, paths
from dmfm.benchmarks.core import NFECounter, load_base_flow, load_c0, load_dmfm_student
from dmfm.utils.flow_utils import gaussian_beta, gaussian_denoiser_flow_step


# --------------------------------------------------------------------------- reward


class SmoothDNAReward:
    """Verbatim from the original producer (``benchmark_yeast_split_value_mc.py``)."""

    def __init__(self, length: int, alphabet_size: int, device: torch.device, *, seed: int, gain: float):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.W = torch.randn((length, alphabet_size), device=device, generator=gen) / math.sqrt(length)
        self.P = torch.randn(
            (length - 1, alphabet_size, alphabet_size), device=device, generator=gen
        ) / math.sqrt(length - 1)
        self.gain = float(gain)
        self.motif_pos = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], device=device)
        self.motif_base = torch.tensor([0, 3, 0, 3, 0, 0, 2, 1], device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lin = (x * self.W).sum(dim=(1, 2))
        pair = (x[:, :-1, :, None] * x[:, 1:, None, :] * self.P[None, :, :, :]).sum(dim=(1, 2, 3))
        motif = x[:, self.motif_pos, self.motif_base].mean(dim=1)
        gc = (x[:, :, 1] + x[:, :, 2]).mean(dim=1)
        entropy = -(x.clamp_min(1e-12) * x.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=1)
        r = (
            torch.tanh(lin)
            + 0.7 * torch.tanh(pair)
            + 1.5 * torch.sigmoid(8.0 * (motif - 0.35))
            - 2.0 * (gc - 0.52).pow(2)
            + 0.15 * entropy
        )
        return self.gain * r


class C0Reward:
    """Robustness variant: the cyclizability reward of Fig 4 (right)."""

    def __init__(self, guide, *, target: float, sigma: float, scale: float):
        self.guide, self.target, self.sigma, self.scale = guide, target, sigma, scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.scale * ((self.guide(x) - self.target) / self.sigma).pow(2)


def value_stats(r: torch.Tensor) -> dict[str, float]:
    """Verbatim from the original producer."""
    V = torch.logsumexp(r, dim=0) - math.log(r.numel())
    w = torch.softmax(r, dim=0)
    ess = 1.0 / w.pow(2).sum()
    return {
        "V": float(V.detach().cpu()),
        "r_mean": float(r.mean().detach().cpu()),
        "r_std": float(r.std().detach().cpu()),
        "r_min": float(r.min().detach().cpu()),
        "r_max": float(r.max().detach().cpu()),
        "ess": float(ess.detach().cpu()),
        "ess_frac": float((ess / r.numel()).detach().cpu()),
    }


# --------------------------------------------------------------------------- protocol


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--out", required=True, help="output .json for this shard")
    p.add_argument("--cond-start", type=int, default=0)
    p.add_argument("--cond-end", type=int, default=8)
    # --- the original's defaults ---
    p.add_argument("--n-conditions", type=int, default=8)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--sde-ref-n", type=int, default=512)
    p.add_argument("--sde-steps", type=int, default=1024)
    p.add_argument("--dmfm-nfe", type=int, default=1)
    p.add_argument("--fmap-nfe", type=int, default=256)
    p.add_argument("--reward-gain", type=float, default=4.0)
    p.add_argument("--reward-seed", type=int, default=1234)
    p.add_argument("--tcond-low", type=float, default=0.05)
    p.add_argument("--tcond-high", type=float, default=0.35)
    p.add_argument("--data-split", default="test", choices=["a_train", "a_val", "b_train", "b_val", "test", "full"])
    p.add_argument("--seed", type=int, default=12345)
    # --- robustness variants (not the submitted experiment) ---
    p.add_argument("--reward", choices=["smooth", "c0"], default="smooth")
    p.add_argument("--reference", choices=["sde", "glass"], default="sde")
    p.add_argument("--glass-ref-n", type=int, default=2048)
    p.add_argument("--glass-nfe", type=int, default=8)
    p.add_argument("--target-c0", type=float, default=0.30)
    p.add_argument("--reward-sigma", type=float, default=0.15)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def make_conditions(cfg, seqs, K, L, device, n: int, *, t_low: float, t_high: float) -> list[dict]:
    """The original ``make_condition`` loop, drawn in order from one seeded stream."""
    out = []
    for cond_idx in range(n):
        seq_idx = torch.randint(len(seqs), (1,)).item()
        seq = seqs[seq_idx : seq_idx + 1].to(device)
        x1 = F.one_hot(seq, num_classes=K).float()
        t_val = t_low + (t_high - t_low) * torch.rand((), device=device).item()
        t_cond_1 = torch.full((1,), t_val, device=device)
        beta_t = gaussian_beta(cfg, t_cond_1)
        noise = torch.randn((1, L, K), device=device)
        x_t_1 = beta_t[:, None, None] * x1 + (1.0 - beta_t[:, None, None]) * noise
        out.append(
            {"cond_idx": cond_idx, "seq_idx": int(seq_idx), "t_cond_value": float(t_val),
             "x_t_1": x_t_1, "t_cond_1": t_cond_1}
        )
    return out


@torch.no_grad()
def sample_teacher_sde_n(teacher, condition, n, *, sde_steps, seed, device, nfe, chunk):
    """Original ``sample_teacher_sde_N``, chunked (chunking changes only memory use)."""
    outs = []
    for start in range(0, int(n), int(chunk)):
        w = min(int(chunk), int(n) - start)
        x_t = condition["x_t_1"].expand(w, -1, -1).contiguous()
        gen = torch.Generator(device=device).manual_seed(int(seed) + start)
        outs.append(
            teacher.sample_sde(
                x_t.clone(), t_start=float(condition["t_cond_value"]), t_end=1.0,
                n_steps=int(sde_steps), generator=gen,
            ).detach()
        )
        nfe.base += w * int(sde_steps)
    return torch.cat(outs)


@torch.no_grad()
def sample_dmfm_n(student, condition, n, *, K, L, dmfm_nfe, seed, device, nfe, chunk):
    """Original ``sample_dmfm_N``: compose ``dmfm_nfe`` maps on linspace(0, 1)."""
    outs = []
    for start in range(0, int(n), int(chunk)):
        w = min(int(chunk), int(n) - start)
        gen = torch.Generator(device=device).manual_seed(int(seed) + start)
        x = torch.randn((w, L, K), device=device, generator=gen)
        x_t = condition["x_t_1"].expand(w, -1, -1).contiguous()
        t_cond = condition["t_cond_1"].expand(w).contiguous()
        for k in range(int(dmfm_nfe)):
            r0 = torch.full((w,), k / int(dmfm_nfe), device=device, dtype=x.dtype)
            r1 = torch.full((w,), (k + 1) / int(dmfm_nfe), device=device, dtype=x.dtype)
            x = student(r0, r1, x, t_cond, x_t).detach()
        nfe.student += w * int(dmfm_nfe)
        outs.append(x)
    return torch.cat(outs)


@torch.no_grad()
def sample_glass_n(base, condition, n, *, K, L, glass_nfe, seed, device, nfe, chunk):
    """Robustness variant: GLASS posterior samples of the base model."""
    outs = []
    for start in range(0, int(n), int(chunk)):
        w = min(int(chunk), int(n) - start)
        gen = torch.Generator(device=device).manual_seed(int(seed) + start)
        eps = torch.randn((w, L, K), device=device, generator=gen)
        x_cond = condition["x_t_1"].expand(w, -1, -1).contiguous()
        t_cond = condition["t_cond_1"].expand(w).contiguous()
        outs.append(api.glass_posterior(base, eps, x_cond, t_cond, n_steps=int(glass_nfe)).detach())
        nfe.base += w * int(glass_nfe)
    return torch.cat(outs)


@torch.no_grad()
def dps_endpoint(student, condition, nfe: NFECounter):
    """Original ``dps_value_estimate``: the *student's* denoiser at the conditioning state."""
    x_t_1, t_1 = condition["x_t_1"], condition["t_cond_1"]
    nfe.student += int(x_t_1.shape[0])
    return student.Psi_st(x_t_1, t_1, t_1, t_cond=t_1, x_cond=x_t_1)


@torch.no_grad()
def fmap_endpoint_teacher(teacher, cfg, condition, *, fmap_nfe, device, nfe: NFECounter):
    """Original ``fmap_endpoint_teacher``: teacher ODE from t_cond to 1 in ``fmap_nfe`` steps."""
    x = condition["x_t_1"].clone()
    grid = torch.linspace(float(condition["t_cond_value"]), 1.0, int(fmap_nfe) + 1, device=device, dtype=x.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        b = x.shape[0]
        x, _, _ = gaussian_denoiser_flow_step(cfg, teacher, x, s0.expand(b), s1.expand(b))
        nfe.base += b
    return x.detach()


def main(argv=None) -> None:
    cli = parse_args(argv)
    out_path = Path(cli.out)
    if out_path.exists():
        print(f"shard exists, skipping: {out_path}")
        return
    device = torch.device(cli.device) if cli.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cli.seed)  # the original seeds the global stream and draws conditions from it
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    nfe = NFECounter()
    flow = load_base_flow(cli.length, device, nfe)
    teacher, cfg = flow.model, flow.cfg
    student = load_dmfm_student(cli.length, device)
    K, L = int(cfg.alphabet_size), int(cfg.seq_len)

    if cli.reward == "smooth":
        reward_fn = SmoothDNAReward(L, K, device, seed=cli.reward_seed, gain=cli.reward_gain)
    else:
        reward_fn = C0Reward(
            load_c0("guide", device, nfe, rc_average=False),
            target=cli.target_c0, sigma=cli.reward_sigma, scale=cli.reward_scale,
        )

    seqs = api.load_split_seqs(cli.length, cli.data_split).long()
    conditions = make_conditions(
        cfg, seqs, K, L, device, cli.n_conditions, t_low=cli.tcond_low, t_high=cli.tcond_high
    )

    rows: list[dict] = []
    t0 = time.time()
    for cond in conditions[cli.cond_start : cli.cond_end]:
        ci = cond["cond_idx"]
        print(f"condition {ci:03d}: seq={cond['seq_idx']} t={cond['t_cond_value']:.3f}", flush=True)

        if cli.reference == "sde":
            x_ref = sample_teacher_sde_n(
                teacher, cond, cli.sde_ref_n, sde_steps=cli.sde_steps,
                seed=cli.seed + 10000 * ci + 1, device=device, nfe=nfe, chunk=cli.chunk,
            )
            ref_n = cli.sde_ref_n
        else:
            x_ref = sample_glass_n(
                teacher, cond, cli.glass_ref_n, K=K, L=L, glass_nfe=cli.glass_nfe,
                seed=cli.seed + 10000 * ci + 1, device=device, nfe=nfe, chunk=cli.chunk,
            )
            ref_n = cli.glass_ref_n
        r_ref = reward_fn(x_ref)
        ref = value_stats(r_ref)
        # the reference's own MC spread: two disjoint halves of its sample set
        half = r_ref.numel() // 2
        ref["selfsplit_abs_diff"] = abs(value_stats(r_ref[:half])["V"] - value_stats(r_ref[half:])["V"])

        dps = value_stats(reward_fn(dps_endpoint(student, cond, nfe)))
        fmap = value_stats(
            reward_fn(fmap_endpoint_teacher(teacher, cfg, cond, fmap_nfe=cli.fmap_nfe, device=device, nfe=nfe))
        )

        for n in cli.n_list:
            ests = {}
            ests["sde_mc"] = value_stats(
                reward_fn(
                    sample_teacher_sde_n(
                        teacher, cond, n, sde_steps=cli.sde_steps,
                        seed=cli.seed + 10000 * ci + 100 + n, device=device, nfe=nfe, chunk=cli.chunk,
                    )
                )
            )
            ests["dmfm"] = value_stats(
                reward_fn(
                    sample_dmfm_n(
                        student, cond, n, K=K, L=L, dmfm_nfe=cli.dmfm_nfe,
                        seed=cli.seed + 10000 * ci + 200 + n, device=device, nfe=nfe, chunk=cli.chunk,
                    )
                )
            )
            ests["dps"], ests["fmap"] = dps, fmap
            for method, est in ests.items():
                rows.append(
                    {
                        "cond_idx": ci, "seq_idx": cond["seq_idx"], "t_cond": cond["t_cond_value"],
                        "method": method, "n_samples": int(n),
                        "V_ref": ref["V"], "V_est": est["V"],
                        "V_err": est["V"] - ref["V"], "abs_V_err": abs(est["V"] - ref["V"]),
                        "ref_selfsplit_abs_diff": ref["selfsplit_abs_diff"],
                        **{f"ref_{k}": v for k, v in ref.items() if k != "selfsplit_abs_diff"},
                        **{f"est_{k}": v for k, v in est.items()},
                    }
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"  V_ref={ref['V']:.4f} (ref n={ref_n}, half-vs-half {ref['selfsplit_abs_diff']:.4f}) "
            f"dps={dps['V']:.4f} fmap={fmap['V']:.4f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    payload = {
        "rows": rows,
        "args": vars(cli),
        "nfe": {"base": nfe.base, "student": nfe.student, "guide": nfe.guide},
        "conditions": [
            {k: v for k, v in c.items() if k not in ("x_t_1", "t_cond_1")} for c in conditions
        ],
        "checkpoints": {
            "base": str(paths.base_ckpt(cli.length)),
            "dmfm": str(paths.dmfm_ckpt(cli.length)),
        },
        "original_producer": "results/dna/original_scripts/benchmark_yeast_split_value_mc.py",
        "elapsed_s": time.time() - t0,
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(out_path)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
