#!/usr/bin/env python
"""Standalone DFM-style DNA DiT trainer (base DFM models, Table 9).

Ported from ``dirichlet-flow-matching/scripts/train_dna_dfm_dit.py`` (Jul 25
2026; launched by ``slurm/train_yeast_parent_L{50,100,200,400}_dfm.sbatch``).

Train the diagonal DFM denoiser with cross-entropy to the clean token,
optionally calibrate the learning rate with a small Armijo probe, save
checkpoints, and run the lightweight DNA-FID diagnostic against a uniform
baseline.

Changes vs. the original: package imports; ``--yeast_data_pt`` defaults to
``data/dna/yeast_parent_disjoint/yeast_parent_L50.pt`` and ``--output_dir`` to
``outputs/dna/<run_name>_<stamp>``; ``main(argv)`` accepts an argument list.

    python -m dmfm.experiments.train_base_dfm --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset

from dmfm import paths
from dmfm.models.dna_models import DiTSequenceModel
from dmfm.utils.dataset import YeastMiddleDataset
from dmfm.utils.flow_utils import gaussian_beta, gaussian_denoiser_flow_step
from dmfm.utils.torch_io import torch_load
from dmfm.utils.yeast_splits import subset_by_yeast_split


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data / output.
    p.add_argument("--dataset_type", choices=["yeast"], default="yeast")
    p.add_argument("--yeast_data_pt", default=str(paths.data_pt(50)))
    p.add_argument("--yeast_split_pt", default=None, help="Optional torch split file with train/val/test indices.")
    p.add_argument("--yeast_split", default="train", help="Named split in --yeast_split_pt used for training.")
    p.add_argument(
        "--yeast_val_split",
        default=None,
        help="Optional named split in --yeast_split_pt used only for checkpoint selection.",
    )
    p.add_argument("--max_train_seqs", type=int, default=20000)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default="yeast_parent_dfm_dit")  # original default: yeast_mid50_dfm_dit
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)

    # Model.
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--transformer_heads", type=int, default=6)
    p.add_argument("--transformer_ff_mult", type=float, default=4.0)
    p.add_argument("--dit_blocks", type=int, default=4)
    p.add_argument("--dit_cond_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--self_condition_ratio", type=float, default=0.0)
    p.add_argument("--use_flash_attn", action="store_true")
    p.add_argument("--softcap", type=float, default=50.0)
    p.add_argument("--flow_temp", type=float, default=1.0)
    p.add_argument("--gaussian_beta_schedule", choices=["linear", "table"], default="linear")
    p.add_argument("--gaussian_beta_table_path", default=None)

    # Optimization.
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument(
        "--val_batches",
        type=int,
        default=0,
        help="Validation batches per checkpoint; 0 evaluates the complete validation split.",
    )
    p.add_argument("--ckpt", default=None, help="Resume checkpoint path.")

    # Optional DFM adaptive CE weighting.
    p.add_argument("--use_adaptive", action="store_true")
    p.add_argument("--adaptive_p", type=float, default=0.5)
    p.add_argument("--adaptive_c", type=float, default=0.01)

    # Armijo LR probe. This estimates a raw-gradient descent LR on fixed noising
    # draws, then uses a safety factor for AdamW.
    p.add_argument("--armijo", action="store_true")
    p.add_argument("--armijo_start_lr", type=float, default=1e-3)
    p.add_argument("--armijo_batches", type=int, default=4)
    p.add_argument("--armijo_c", type=float, default=1e-4)
    p.add_argument("--armijo_tau", type=float, default=0.5)
    p.add_argument("--armijo_max_backtracks", type=int, default=20)
    p.add_argument("--armijo_min_lr", type=float, default=1e-8)
    p.add_argument("--armijo_safety", type=float, default=0.25)

    # End-of-run diagnostics.
    p.add_argument("--eval_samples", type=int, default=2048)
    p.add_argument("--gen_steps", type=int, default=64)
    p.add_argument("--gen_batch", type=int, default=256)
    p.add_argument("--num_examples", type=int, default=12)

    return p.parse_args(argv)


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    hidden = int(args.hidden_dim)
    return SimpleNamespace(
        hidden_dim=hidden,
        transformer_heads=int(args.transformer_heads),
        transformer_ff_mult=float(args.transformer_ff_mult),
        dit_blocks=int(args.dit_blocks),
        dit_cond_dim=int(args.dit_cond_dim or hidden),
        dropout=float(args.dropout),
        clean_data=False,
        self_condition_ratio=float(args.self_condition_ratio),
        time_delta_param=True,
        double_temb=True,
        scale_by_sigma=False,
        preserve_denoiser=False,
        use_flash_attn=bool(args.use_flash_attn),
        softcap=float(args.softcap),
        gaussian_beta_schedule=args.gaussian_beta_schedule,
        gaussian_beta_table_path=args.gaussian_beta_table_path,
        flow_temp=float(args.flow_temp),
    )


def seq_from_batch(batch: object) -> torch.Tensor:
    seq = batch["seq"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (tuple, list)) else batch)
    if seq.ndim == 3 and seq.shape[-1] == 1:
        seq = seq.squeeze(-1)
    return seq.long()


def unwrap_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def make_loader(args: argparse.Namespace, *, split: str | None = None, shuffle: bool = True) -> DataLoader:
    if args.dataset_type != "yeast":
        raise ValueError(f"Unsupported dataset_type={args.dataset_type!r}; this repo keeps only the yeast DNA path.")
    ds = YeastMiddleDataset(args, path=args.yeast_data_pt)
    ds = subset_by_yeast_split(ds, args.yeast_split_pt, args.yeast_split if split is None else split)

    if shuffle and hasattr(ds, "__len__") and args.max_train_seqs and len(ds) > args.max_train_seqs:
        g = torch.Generator().manual_seed(args.seed)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[: args.max_train_seqs])

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def validation_metrics(
    model: DiTSequenceModel,
    loader: DataLoader,
    cfg: SimpleNamespace,
    *,
    device: torch.device,
    seed: int,
    max_batches: int,
) -> dict[str, float]:
    """Evaluate a fixed diagonal-DFM denoising task for reproducible checkpoint selection."""
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    n_examples = 0
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            seq = seq_from_batch(batch).to(device, non_blocking=True)
            loss, log_psi = dfm_vfm_loss(model, seq, cfg)
            losses.append(loss.detach() * seq.shape[0])
            accuracies.append((log_psi.argmax(-1) == seq).float().mean().detach() * seq.shape[0])
            n_examples += seq.shape[0]
    if was_training:
        model.train()
    if not losses:
        raise RuntimeError("Validation loader yielded no batches")
    return {
        "loss": float(torch.stack(losses).sum().item() / n_examples),
        "token_acc": float(torch.stack(accuracies).sum().item() / n_examples),
        "n_examples": float(n_examples),
    }


def fixed_noise(seq: torch.Tensor, cfg: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = F.one_hot(seq, num_classes=cfg.alphabet_size).float()
    x0 = torch.randn_like(x1)
    t = torch.rand(seq.shape[0], device=seq.device)
    return x0, x1, t


def dfm_vfm_loss(
    model: DiTSequenceModel,
    seq: torch.Tensor,
    cfg: SimpleNamespace,
    *,
    x0: torch.Tensor | None = None,
    x1: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x0 is None or x1 is None or t is None:
        x0, x1, t = fixed_noise(seq, cfg)

    beta = gaussian_beta(cfg, t)[:, None, None]
    xt = (1.0 - beta) * x0 + beta * x1

    log_psi = model.psi_st(xt, t, t, return_log_psi=True)
    ce = -log_psi.gather(-1, seq[..., None]).squeeze(-1)

    if cfg.use_adaptive:
        with torch.no_grad():
            psi = log_psi.exp()
            q_true = psi.gather(-1, seq[..., None]).squeeze(-1)
            delta = psi.pow(2).sum(-1) - 2.0 * q_true + 1.0
            weight = (delta + cfg.adaptive_c).pow(-cfg.adaptive_p)
        ce = weight * ce

    return ce.mean(), log_psi


def armijo_probe_batch(
    model: DiTSequenceModel,
    seq: torch.Tensor,
    cfg: SimpleNamespace,
    *,
    start_lr: float,
    c: float,
    tau: float,
    max_backtracks: int,
    min_lr: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    x0, x1, t = fixed_noise(seq, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    loss0, _ = dfm_vfm_loss(model, seq, cfg, x0=x0, x1=x1, t=t)
    grads = torch.autograd.grad(loss0, params, create_graph=False)
    grad_norm_sq = torch.stack([g.detach().float().pow(2).sum() for g in grads]).sum()

    backups = [p.detach().clone() for p in params]
    lr = float(start_lr)
    accepted = False
    loss_new = torch.tensor(float("nan"), device=seq.device)

    with torch.no_grad():
        if torch.isfinite(grad_norm_sq) and grad_norm_sq.item() > 0.0:
            for n_backtracks in range(max_backtracks + 1):
                for p, p0, g in zip(params, backups, grads):
                    p.copy_(p0 - lr * g)
                loss_new, _ = dfm_vfm_loss(model, seq, cfg, x0=x0, x1=x1, t=t)
                sufficient_decrease = loss0 - c * lr * grad_norm_sq
                if torch.isfinite(loss_new) and loss_new <= sufficient_decrease:
                    accepted = True
                    break
                lr *= tau
                if lr < min_lr:
                    break
        else:
            n_backtracks = 0

        for p, p0 in zip(params, backups):
            p.copy_(p0)

    if was_training:
        model.train()

    return {
        "accepted": float(accepted),
        "lr": float(lr if accepted else min_lr),
        "loss0": float(loss0.detach().item()),
        "loss_new": float(loss_new.detach().item()) if torch.isfinite(loss_new) else float("nan"),
        "grad_norm": float(grad_norm_sq.sqrt().detach().item()),
        "backtracks": float(n_backtracks),
    }


def armijo_find_lr(
    model: DiTSequenceModel,
    loader: DataLoader,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    stats = []
    it = iter(loader)
    for _ in range(args.armijo_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        seq = seq_from_batch(batch).to(device, non_blocking=True)
        stats.append(
            armijo_probe_batch(
                model,
                seq,
                cfg,
                start_lr=args.armijo_start_lr,
                c=args.armijo_c,
                tau=args.armijo_tau,
                max_backtracks=args.armijo_max_backtracks,
                min_lr=args.armijo_min_lr,
            )
        )

    accepted = [s["lr"] for s in stats if s["accepted"] > 0.0]
    if not accepted:
        lr = args.armijo_min_lr
    else:
        lr = float(np.median(np.asarray(accepted)) * args.armijo_safety)

    print("Armijo LR probe:")
    for i, s in enumerate(stats):
        print(
            f"  batch={i} accepted={bool(s['accepted'])} lr={s['lr']:.3e} "
            f"loss0={s['loss0']:.4f} loss_new={s['loss_new']:.4f} "
            f"grad={s['grad_norm']:.3e} backtracks={int(s['backtracks'])}"
        )
    print(f"Using lr={lr:.3e} after safety={args.armijo_safety:g}")
    return lr


def save_checkpoint(
    path: Path,
    *,
    model: DiTSequenceModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    step: int,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "model_cfg": vars(cfg),
            "metrics": metrics or {},
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(
    path: str,
    *,
    model: DiTSequenceModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device,
) -> int:
    ckpt = torch_load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("step", 0))


@torch.no_grad()
def generate(model: DiTSequenceModel, cfg: SimpleNamespace, args: argparse.Namespace, n: int, device: torch.device) -> torch.Tensor:
    model.eval()
    outs = []
    grid = torch.linspace(0, 1, args.gen_steps + 1, device=device)
    for start in range(0, n, args.gen_batch):
        b = min(args.gen_batch, n - start)
        x = torch.randn(b, cfg.seq_len, cfg.alphabet_size, device=device)
        for s, t in zip(grid[:-1], grid[1:]):
            x, _, _ = gaussian_denoiser_flow_step(cfg, model, x, s.expand(b), t.expand(b))
        outs.append(x.argmax(-1).cpu())
    return torch.cat(outs, 0)


def collect_real(loader: DataLoader, n: int) -> torch.Tensor:
    seqs = []
    it = iter(loader)
    while sum(x.shape[0] for x in seqs) < n:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        seqs.append(seq_from_batch(batch).cpu())
    return torch.cat(seqs, 0)[:n]


def kmer_features(seq: torch.Tensor, k: int, alphabet_size: int) -> torch.Tensor:
    n, length = seq.shape
    code = torch.zeros(n, length - k + 1, dtype=torch.long)
    for j in range(k):
        code = code + seq[:, j : length - k + 1 + j] * (alphabet_size ** (k - 1 - j))
    return F.one_hot(code, num_classes=alphabet_size**k).float().mean(1)


def feature_sets(seq: torch.Tensor, alphabet_size: int) -> dict[str, np.ndarray]:
    onehot = F.one_hot(seq, num_classes=alphabet_size).float()
    pos = onehot.reshape(seq.shape[0], -1)
    k23 = torch.cat([kmer_features(seq, 2, alphabet_size), kmer_features(seq, 3, alphabet_size)], -1)
    return {
        "pos": pos.numpy(),
        "kmer23": k23.numpy(),
        "all": torch.cat([pos, k23], -1).numpy(),
    }


def frechet(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])
    cm = sqrtm(s1 @ s2)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm))


def run_dna_fid(
    model: DiTSequenceModel,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    real = collect_real(loader, args.eval_samples)
    gen = generate(model, cfg, args, args.eval_samples, device)
    rand = torch.randint(0, cfg.alphabet_size, real.shape)

    real_a = real[: args.eval_samples // 2]
    real_b = real[args.eval_samples // 2 :]
    features = {
        "real": feature_sets(real, cfg.alphabet_size),
        "real_a": feature_sets(real_a, cfg.alphabet_size),
        "real_b": feature_sets(real_b, cfg.alphabet_size),
        "gen": feature_sets(gen, cfg.alphabet_size),
        "rand": feature_sets(rand, cfg.alphabet_size),
    }

    metrics = {}
    print("\nDNA-FID lower is better")
    for name in ("pos", "kmer23", "all"):
        floor = frechet(features["real_a"][name], features["real_b"][name])
        fid_gen = frechet(features["real"][name], features["gen"][name])
        fid_rand = frechet(features["real"][name], features["rand"][name])
        metrics[name] = {
            "real_split": floor,
            "generated": fid_gen,
            "uniform": fid_rand,
            "gen_uniform_ratio": fid_gen / fid_rand,
        }
        print(
            f"{name:7s} real_split={floor:.4f}  generated={fid_gen:.4f}  "
            f"uniform={fid_rand:.4f}  gen/uniform={fid_gen / fid_rand:.3f}"
        )

    alphabet = "ACGT"
    print("\nGenerated examples:")
    examples = []
    for s in gen[: args.num_examples]:
        decoded = "".join(alphabet[int(i)] if cfg.alphabet_size == 4 else str(int(i)) for i in s)
        examples.append(decoded)
        print(decoded)
    metrics["examples"] = {"seqs": examples}
    return metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = str(paths.OUTPUTS / f"{args.run_name}_{stamp}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}", flush=True)

    loader = make_loader(args)
    val_loader = None
    if args.yeast_val_split is not None:
        if not args.yeast_split_pt:
            raise ValueError("--yeast_val_split requires --yeast_split_pt")
        val_loader = make_loader(args, split=args.yeast_val_split, shuffle=False)
    first_seq = seq_from_batch(next(iter(loader)))
    base_ds = unwrap_dataset(loader.dataset)

    cfg = make_model_args(args)
    cfg.alphabet_size = int(getattr(base_ds, "alphabet_size", first_seq.max().item() + 1))
    cfg.seq_len = int(first_seq.shape[1])
    cfg.use_adaptive = bool(args.use_adaptive)
    cfg.adaptive_p = float(args.adaptive_p)
    cfg.adaptive_c = float(args.adaptive_c)

    model = DiTSequenceModel(cfg, alphabet_size=cfg.alphabet_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.ckpt:
        start_step = load_checkpoint(args.ckpt, model=model, optimizer=optimizer, device=device)
        print(f"Resumed from {args.ckpt} at step={start_step}")

    if args.armijo:
        args.lr = armijo_find_lr(model, loader, cfg, args, device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    with open(out_dir / "args.json", "w") as f:
        json.dump({"args": vars(args), "model_cfg": vars(cfg)}, f, indent=2, sort_keys=True)

    print(
        f"Training {args.dataset_type} DFM-DiT on {device}; "
        f"N={len(loader.dataset)} L={cfg.seq_len} K={cfg.alphabet_size} "
        f"steps={args.steps} batch={args.batch_size} lr={args.lr:g}"
    )

    it = iter(loader)
    model.train()
    last_metrics = {}
    best_val_loss = float("inf")
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        seq = seq_from_batch(batch).to(device, non_blocking=True)
        loss, log_psi = dfm_vfm_loss(model, seq, cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.print_every == 0:
            acc = (log_psi.argmax(-1) == seq).float().mean().item()
            last_metrics = {"loss": float(loss.item()), "token_acc": float(acc), "grad": float(grad)}
            print(
                f"{step:06d} loss={loss.item():.4f} token_acc={acc:.4f} "
                f"grad={float(grad):.3f} lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )

        if val_loader is not None and args.val_every > 0 and (step % args.val_every == 0 or step == args.steps):
            val = validation_metrics(
                model,
                val_loader,
                cfg,
                device=device,
                seed=args.seed + 12_345,
                max_batches=args.val_batches,
            )
            last_metrics["validation"] = val
            print(
                f"{step:06d} val_loss={val['loss']:.4f} val_token_acc={val['token_acc']:.4f} "
                f"n={int(val['n_examples'])}",
                flush=True,
            )
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save_checkpoint(
                    out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    cfg=cfg,
                    step=step,
                    metrics=last_metrics,
                )
                print(f"Saved best validation checkpoint at step={step}", flush=True)

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                out_dir / f"step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                cfg=cfg,
                step=step,
                metrics=last_metrics,
            )
            save_checkpoint(
                out_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                cfg=cfg,
                step=step,
                metrics=last_metrics,
            )
            print(f"Saved checkpoint at step={step} to {out_dir / 'last.pt'}", flush=True)

    eval_metrics = {}
    if args.eval_samples > 0:
        eval_metrics = run_dna_fid(model, cfg, args, loader, device)
        with open(out_dir / "eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2, sort_keys=True)

    save_checkpoint(
        out_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        cfg=cfg,
        step=args.steps,
        metrics={"train": last_metrics, "eval": eval_metrics},
    )
    save_checkpoint(
        out_dir / f"step_{args.steps:08d}.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        cfg=cfg,
        step=args.steps,
        metrics={"train": last_metrics, "eval": eval_metrics},
    )
    print(f"\nSaved checkpoint to {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
