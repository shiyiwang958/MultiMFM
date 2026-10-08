"""QM9 data loading + TFG-Flow property regressors.

Loads QM9 in the TFG-Flow representation (so the pretrained property regressors
apply directly) and provides ``load_tfg_regressor`` for the guide/oracle heads.
The TFG-Flow QM9 convention followed here:

- heavy-only molecules by default, padded to ``max_len=9``
- atom order ``C,H,N,O,F,P,S,Cl,*`` where ``*`` is TFG's ``MASK``
- coordinates centered but not divided by a dataset normalizer
- split after a deterministic NumPy shuffle with seed 42

``load_tfg_regressor(kind, prop, device)`` instantiates a TFG ``Predictor`` and
loads the guide/oracle weights from ``checkpoints/regressors/{kind}_clf_ckpt.zip``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "qm9"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "qm9" / "datasets")
)
os.environ.setdefault(
    "HF_HUB_CACHE", str(REPO_ROOT / "data" / "qm9" / "hub")
)
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "cache" / "xdg"))

from datasets import load_dataset  # noqa: E402

from tabasco.chem.convert import MoleculeConverter  # noqa: E402
from tabasco.data.utils import TensorDictCollator  # noqa: E402
from multimfm.train_base_flow import (  # noqa: E402
    build_model,
    compute_pb_summary,
    install_posebusters_compat,
    set_seed,
)

TFG_QM9_ATOM_NAMES = ["C", "H", "N", "O", "F", "P", "S", "Cl", "*"]
TFG_QM9_ATOM_TO_INDEX = {atom: idx for idx, atom in enumerate(TFG_QM9_ATOM_NAMES)}
TFG_SPLITS = {
    "train_flow": (0, 50_000),
    "train_classifier": (50_000, 100_000),
    "validation": (100_000, 101_800),
    "test": (101_800, None),
}
REGRESSOR_LOSSES = {
    "guide": {
        "alpha": "0.12717216682434093",
        "homo": "0.0019835512804612524",
        "lumo": "0.0017945603752836663",
        "mu": "0.20151185317914316",
        "cv": "0.0604787247697512",
        "gap": "0.0030602489901830746",
    },
    "oracle": {
        "alpha": "0.13345399435361235",
        "homo": "0.0021804291078696644",
        "lumo": "0.0017926631986916373",
        "mu": "0.18175863907186626",
        "cv": "0.05778919728199628",
        "gap": "0.003081417501829565",
    },
}


class TensorListDataset(Dataset):
    def __init__(self, tensors: list[TensorDict]):
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> TensorDict:
        return self.tensors[index].clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=sorted(TFG_SPLITS), default="train_flow")
    parser.add_argument("--limit-data", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=9)
    parser.add_argument("--include-hydrogens", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Write training metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/qm9_dfm_smoke/train_metrics.csv"),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N optimizer steps; 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/qm9_dfm_smoke/checkpoints"),
    )
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--check-regressor", action="store_true")
    parser.add_argument("--regressor-kind", choices=["guide", "oracle"], default="guide")
    parser.add_argument(
        "--property",
        choices=["alpha", "homo", "lumo", "mu", "cv", "gap"],
        default="alpha",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/qm9_dfm_smoke/summary.json"),
    )
    parser.add_argument(
        "--pb-every",
        type=int,
        default=0,
        help="If >0, sample and log PoseBusters metrics every N training steps.",
    )
    parser.add_argument(
        "--pb-eval-step-zero",
        action="store_true",
        help="Also run PB evaluation before the first optimizer step.",
    )
    parser.add_argument("--pb-samples", type=int, default=16)
    parser.add_argument("--pb-batch-size", type=int, default=8)
    parser.add_argument("--pb-sample-steps", type=int, default=32)
    parser.add_argument(
        "--posebusters-config",
        type=Path,
        default=Path("src/tabasco/utils/posebusters_no_strain.yaml"),
    )
    parser.add_argument(
        "--pb-csv",
        type=Path,
        default=Path("outputs/qm9_dfm_smoke/pb_curve.csv"),
    )
    parser.add_argument("--pb-no-sanitize", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tfg_partition_indices(num_rows: int, partition: str, limit: int) -> np.ndarray:
    start, stop = TFG_SPLITS[partition]
    indices = np.arange(num_rows)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    selected = indices[start:stop]
    if limit > 0:
        selected = selected[:limit]
    return selected


def datapoint_to_tensor(
    datapoint: dict,
    *,
    max_len: int,
    include_hydrogens: bool,
) -> TensorDict | None:
    symbols = list(datapoint["atomic_symbols"])
    positions = np.asarray(datapoint["pos"], dtype=np.float32)
    keep = [
        idx
        for idx, symbol in enumerate(symbols)
        if include_hydrogens or symbol != "H"
    ]
    symbols = [symbols[idx] for idx in keep]
    positions = positions[keep]
    if len(symbols) == 0 or len(symbols) > max_len:
        return None

    coords = torch.zeros(max_len, 3, dtype=torch.float32)
    centered = positions - positions.mean(axis=0, keepdims=True)
    coords[: len(symbols)] = torch.from_numpy(centered)

    atomics = torch.zeros(max_len, len(TFG_QM9_ATOM_NAMES), dtype=torch.float32)
    mask_idx = TFG_QM9_ATOM_TO_INDEX["*"]
    for atom_idx, symbol in enumerate(symbols):
        atomics[atom_idx, TFG_QM9_ATOM_TO_INDEX.get(symbol, mask_idx)] = 1.0
    if len(symbols) < max_len:
        atomics[len(symbols) :, mask_idx] = 1.0

    padding_mask = torch.ones(max_len, dtype=torch.bool)
    padding_mask[: len(symbols)] = False
    return TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=[],
    )


def load_qm9_tensors(args: argparse.Namespace) -> tuple[list[TensorDict], dict]:
    dataset = load_dataset(
        "yairschiff/qm9",
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    selected = tfg_partition_indices(len(dataset), args.partition, args.limit_data)
    tensors: list[TensorDict] = []
    atom_counts: Counter[int] = Counter()
    element_counts: Counter[str] = Counter()
    skipped = 0
    for idx in selected:
        datapoint = dataset[int(idx)]
        tensor = datapoint_to_tensor(
            datapoint,
            max_len=args.max_len,
            include_hydrogens=args.include_hydrogens,
        )
        if tensor is None:
            skipped += 1
            continue
        real_mask = ~tensor["padding_mask"]
        n_atoms = int(real_mask.sum().item())
        atom_counts[n_atoms] += 1
        atom_idx = tensor["atomics"][real_mask].argmax(dim=-1).tolist()
        element_counts.update(TFG_QM9_ATOM_NAMES[i] for i in atom_idx)
        tensors.append(tensor)

    if not tensors:
        raise RuntimeError("No usable QM9 tensors were built.")

    stats = {
        "num_atoms_histogram": dict(atom_counts),
        "max_num_atoms": args.max_len,
        "spatial_dim": 3,
        "atom_dim": len(TFG_QM9_ATOM_NAMES),
        "atom_names": list(TFG_QM9_ATOM_NAMES),
        "include_hydrogens": args.include_hydrogens,
        "coordinate_normalizer": 1.0,
        "partition": args.partition,
        "source": "yairschiff/qm9",
        "tfg_shuffle_seed": 42,
        "skipped": skipped,
        "element_counts": dict(element_counts),
    }
    return tensors, stats


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        implementation="reimplemented",
        no_cross_attention=False,
        num_random_augmentations=0,
        sample_schedule="linear",
        atomics_mode="dfm_gaussian",
        dfm_beta_schedule="linear",
        dfm_beta_power=1.0,
    )


def tensor_batch_to_tfg_batch(batch: TensorDict) -> dict[str, torch.Tensor]:
    return {
        "coors": batch["coords"],
        "atom_types": batch["atomics"].argmax(dim=-1).long(),
        "mask": (~batch["padding_mask"]).long(),
    }


def load_tfg_regressor(kind: str, prop: str, device: torch.device):
    tfg_root = REPO_ROOT / "third_party" / "tfg_flow"
    sys.path.insert(0, str(tfg_root))
    from diffusion.predictor import Predictor  # noqa: PLC0415

    model = Predictor(
        max_len=9,
        device=str(device),
        h_embed_size=128,
        num_layers=6,
        cls_embed_size=64,
        e_embed_size=128,
        class_num=1,
    ).to(device)
    archive = REPO_ROOT / "checkpoints" / "regressors" / f"{kind}_clf_ckpt.zip"
    inner_dir = f"{kind}_clf_ckpt"
    loss = REGRESSOR_LOSSES[kind][prop]
    inner_name = f"{inner_dir}/best_loss={loss}.pth"
    with zipfile.ZipFile(archive) as zf:
        state = torch.load(io.BytesIO(zf.read(inner_name)), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_posebusters(config_path: Path):
    install_posebusters_compat()
    import yaml
    from posebusters import PoseBusters

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return PoseBusters(config=config)


def sample_molecules_for_pb(
    model,
    *,
    stats: dict,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    sanitize: bool,
) -> tuple[list, list[int], int]:
    converter = MoleculeConverter(
        atom_names=list(stats["atom_names"]),
        dataset_normalizer=float(stats.get("coordinate_normalizer", 1.0)),
    )
    mols = []
    mol_indices = []
    total_generated = 0

    model.eval()
    while total_generated < num_samples:
        current_batch_size = min(batch_size, num_samples - total_generated)
        with torch.inference_mode():
            generated = model.sample(
                batch_size=current_batch_size,
                num_steps=num_steps,
            )
        batch_mols = converter.from_batch(
            generated.detach().cpu(),
            sanitize=sanitize,
            rescale_coords=True,
        )
        for local_idx, mol in enumerate(batch_mols):
            if mol is not None:
                mols.append(mol)
                mol_indices.append(total_generated + local_idx)
        total_generated += current_batch_size
    return mols, mol_indices, total_generated


def evaluate_pb(model, posebusters, args: argparse.Namespace, step: int, loss: float):
    mols, mol_indices, total_generated = sample_molecules_for_pb(
        model,
        stats=model.data_stats,
        num_samples=args.pb_samples,
        batch_size=args.pb_batch_size,
        num_steps=args.pb_sample_steps,
        sanitize=not args.pb_no_sanitize,
    )
    summary = compute_pb_summary(
        posebusters=posebusters,
        mols=mols,
        mol_indices=mol_indices,
        total_generated=total_generated,
    )
    row = {"step": step, "train_loss": loss, **summary}
    append_csv_row(args.pb_csv, row)
    print(
        "PB "
        f"step={step} "
        f"loss={loss:.4f} "
        f"converted={summary.get('conversion_rate', 0.0):.3f} "
        f"pb_valid={summary.get('pb_valid', 0.0):.3f}"
    )
    return row


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    stats: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "stats": stats,
            "args": vars(args),
            "model_args": vars(make_model_args(args)),
        },
        path,
    )


def train_metric_row(step: int, loss, stat_dict: dict) -> dict:
    row = {"step": step, "train_loss": float(loss.detach().cpu())}
    for key, value in stat_dict.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, (int, float)):
            row[key] = float(value)
    return row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)

    tensors, stats = load_qm9_tensors(args)
    loader = DataLoader(
        TensorListDataset(tensors),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TensorDictCollator(),
    )

    model = build_model(make_model_args(args), stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []
    last_stats: dict[str, float] = {}
    pb_history: list[dict] = []
    posebusters = None
    if args.pb_every > 0 or args.pb_eval_step_zero:
        posebusters = build_posebusters(args.posebusters_config)
        if args.pb_csv.exists():
            args.pb_csv.unlink()
    if args.train_csv.exists():
        args.train_csv.unlink()

    model.train()
    iterator = iter(loader)
    if args.pb_eval_step_zero:
        pb_history.append(evaluate_pb(model, posebusters, args, step=0, loss=float("nan")))
        model.train()

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        loss, stat_dict = model(batch, compute_stats=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        last_stats = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in stat_dict.items()
            if key.endswith("_loss") or key in {"atomics_logit_norm", "coords_logit_norm"}
        }
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            row = train_metric_row(step, loss, stat_dict)
            append_csv_row(args.train_csv, row)
            print(
                "TRAIN "
                f"step={step} "
                f"loss={row.get('train_loss', float('nan')):.4f} "
                f"coords={row.get('coords_loss', float('nan')):.4f} "
                f"atomics={row.get('atomics_loss', float('nan')):.4f}"
            )
        if args.pb_every > 0 and step % args.pb_every == 0:
            pb_history.append(
                evaluate_pb(
                    model,
                    posebusters,
                    args,
                    step=step,
                    loss=float(loss.detach().cpu()),
                )
            )
            model.train()
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"model_step_{step}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                stats=stats,
                args=args,
            )

    model.eval()
    with torch.no_grad():
        sample = model.sample(
            batch_size=args.sample_batch_size,
            num_steps=args.sample_steps,
        )
    sample_atom_idx = sample["atomics"].argmax(dim=-1)
    dummy_idx = TFG_QM9_ATOM_TO_INDEX["*"]
    real_mask = ~sample["padding_mask"]
    summary = {
        "device": str(device),
        "num_tensors": len(tensors),
        "stats": stats,
        "losses": losses,
        "last_train_stats": last_stats,
        "sample_shape": {
            "coords": list(sample["coords"].shape),
            "atomics": list(sample["atomics"].shape),
        },
        "sample_atom_counts": real_mask.sum(dim=1).cpu().tolist(),
        "sample_dummy_fraction_real": float(
            ((sample_atom_idx == dummy_idx) & real_mask).sum().cpu()
            / real_mask.sum().clamp_min(1).cpu()
        ),
        "sample_coords_finite": bool(torch.isfinite(sample["coords"]).all().item()),
        "train_csv": str(args.train_csv),
        "pb_csv": str(args.pb_csv) if pb_history else None,
        "pb_history": pb_history,
    }

    if args.check_regressor:
        regressor = load_tfg_regressor(args.regressor_kind, args.property, device)
        first_batch = next(iter(loader)).to(device)
        tfg_batch = tensor_batch_to_tfg_batch(first_batch)
        with torch.no_grad():
            pred = regressor(
                tfg_batch["coors"].float(),
                tfg_batch["atom_types"],
                tfg_batch["mask"],
            )
        summary["regressor"] = {
            "kind": args.regressor_kind,
            "property": args.property,
            "prediction_shape": list(pred.shape),
            "prediction_mean": float(pred.mean().detach().cpu()),
            "prediction_std": float(pred.std(unbiased=False).detach().cpu()),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_final_checkpoint:
        save_checkpoint(
            args.checkpoint_dir / f"model_step_{args.train_steps}.pt",
            model=model,
            optimizer=optimizer,
            step=args.train_steps,
            stats=stats,
            args=args,
        )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
