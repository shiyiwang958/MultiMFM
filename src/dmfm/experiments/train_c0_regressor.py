#!/usr/bin/env python
"""Train a parent-disjoint 50-bp C0 guide or independent oracle.

Ported from dirichlet-flow-matching/scripts/train_parent_c0_regressor.py (Table 11).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dmfm import paths
from dmfm.regressors.c0 import build_c0_regressor
from dmfm.utils.torch_io import torch_load
from dmfm.utils.yeast_splits import load_yeast_split_indices


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pt", default=str(paths.data_pt(50)))
    parser.add_argument("--split_pt", default=str(paths.split_pt(50)))
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--validation_split", required=True)
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--role", choices=["guide", "oracle"], required=True)
    parser.add_argument("--model_type", choices=["park_cnn", "independent_oracle"], default="park_cnn")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reverse_complement_augmentation", action="store_true")
    parser.add_argument("--reverse_complement_average", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def reverse_complement_one_hot(x_acgt: torch.Tensor) -> torch.Tensor:
    return x_acgt.flip(dims=(1,))[..., [3, 2, 1, 0]]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman": float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank(), method="pearson")),
    }


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    reverse_complement_average: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        prediction = model(x)
        if reverse_complement_average:
            prediction = 0.5 * (prediction + model(reverse_complement_one_hot(x)))
        labels.append(y.cpu())
        predictions.append(prediction.cpu())
    return torch.cat(labels).numpy(), torch.cat(predictions).numpy()


def assert_disjoint_parent_splits(payload: dict, split_indices: dict[str, torch.Tensor]) -> dict[str, int]:
    parent_id = payload.get("parent_id")
    if not isinstance(parent_id, torch.Tensor):
        raise ValueError("Parent-disjoint C0 training requires parent_id in the dataset payload")
    parent_sets = {name: set(parent_id[index].tolist()) for name, index in split_indices.items()}
    names = list(parent_sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = parent_sets[first] & parent_sets[second]
            if overlap:
                raise ValueError(f"Parent leakage between {first} and {second}: {sorted(overlap)[:5]}")
    return {name: len(parents) for name, parents in parent_sets.items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload = torch_load(args.data_pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary payload with seqs, c0, and parent_id")
    tokens, labels = payload.get("seqs"), payload.get("c0")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[1] != 50:
        raise ValueError("Expected 50-bp integer tokens in payload['seqs']")
    if not isinstance(labels, torch.Tensor) or labels.shape != (len(tokens),):
        raise ValueError("Expected one measured C0 label per 50-bp sequence in payload['c0']")
    if not torch.isfinite(labels).all():
        raise ValueError("C0 labels must be finite")

    split_indices = {
        name: load_yeast_split_indices(args.split_pt, split)
        for name, split in {
            "train": args.train_split,
            "validation": args.validation_split,
            "test": args.test_split,
        }.items()
    }
    parent_counts = assert_disjoint_parent_splits(payload, split_indices)
    if any(len(index) == 0 for index in split_indices.values()):
        raise ValueError("Training, validation, and test partitions must all be non-empty")

    x = F.one_hot(tokens.long(), num_classes=4).float()
    y = labels.float()
    make_loader = lambda index, shuffle: DataLoader(
        TensorDataset(x[index], y[index]),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    train_loader = make_loader(split_indices["train"], True)
    val_loader = make_loader(split_indices["validation"], False)
    test_loader = make_loader(split_indices["test"], False)

    device = torch.device(args.device)
    model = build_c0_regressor(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_mse = 0.0
        total_examples = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if args.reverse_complement_augmentation:
                batch_x = torch.cat((batch_x, reverse_complement_one_hot(batch_x)), dim=0)
                batch_y = torch.cat((batch_y, batch_y), dim=0)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.detach()) * len(batch_y)
            total_examples += len(batch_y)

        val_y, val_prediction = predict(model, val_loader, device, args.reverse_complement_average)
        val_metrics = metrics(val_y, val_prediction)
        row = {"epoch": epoch, "train_mse": total_mse / total_examples, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best_state.pt")
    val_y, val_prediction = predict(model, val_loader, device, args.reverse_complement_average)
    test_y, test_prediction = predict(model, test_loader, device, args.reverse_complement_average)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame({"label": val_y, "prediction": val_prediction, "partition": "validation"}).to_csv(
        out_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame({"label": test_y, "prediction": test_prediction, "partition": "test"}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )
    metadata = {
        "args": vars(args),
        "parent_counts": parent_counts,
        "n_train": int(len(split_indices["train"])),
        "n_validation": int(len(split_indices["validation"])),
        "n_test": int(len(split_indices["test"])),
        "validation_metrics": metrics(val_y, val_prediction),
        "test_metrics": metrics(test_y, test_prediction),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
