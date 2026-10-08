"""Train a small TABASCO model and track PB validity during training.

This is intentionally lighter than the main Hydra trainer. It builds the same
model components used by the repo config, trains for a fixed number of gradient
steps, periodically samples molecules, and writes a CSV of PoseBusters rates.

Examples:
    # Use the Hugging Face GEOM-Drugs train split, cached under ./cache/datasets
    python src/train_pb_valid_curve.py --geom-drugs-split train

    # Use a provided processed .pt file in the repo format: [(protein, mol), ...]
    python src/train_pb_valid_curve.py --data-path data/processed_geom_train.pt

    # Quick architecture smoke test without external data
    python src/train_pb_valid_curve.py --tiny-smiles --max-steps 20 --eval-every 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from contextlib import contextmanager
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import re

# Keep matplotlib/posebusters import-time cache files out of the home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))

import numpy as np
import pandas as pd
import torch
import yaml
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from tabasco.chem.convert import MoleculeConverter
from tabasco.chem.constants import ATOM_NAMES, ATOM_NAMES_WITH_H
from tabasco.data.utils import TensorDictCollator
from tabasco.flow.interpolate import (
    DiscreteInterpolant,
    GaussianSimplexAtomInterpolant,
    SDEMetricInterpolant,
)
from tabasco.flow.time_factor import InverseTimeFactor
from tabasco.models.components.transformer_module import TransformerModule
from tabasco.models.flow_model import FlowMatchingModel
from tabasco.sample.noise_schedule import SampleNoiseSchedule
from multimfm.posebusters_compat import install_posebusters_compat


DEFAULT_SMILES = [
    "CCO",
    "CCN",
    "CCC",
    "CCCO",
    "CCOC",
    "CCS",
    "CCCl",
    "CCBr",
    "CCF",
    "CC(=O)O",
    "CC(=O)N",
    "CC(C)O",
    "CC(C)N",
    "c1ccccc1",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "CCc1ccccc1",
    "COc1ccccc1",
    "CCN(CC)CC",
]

GEOM_DRUGS_URLS = {
    "train": (
        "https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs/"
        "resolve/main/processed_geom_train.pt"
    ),
    "val": (
        "https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs/"
        "resolve/main/processed_geom_val.pt"
    ),
    "test": (
        "https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs/"
        "resolve/main/processed_geom_test.pt"
    ),
}


class TensorMolDataset(Dataset):
    """Small in-memory dataset of padded molecule TensorDicts."""

    def __init__(
        self,
        tensors: list[TensorDict],
        stats: dict,
        repeat: int = 1,
    ) -> None:
        if not tensors:
            raise ValueError("TensorMolDataset requires at least one molecule.")
        self.tensors = tensors
        self.stats = stats
        self.repeat = max(1, repeat)

    def __len__(self) -> int:
        return len(self.tensors) * self.repeat

    def __getitem__(self, index: int) -> TensorDict:
        return self.tensors[index % len(self.tensors)].clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Local path or URL to a processed .pt file containing RDKit molecules "
            "or (protein, mol) tuples."
        ),
    )
    data.add_argument(
        "--geom-drugs-split",
        choices=["train", "val", "test"],
        default=None,
        help="Download/use carlosinator/tabasco-geom-drugs processed split.",
    )
    data.add_argument(
        "--tiny-smiles",
        action="store_true",
        help="Use a built-in tiny RDKit/SMILES dataset instead of --data-path.",
    )
    data.add_argument(
        "--limit-data",
        type=int,
        default=512,
        help="Maximum molecules to load from --data-path; use <=0 for no limit.",
    )
    data.add_argument(
        "--repeat-dataset",
        type=int,
        default=1,
        help="Repeat the in-memory dataset this many times per epoch.",
    )
    data.add_argument(
        "--include-hydrogens",
        action="store_true",
        help=(
            "Train on explicit hydrogens and add H to the atom vocabulary. "
            "Default keeps the existing heavy-atom-only behavior."
        ),
    )
    data.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/datasets"),
        help="Repo-local cache for downloaded dataset files.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-dim", type=int, default=64)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument(
        "--implementation",
        choices=["reimplemented", "pytorch"],
        default="reimplemented",
    )
    model.add_argument("--no-cross-attention", action="store_true")
    model.add_argument("--num-random-augmentations", type=int, default=0)
    model.add_argument(
        "--sample-schedule",
        choices=["linear", "power", "log"],
        default="log",
    )
    model.add_argument(
        "--atomics-mode",
        choices=["original", "dfm_gaussian"],
        default="original",
        help=(
            "Atom-type architecture. 'original' keeps the categorical TABASCO "
            "path; 'dfm_gaussian' uses Gaussian atom noise with a DFM-style "
            "continuous atom interpolant."
        ),
    )
    model.add_argument(
        "--dfm-beta-schedule",
        choices=["linear", "power"],
        default="linear",
        help="Beta schedule for --atomics-mode dfm_gaussian.",
    )
    model.add_argument(
        "--dfm-beta-power",
        type=float,
        default=1.0,
        help="Power gamma for --dfm-beta-schedule power, beta(t)=t**gamma.",
    )

    train = parser.add_argument_group("training")
    train.add_argument("--max-steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--grad-clip", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--num-workers", type=int, default=0)

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--eval-every", type=int, default=25)
    eval_group.add_argument("--eval-samples", type=int, default=32)
    eval_group.add_argument("--eval-batch-size", type=int, default=16)
    eval_group.add_argument("--sample-steps", type=int, default=100)
    eval_group.add_argument(
        "--posebusters-config",
        type=Path,
        default=Path("src/tabasco/utils/posebusters_no_strain.yaml"),
    )
    eval_group.add_argument(
        "--pb-rdkit-add-hydrogens",
        action="store_true",
        help=(
            "Add RDKit hydrogens with coordinates to converted molecules before "
            "PoseBusters. This is intended for heavy-only models evaluated with "
            "hydrogen-inclusive PB checks; explicit-H models should leave this off."
        ),
    )
    eval_group.add_argument(
        "--pb-remove-hydrogens",
        action="store_true",
        help=(
            "Remove hydrogens from converted molecules before PoseBusters. This "
            "is intended for explicit-H models when logging a heavy-skeleton PB "
            "metric from the same generated samples."
        ),
    )
    eval_group.add_argument(
        "--extra-posebusters-config",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Additional PoseBusters config to run on the same generated samples. "
            "Metrics are prefixed with LABEL_. May be repeated."
        ),
    )
    eval_group.add_argument(
        "--extra-pb-rdkit-add-hydrogens",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "For the matching --extra-posebusters-config LABEL, add RDKit H "
            "before PoseBusters. May be repeated."
        ),
    )
    eval_group.add_argument(
        "--extra-pb-remove-hydrogens",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "For the matching --extra-posebusters-config LABEL, strip H before "
            "PoseBusters. May be repeated."
        ),
    )
    eval_group.add_argument("--no-sanitize", action="store_true")
    eval_group.add_argument(
        "--skip-step-zero-eval",
        action="store_true",
        help="Do not evaluate PB validity before the first optimizer step.",
    )
    eval_group.add_argument(
        "--deterministic-eval",
        action="store_true",
        help=(
            "Use a deterministic per-step seed for sampling/PB evaluation and "
            "restore training RNG state afterward."
        ),
    )
    eval_group.add_argument(
        "--eval-seed-base",
        type=int,
        default=1_000_000,
        help="Base offset added to --seed and evaluation step for deterministic eval.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pb_training_curve"),
    )
    output.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit subdirectory name under --output-dir.",
    )
    output.add_argument(
        "--save-final-checkpoint",
        action="store_true",
        help="Save model/optimizer state_dicts at the end of the run.",
    )
    output.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N steps; 0 disables.",
    )
    output.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume model and optimizer state from a saved checkpoint. "
            "--max-steps is interpreted as the absolute final training step."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def temporary_seed(seed: int):
    """Temporarily set RNG seeds and restore the previous state afterward."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def embed_smiles(smiles: str, seed: int) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
    if status != 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def extract_mol(item) -> Chem.Mol | None:
    if isinstance(item, Chem.Mol):
        return item
    if isinstance(item, (tuple, list)):
        for value in reversed(item):
            if isinstance(value, Chem.Mol):
                return value
    if isinstance(item, dict):
        for key in ("molecule", "mol", "ligand"):
            value = item.get(key)
            if isinstance(value, Chem.Mol):
                return value
    return None


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def normalize_download_url(url: str) -> str:
    if "huggingface.co" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def download_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_url = normalize_download_url(url)
    parsed = urlparse(normalized_url)
    filename = Path(parsed.path).name or "dataset.pt"
    output_path = cache_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"using cached dataset: {output_path}")
        return output_path

    print(f"downloading dataset to repo cache: {output_path}")
    request = Request(normalized_url, headers={"User-Agent": "tabasco-smoke-train"})
    with urlopen(request) as response, open(output_path, "wb") as handle:
        shutil.copyfileobj(response, handle, length=16 * 1024 * 1024)
    return output_path


def resolve_data_path(args: argparse.Namespace) -> Path:
    data_path = args.data_path
    if data_path is None and args.geom_drugs_split is not None:
        data_path = GEOM_DRUGS_URLS[args.geom_drugs_split]

    if data_path is None:
        raise ValueError("Pass --data-path, --geom-drugs-split, or --tiny-smiles.")

    if is_url(data_path):
        return download_to_cache(data_path, args.cache_dir)
    return Path(data_path)


def load_processed_mols(path: Path, limit: int) -> list[Chem.Mol]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if limit > 0:
        raw = raw[:limit]

    mols = []
    for item in raw:
        mol = extract_mol(item)
        if mol is not None and mol.GetNumConformers() > 0:
            mols.append(mol)
    if not mols:
        raise ValueError(f"No RDKit molecules with conformers found in {path}.")
    return mols


def load_tiny_mols(seed: int) -> list[Chem.Mol]:
    mols = []
    for i, smiles in enumerate(DEFAULT_SMILES):
        mol = embed_smiles(smiles, seed + i)
        if mol is not None:
            mols.append(mol)
    if not mols:
        raise RuntimeError("Could not embed any built-in SMILES molecules.")
    return mols


def get_atom_names(include_hydrogens: bool) -> list[str]:
    return list(ATOM_NAMES_WITH_H if include_hydrogens else ATOM_NAMES)


def converter_from_stats(stats: dict | None = None, *, include_hydrogens: bool = False):
    atom_names = None
    if stats is not None:
        atom_names = stats.get("atom_names")
    if atom_names is None:
        atom_names = get_atom_names(include_hydrogens)
    return MoleculeConverter(atom_names=list(atom_names))


def mols_to_dataset(
    mols: Iterable[Chem.Mol], repeat: int, include_hydrogens: bool = False
) -> TensorMolDataset:
    atom_names = get_atom_names(include_hydrogens)
    converter = MoleculeConverter(atom_names=atom_names)
    processed_mols = []
    all_smiles = []
    atom_counts = []
    hydrogen_counts = []

    for mol in mols:
        try:
            mol_copy = Chem.Mol(mol)
            if not include_hydrogens:
                mol_copy = Chem.RemoveAllHs(mol_copy)

            if mol_copy.GetNumConformers() == 0:
                continue

            if include_hydrogens:
                num_h = sum(1 for atom in mol_copy.GetAtoms() if atom.GetSymbol() == "H")
                if num_h == 0:
                    continue
                hydrogen_counts.append(num_h)

            processed_mols.append(mol_copy)
            atom_counts.append(mol_copy.GetNumAtoms())
            all_smiles.append(Chem.MolToSmiles(mol_copy))
        except Exception:
            continue

    if not processed_mols:
        if include_hydrogens:
            raise ValueError("No usable molecules with explicit hydrogens.")
        raise ValueError("No usable molecules after hydrogen removal.")

    max_num_atoms = max(atom_counts)
    tensors = [
        converter.to_tensor(
            mol,
            pad_to_size=max_num_atoms,
            remove_hydrogens=not include_hydrogens,
        )
        for mol in processed_mols
    ]
    stats = {
        "num_atoms_histogram": dict(Counter(atom_counts)),
        "max_num_atoms": max_num_atoms,
        "spatial_dim": 3,
        "atom_dim": len(converter._atom_types),
        "atom_names": atom_names,
        "include_hydrogens": include_hydrogens,
        "num_hydrogens_histogram": dict(Counter(hydrogen_counts)),
        "all_smiles": all_smiles,
    }
    return TensorMolDataset(tensors=tensors, stats=stats, repeat=repeat)


def build_dataset(args: argparse.Namespace) -> TensorMolDataset:
    if args.tiny_smiles:
        mols = load_tiny_mols(args.seed)
    else:
        data_path = resolve_data_path(args)
        mols = load_processed_mols(data_path, args.limit_data)
    return mols_to_dataset(
        mols,
        repeat=args.repeat_dataset,
        include_hydrogens=args.include_hydrogens,
    )


def build_model(args: argparse.Namespace, stats: dict) -> FlowMatchingModel:
    time_factor = InverseTimeFactor(
        max_value=100.0,
        min_value=0.05,
        zero_before=0.0,
        eps=1e-6,
    )
    atom_input_mode = (
        "continuous_linear"
        if args.atomics_mode == "dfm_gaussian"
        else "discrete_embedding"
    )
    net = TransformerModule(
        spatial_dim=stats["spatial_dim"],
        atom_dim=stats["atom_dim"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        activation="SiLU",
        implementation=args.implementation,
        cross_attention=not args.no_cross_attention,
        atom_input_mode=atom_input_mode,
        max_num_atoms=stats["max_num_atoms"],
    )
    if args.atomics_mode == "original":
        atomics_interpolant = DiscreteInterpolant(
            key="atomics",
            loss_weight=0.1,
            time_factor=time_factor,
        )
    elif args.atomics_mode == "dfm_gaussian":
        atomics_interpolant = GaussianSimplexAtomInterpolant(
            key="atomics",
            loss_weight=0.1,
            time_factor=time_factor,
            beta_schedule=getattr(args, "dfm_beta_schedule", "linear"),
            beta_power=getattr(args, "dfm_beta_power", 1.0),
        )
    else:
        raise ValueError(f"Invalid atomics_mode: {args.atomics_mode!r}")

    model = FlowMatchingModel(
        net=net,
        coords_interpolant=SDEMetricInterpolant(
            key="coords",
            loss_weight=1.0,
            scale_noise_by_log_num_atoms=False,
            noise_scale=1.0,
            langevin_sampling_schedule=SampleNoiseSchedule(cutoff=0.9),
            white_noise_sampling_scale=0.01,
            time_factor=time_factor,
        ),
        atomics_interpolant=atomics_interpolant,
        time_distribution="beta",
        time_alpha_factor=1.8,
        num_random_augmentations=args.num_random_augmentations,
        sample_schedule=args.sample_schedule,
        compile=False,
    )
    model.set_data_stats(stats)
    return model


def sample_molecules(
    model: FlowMatchingModel,
    *,
    stats: dict,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    sanitize: bool,
) -> tuple[list, list[int], int]:
    converter = converter_from_stats(stats)
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
        batch_mols = converter.from_batch(generated.detach().cpu(), sanitize=sanitize)
        for local_idx, mol in enumerate(batch_mols):
            if mol is not None:
                mols.append(mol)
                mol_indices.append(total_generated + local_idx)
        total_generated += current_batch_size
    return mols, mol_indices, total_generated


def clean_metric_label(label: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", label.strip())
    label = re.sub(r"_+", "_", label).strip("_").lower()
    if not label:
        raise ValueError("PoseBusters metric label cannot be empty.")
    if label[0].isdigit():
        label = f"pb_{label}"
    return label


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Expected --extra-posebusters-config as LABEL=PATH, got {spec!r}."
        )
    label, path = spec.split("=", 1)
    return clean_metric_label(label), Path(path)


def prepare_pb_molecules(
    mols: list,
    mol_indices: list[int],
    *,
    pb_remove_hydrogens: bool = False,
    pb_rdkit_add_hydrogens: bool = False,
) -> tuple[list, list[int], int, int]:
    pb_mols = []
    pb_indices = []
    remove_h_failures = 0
    add_h_failures = 0

    for mol, mol_index in zip(mols, mol_indices):
        try:
            pb_mol = Chem.Mol(mol)
            if pb_remove_hydrogens:
                pb_mol = Chem.RemoveAllHs(pb_mol)
        except Exception:
            remove_h_failures += 1
            continue

        if pb_rdkit_add_hydrogens:
            try:
                pb_mol = Chem.AddHs(pb_mol, addCoords=True)
            except Exception:
                add_h_failures += 1
                continue
            if pb_mol is None:
                add_h_failures += 1
                continue

        if pb_mol is None or pb_mol.GetNumAtoms() == 0:
            remove_h_failures += 1
            continue

        pb_mols.append(pb_mol)
        pb_indices.append(mol_index)

    return pb_mols, pb_indices, remove_h_failures, add_h_failures


def compute_pb_summary(
    posebusters: PoseBusters,
    mols: list,
    mol_indices: list[int],
    total_generated: int,
    train_smiles_set: set[str] | None = None,
    pb_rdkit_add_hydrogens: bool = False,
    pb_remove_hydrogens: bool = False,
) -> dict:
    smiles = []
    for mol in mols:
        try:
            smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            continue

    unique_smiles = set(smiles)
    uniqueness_denominator = max(1, len(smiles))
    novelty_denominator = max(1, len(unique_smiles))

    diversity_summary = {
        "num_smiles": len(smiles),
        "num_unique_smiles": len(unique_smiles),
        "unique_smiles_rate": len(unique_smiles) / uniqueness_denominator,
        "unique_smiles_rate_total": len(unique_smiles) / max(1, total_generated),
    }
    if train_smiles_set is not None:
        novel_smiles = unique_smiles - train_smiles_set
        diversity_summary.update(
            {
                "num_novel_smiles": len(novel_smiles),
                "novel_smiles_rate": len(novel_smiles) / novelty_denominator,
                "novel_smiles_rate_total": len(novel_smiles)
                / max(1, total_generated),
            }
        )

    if not mols:
        return {
            "num_generated": total_generated,
            "num_converted": 0,
            "conversion_rate": 0.0,
            "num_pb_input_mols": 0,
            "num_pb_remove_h_failures": 0,
            "num_pb_add_h_failures": 0,
            "pb_valid": 0.0,
            "pb_intersection": 0.0,
            **diversity_summary,
        }

    pb_mols, pb_indices, remove_h_failures, add_h_failures = prepare_pb_molecules(
        mols,
        mol_indices,
        pb_remove_hydrogens=pb_remove_hydrogens,
        pb_rdkit_add_hydrogens=pb_rdkit_add_hydrogens,
    )

    if not pb_mols:
        return {
            "num_generated": total_generated,
            "num_converted": len(mols),
            "conversion_rate": len(mols) / total_generated,
            "num_pb_input_mols": 0,
            "num_pb_remove_h_failures": remove_h_failures,
            "num_pb_add_h_failures": add_h_failures,
            "pb_valid": 0.0,
            "pb_intersection": 0.0,
            **diversity_summary,
        }

    try:
        results = posebusters.bust(mol_pred=pb_mols)
    except RuntimeError:
        return {
            "num_generated": total_generated,
            "num_converted": len(mols),
            "conversion_rate": len(mols) / total_generated,
            "num_pb_input_mols": len(pb_mols),
            "num_pb_remove_h_failures": remove_h_failures,
            "num_pb_add_h_failures": add_h_failures,
            "pb_valid": 0.0,
            "pb_intersection": 0.0,
            **diversity_summary,
        }
    results.insert(0, "sample_idx", pb_indices)
    check_columns = [column for column in results.columns if column != "sample_idx"]
    passes_all = ~results[check_columns].isin([False]).any(axis=1)

    summary = {
        "num_generated": total_generated,
        "num_converted": len(mols),
        "conversion_rate": len(mols) / total_generated,
        "num_pb_input_mols": len(pb_mols),
        "num_pb_remove_h_failures": remove_h_failures,
        "num_pb_add_h_failures": add_h_failures,
        "pb_valid": passes_all.sum() / total_generated,
        "pb_intersection": passes_all.sum() / total_generated,
        **diversity_summary,
    }
    for column in check_columns:
        summary[f"pb_{column}"] = results[column].sum() / total_generated
    return summary


def prefix_summary(summary: dict, label: str) -> dict:
    return {f"{label}_{key}": value for key, value in summary.items()}


def build_posebusters_evaluators(args: argparse.Namespace) -> list[dict]:
    install_posebusters_compat()

    evaluators = []
    with open(args.posebusters_config, encoding="utf-8") as handle:
        pb_config = yaml.safe_load(handle)
    evaluators.append(
        {
            "label": "primary",
            "prefix": None,
            "config_path": str(args.posebusters_config),
            "posebusters": PoseBusters(config=pb_config),
            "rdkit_add_hydrogens": args.pb_rdkit_add_hydrogens,
            "remove_hydrogens": args.pb_remove_hydrogens,
        }
    )

    add_h_labels = {
        clean_metric_label(label)
        for label in (args.extra_pb_rdkit_add_hydrogens or [])
    }
    remove_h_labels = {
        clean_metric_label(label) for label in (args.extra_pb_remove_hydrogens or [])
    }
    seen_labels = set()
    for spec in args.extra_posebusters_config:
        label, path = parse_labeled_path(spec)
        if label in seen_labels:
            raise ValueError(f"Duplicate extra PoseBusters label: {label}")
        seen_labels.add(label)
        with open(path, encoding="utf-8") as handle:
            extra_config = yaml.safe_load(handle)
        evaluators.append(
            {
                "label": label,
                "prefix": label,
                "config_path": str(path),
                "posebusters": PoseBusters(config=extra_config),
                "rdkit_add_hydrogens": label in add_h_labels,
                "remove_hydrogens": label in remove_h_labels,
            }
        )

    unused_add = add_h_labels - seen_labels
    unused_remove = remove_h_labels - seen_labels
    if unused_add:
        raise ValueError(f"Unknown --extra-pb-rdkit-add-hydrogens labels: {unused_add}")
    if unused_remove:
        raise ValueError(f"Unknown --extra-pb-remove-hydrogens labels: {unused_remove}")
    return evaluators


def evaluate(
    model: FlowMatchingModel,
    evaluators: list[dict],
    args: argparse.Namespace,
    step: int | None = None,
    train_smiles_set: set[str] | None = None,
) -> dict:
    if args.deterministic_eval and step is not None:
        seed = args.seed + args.eval_seed_base + step
        with temporary_seed(seed):
            return evaluate(
                model=model,
                evaluators=evaluators,
                args=args,
                step=None,
                train_smiles_set=train_smiles_set,
            )

    mols, mol_indices, total_generated = sample_molecules(
        model,
        stats=model.data_stats,
        num_samples=args.eval_samples,
        batch_size=args.eval_batch_size,
        num_steps=args.sample_steps,
        sanitize=not args.no_sanitize,
    )
    combined_summary = {}
    for evaluator in evaluators:
        summary = compute_pb_summary(
            posebusters=evaluator["posebusters"],
            mols=mols,
            mol_indices=mol_indices,
            total_generated=total_generated,
            train_smiles_set=train_smiles_set,
            pb_rdkit_add_hydrogens=evaluator["rdkit_add_hydrogens"],
            pb_remove_hydrogens=evaluator["remove_hydrogens"],
        )
        if evaluator["prefix"] is None:
            combined_summary.update(summary)
        else:
            combined_summary.update(prefix_summary(summary, evaluator["prefix"]))
    return combined_summary


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_eval_log(step: int, train_loss: float | None, summary: dict) -> str:
    """Return one compact log line with all PB rates from an evaluation."""
    fields = [f"step {step}"]
    if train_loss is not None:
        fields.append(f"loss={train_loss:.4f}")
    fields.extend(
        [
            f"conversion={summary.get('conversion_rate', 0.0):.4f}",
            f"pb_valid={summary.get('pb_valid', 0.0):.4f}",
        ]
    )
    for key in sorted(summary):
        if key in {"pb_valid", "pb_intersection"}:
            continue
        if key.startswith("pb_") or "_pb_" in key:
            fields.append(f"{key}={summary[key]:.4f}")
    return " ".join(fields)


def make_output_dir(root: Path, run_name: str | None = None) -> Path:
    run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(
    output_dir: Path,
    model: FlowMatchingModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    data_stats: dict,
    step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "data_stats": data_stats,
            "step": step,
        },
        output_dir / f"model_step_{step}.pt",
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def require_resume_match(
    *,
    name: str,
    checkpoint_value,
    current_value,
) -> None:
    if checkpoint_value != current_value:
        raise ValueError(
            "Resume checkpoint is incompatible: "
            f"{name} checkpoint={checkpoint_value!r} current={current_value!r}"
        )


def load_training_checkpoint(
    checkpoint_path: Path,
    model: FlowMatchingModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stats: dict,
    device: torch.device,
) -> int:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_stats = checkpoint.get("data_stats", {})
    for key in (
        "max_num_atoms",
        "spatial_dim",
        "atom_dim",
        "atom_names",
        "include_hydrogens",
    ):
        if key in checkpoint_stats:
            require_resume_match(
                name=f"data_stats.{key}",
                checkpoint_value=checkpoint_stats[key],
                current_value=stats.get(key),
            )

    checkpoint_args = checkpoint.get("args", {})
    for key in (
        "atomics_mode",
        "dfm_beta_schedule",
        "dfm_beta_power",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "implementation",
        "no_cross_attention",
        "num_random_augmentations",
        "include_hydrogens",
    ):
        if key in checkpoint_args:
            require_resume_match(
                name=f"args.{key}",
                checkpoint_value=checkpoint_args[key],
                current_value=getattr(args, key),
            )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state_to_device(optimizer, device)
    return int(checkpoint.get("step", 0))


def next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    output_dir = make_output_dir(args.output_dir, args.run_name)
    metrics_path = output_dir / "pb_training_curve.csv"
    losses_path = output_dir / "losses.csv"

    dataset = build_dataset(args)
    train_smiles_set = set(dataset.stats.get("all_smiles", []))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TensorDictCollator(),
        drop_last=False,
    )
    loader_iter = iter(loader)

    model = build_model(args, dataset.stats).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.995),
    )

    evaluators = build_posebusters_evaluators(args)
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = load_training_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            args=args,
            stats=dataset.stats,
            device=device,
        )
        if start_step >= args.max_steps:
            raise ValueError(
                "--max-steps must be greater than the resumed checkpoint step: "
                f"max_steps={args.max_steps} resume_step={start_step}"
            )

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "num_train_molecules": len(dataset.tensors),
            "data_stats": dataset.stats,
            "resume_start_step": start_step,
        }
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str, sort_keys=True)

    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(
        "dataset: "
        f"{len(dataset.tensors)} molecules, max_atoms={dataset.stats['max_num_atoms']}"
    )
    print(f"atomics_mode: {args.atomics_mode}")
    if args.resume_checkpoint is not None:
        print(f"resume_checkpoint: {args.resume_checkpoint}")
        print(f"resume_start_step: {start_step}")
        print(f"resume_target_step: {args.max_steps}")
    print(f"deterministic_eval: {args.deterministic_eval}")
    print(f"pb_rdkit_add_hydrogens: {args.pb_rdkit_add_hydrogens}")
    print(f"pb_remove_hydrogens: {args.pb_remove_hydrogens}")
    for evaluator in evaluators:
        prefix = evaluator["prefix"] or "primary"
        print(
            "posebusters_eval: "
            f"{prefix} config={evaluator['config_path']} "
            f"remove_hydrogens={evaluator['remove_hydrogens']} "
            f"rdkit_add_hydrogens={evaluator['rdkit_add_hydrogens']}"
        )

    if not args.skip_step_zero_eval:
        summary = evaluate(
            model,
            evaluators,
            args,
            step=start_step,
            train_smiles_set=train_smiles_set,
        )
        row = {"step": start_step, "train_loss": float("nan"), **summary}
        append_csv_row(metrics_path, row)
        print(format_eval_log(step=start_step, train_loss=None, summary=summary))

    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        batch, loader_iter = next_batch(loader_iter, loader)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch, compute_stats=False)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        append_csv_row(losses_path, {"step": step, "train_loss": loss_value})

        if step % args.eval_every == 0 or step == args.max_steps:
            summary = evaluate(
                model,
                evaluators,
                args,
                step=step,
                train_smiles_set=train_smiles_set,
            )
            row = {"step": step, "train_loss": loss_value, **summary}
            append_csv_row(metrics_path, row)
            print(format_eval_log(step=step, train_loss=loss_value, summary=summary))

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(output_dir, model, optimizer, args, dataset.stats, step)

    if args.save_final_checkpoint:
        save_checkpoint(
            output_dir, model, optimizer, args, dataset.stats, args.max_steps
        )

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        print("\nPB training curve")
        print(df.to_string(index=False))
    print(f"metrics_csv: {metrics_path}")
    print(f"losses_csv: {losses_path}")


if __name__ == "__main__":
    main()
