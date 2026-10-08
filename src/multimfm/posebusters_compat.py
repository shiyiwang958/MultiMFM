"""Compatibility shims for PoseBusters configs used by related repos."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts
from rdkit.Chem.rdmolops import SanitizeMol


def check_flatness_compat(
    mol_pred: Mol,
    threshold_flatness: float = 0.1,
    flat_systems: dict[str, str] | None = None,
    check_nonflat: bool = False,
) -> dict[str, Any]:
    """PoseBusters flatness check with FlowMol's ``check_nonflat`` option.

    ``posebusters==0.3.1`` does not accept ``check_nonflat``. FlowMol's
    ``pb_config.yaml`` uses it for non-aromatic ring non-flatness, where a
    substructure passes when its maximum distance from the best-fit plane is at
    least ``threshold_flatness``.
    """
    if flat_systems is None:
        flat_systems = {
            "aromatic_5_membered_rings_sp2": "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1",
            "aromatic_6_membered_rings_sp2": "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1",
            "trigonal_planar_double_bonds": "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)",
        }

    empty = {
        "results": {
            "num_systems_checked": np.nan,
            "num_systems_passed": np.nan,
            "max_distance": np.nan,
            "flatness_passes": np.nan,
        }
    }

    mol = deepcopy(mol_pred)
    try:
        assert mol_pred.GetNumConformers() > 0, "Molecule does not have a conformer."
        SanitizeMol(mol)
    except Exception:
        return empty

    planar_groups = []
    types = []
    for flat_system, smarts in flat_systems.items():
        match = MolFromSmarts(smarts)
        atom_groups = list(mol.GetSubstructMatches(match))
        planar_groups += atom_groups
        types += [flat_system] * len(atom_groups)

    coords = [
        np.array([mol.GetConformer().GetAtomPosition(i) for i in group])
        for group in planar_groups
    ]
    max_distances = [_max_distance_to_plane(coord) for coord in coords]
    if check_nonflat:
        flatness_passes = [bool(distance >= threshold_flatness) for distance in max_distances]
    else:
        flatness_passes = [bool(distance <= threshold_flatness) for distance in max_distances]

    return {
        "results": {
            "num_systems_checked": len(planar_groups),
            "num_systems_passed": sum(flatness_passes),
            "max_distance": max(max_distances) if max_distances else np.nan,
            "flatness_passes": all(flatness_passes) if flatness_passes else True,
        },
        "details": {
            "type": types,
            "planar_group": planar_groups,
            "max_distance": max_distances,
            "flatness_passes": flatness_passes,
        },
    }


def _max_distance_to_plane(coords: np.ndarray) -> float:
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    return float(np.abs(np.dot(centered, normal)).max())


def install_posebusters_compat() -> None:
    """Patch PoseBusters' module registry for FlowMol config compatibility."""
    import posebusters.posebusters as pb_module

    pb_module.module_dict["flatness"] = check_flatness_compat
