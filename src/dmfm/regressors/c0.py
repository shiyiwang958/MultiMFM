"""PyTorch regressors for 50 bp intrinsic cyclizability (C0).

Ported verbatim from ``dirichlet-flow-matching/dinko/c0_regressors.py`` (July
2026; not part of DNA-MFM@187fe7b). ``park_cnn`` is the Park et al. C0 CNN
architecture used for both the guide and the oracle of the paper
(``checkpoints/dna/c0/{guide,oracle}/best_state.pt``, trained on disjoint parent
splits by ``dmfm.experiments.train_c0_regressor``). The older Keras-converted
Park model (``dinko/C0free_torch.pt`` + ``dinko/torch_model.py``, purged) is not
ported: it was the guide of the split65k-era runs only (Fig 4 right, Tables 4-6 as
printed); every parent-disjoint experiment uses the ``park_cnn`` weights above.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ParkC0Regressor(nn.Module):
    """Park-compatible C0 CNN accepting A/C/G/T probabilities [B, 50, 4]."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=28, stride=4)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=33, stride=1)
        self.fc1 = nn.Linear(384, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x_acgt: torch.Tensor) -> torch.Tensor:
        if x_acgt.ndim != 3 or x_acgt.shape[1:] != (50, 4):
            raise ValueError(f"Expected [B, 50, 4] A/C/G/T probabilities, got {tuple(x_acgt.shape)}")
        # The published Park encoding orders bases A/T/G/C within each position.
        x_atgc = x_acgt[..., [0, 3, 2, 1]].reshape(x_acgt.shape[0], 200, 1)
        x = x_atgc.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2).flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class IndependentC0Oracle(nn.Module):
    """Independent C0 evaluator with a distinct convolutional architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_acgt: torch.Tensor) -> torch.Tensor:
        if x_acgt.ndim != 3 or x_acgt.shape[1:] != (50, 4):
            raise ValueError(f"Expected [B, 50, 4] A/C/G/T probabilities, got {tuple(x_acgt.shape)}")
        return self.head(self.features(x_acgt.transpose(1, 2))).squeeze(-1)


MODEL_TYPES = {
    "park_cnn": ParkC0Regressor,
    "independent_oracle": IndependentC0Oracle,
}


def build_c0_regressor(model_type: str) -> nn.Module:
    try:
        return MODEL_TYPES[model_type]()
    except KeyError as exc:
        raise ValueError(f"Unknown C0 regressor {model_type!r}; choose from {sorted(MODEL_TYPES)}") from exc


def load_c0_regressor(path, model_type: str = "park_cnn", device="cpu") -> nn.Module:
    """Build a C0 regressor and load a ``best_state.pt`` state dict (eval mode, frozen)."""
    from dmfm.utils.torch_io import torch_load

    model = build_c0_regressor(model_type)
    state = torch_load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
