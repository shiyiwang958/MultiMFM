"""Port side of the forward-pass parity check (multimfm env): python parity_port.py out.pt"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from dmfm.experiments import ablate_glass_gradient_mc as G, ablate_dmfm_one_step_gradient_mc as D, sample_c0_guidance as H
from dmfm.regressors.c0 import build_c0_regressor
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step
import parity_common
out = parity_common.run(G, D, H, build_c0_regressor, gaussian_denoiser_flow_step)
torch.save(out, sys.argv[1]); print("port", len(out), torch.__version__)
