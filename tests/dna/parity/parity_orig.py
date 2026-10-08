"""Original-code side of the forward-pass parity check. Run with the seq_h100 env:
    ORIG=<tree from build_orig_tree.sh> PYTHONPATH=$ORIG:<recon>/pydeps ~/micromamba/envs/seq_h100/bin/python parity_orig.py out.pt"""
import os, sys
from pathlib import Path
O = os.environ["ORIG"]
sys.path[:0] = [O, O + "/scripts", str(Path(__file__).resolve().parent)]
import torch
import ablate_glass_gradient_mc as G
import ablate_dmfm_one_step_gradient_mc as D
import sample_yeast_c0_guidance_hist as H
from dinko.c0_regressors import build_c0_regressor
from utils.flow_utils import gaussian_denoiser_flow_step
import parity_common
out = parity_common.run(G, D, H, build_c0_regressor, gaussian_denoiser_flow_step)
torch.save(out, sys.argv[1]); print("orig", len(out), torch.__version__)
