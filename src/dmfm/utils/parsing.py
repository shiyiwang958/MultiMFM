"""Command-line arguments of the dMFM Lightning trainer (``dmfm.experiments.train_dmfm``).

Ported from the July 2026 ``dirichlet-flow-matching/utils/parsing.py`` (newer
than DNA-MFM@187fe7b: adds ``--init_ckpt``, free-form split names,
``--mfm_checkpoint_metric`` and writes ``args.json``). All flags and defaults
that affect training are unchanged, so the saved ``args.json`` of the paper
runs replay exactly. Changes:

- ``--yeast_data_pt`` default is ``data/dna/yeast_parent_disjoint/yeast_parent_L50.pt``
  (the old default ``data/yeast_mid50.pt`` was purged pre-split data).
- ``MODEL_DIR`` defaults to ``outputs/dna/<run_name>_<timestamp>`` instead of ``workdir/``.
- ``--wandb`` defaults to off (every paper run used ``--no-wandb``); the
  personal W&B entity/project defaults were removed (set ``WANDB_ENTITY`` /
  ``WANDB_PROJECT`` yourself).
- ``git rev-parse HEAD`` failures no longer abort (``args.commit = "unknown"``).
"""

import argparse
import json
import os
import subprocess
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime

from dmfm import paths


def build_train_parser() -> ArgumentParser:
    parser = ArgumentParser()
    
    # Run settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help=(
            "Optional Lightning checkpoint used to initialize model weights only. "
            "Unlike --ckpt, this does not restore optimizer, scheduler, or global step."
        ),
    )
    parser.add_argument("--cls_ckpt", type=str, default=None)
    parser.add_argument("--cls_ckpt_hparams", type=str, default=None)
    parser.add_argument("--clean_cls_ckpt", type=str, default=None, help='cls model for evaluation purposes')
    parser.add_argument("--clean_cls_ckpt_hparams", type=str, default=None)
    parser.add_argument("--distill_ckpt", type=str, default=None, help='cls model for evaluation purposes')
    parser.add_argument("--distill_ckpt_hparams", type=str, default=None)
    parser.add_argument("--ckpt_has_cls", action='store_true')
    parser.add_argument("--ckpt_has_clean_cls", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--subset_train_as_val", action='store_true')
    parser.add_argument("--validate_on_train", action='store_true')
    parser.add_argument("--validate_on_test", action='store_true')

    # Training
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--constant_val_len", type=int, default=None)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)
    parser.add_argument("--check_grad", action="store_true")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--checkpoint_layers", action="store_true")
    parser.add_argument("--max_steps", type=int, default=450000)
    parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu", "auto"],
        default="gpu",
        help="Lightning accelerator. Use cpu for local smoke tests; Slurm training uses gpu.",
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # Validate
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--fid_early_stop", action="store_true")
    parser.add_argument("--val_loss_es", action="store_true", help='only for cls train')
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--ckpt_iterations", type=int, nargs='+', default=None)
    parser.add_argument("--random_sequences", action="store_true")
    parser.add_argument("--taskiran_seq_path", type=str, default=None)

    # Data
    parser.add_argument("--dataset_type", type=str, choices=["yeast"], default="yeast")
    parser.add_argument("--yeast_data_pt", type=str, default=str(paths.data_pt(50)))
    parser.add_argument("--yeast_split_pt", type=str, default=None)
    parser.add_argument(
        "--yeast_train_split",
        type=str,
        default="train",
        help="Named training split in --yeast_split_pt (for example train or a_train).",
    )
    parser.add_argument(
        "--yeast_val_split",
        type=str,
        default="val",
        help="Named validation split in --yeast_split_pt (for example val or a_val).",
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # Guidance
    parser.add_argument("--cls_guidance", action='store_true')
    parser.add_argument("--binary_guidance", action='store_true', help='the model is trained with only the target class and the auxiliary class')
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--all_class_inference", action='store_true', help='ignores target_class and guides towards all classes during inference. Helfpul for seeing if we improve the general FID.')
    parser.add_argument("--cls_free_noclass_ratio", type=float, default=0.3)
    parser.add_argument("--cls_free_guidance", action='store_true')
    parser.add_argument("--probability_addition", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--adaptive_prob_add", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--vectorfield_addition", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--probability_tilt", action='store_true', help='if this is activated then cls_free_guidance also needs to be activated and we then do it with probs tilting instead of with score conversion')
    parser.add_argument("--score_free_guidance", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--analytic_cls_score", action='store_true', help='unsupported legacy toy-experiment flag')
    parser.add_argument("--scale_cls_score", action='store_true')
    parser.add_argument("--allow_nan_cfactor", action="store_true")

    # Model
    parser.add_argument("--model", choices=["dit"], default="dit")
    parser.add_argument("--cls_model", choices=['mlp','cnn','transformer', 'deepflybrain'], default='cnn')
    parser.add_argument("--clean_cls_model", choices=['mlp', 'cnn', 'transformer', 'deepflybrain'], default='cnn')
    parser.add_argument("--clean_data", action="store_true", help='do not noise to the model input. E.g. for training a clean calssifier.')
    parser.add_argument("--mode", choices=["gaussian"], default="gaussian")
    parser.add_argument("--simplex_spacing", type=int, default=1000, help='deprecated, has no influence')
    parser.add_argument("--prior_pseudocount", type=float, default=2, help='hyperparameter for expand_simplex function. Can be kept at 2 and should not matter much.')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_cnn_stacks", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # Transformer-specific knobs (used when --model transformer)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_ff_mult", type=int, default=4, help="FFN dim = ff_mult * hidden_dim")
    # DiT-specific: number of blocks (depth). Uses hidden_dim + transformer_heads + transformer_ff_mult.
    parser.add_argument("--dit_blocks", type=int, default=12)
    parser.add_argument("--self_condition_ratio", type=float, default=0)
    parser.add_argument("--prior_self_condition", action="store_true")
    parser.add_argument("--no_token_dropout", action="store_true")
    parser.add_argument("--time_embed", action="store_true")
    parser.add_argument("--fix_alpha", type=float, default=None)
    parser.add_argument("--alpha_scale", type=float, default=2, help='controls the alphas at which training is performed by scaling them. If this is higher, then higher alphas are sampled for training. Recall that alphas correspond to diffusion time in dirichlet flow matching. alpha=1 corresponds to full noise, and alpha_max, such as alpha_max = 8 corresponds to clean data.')
    parser.add_argument("--alpha_max", type=float, default=8, help='controls the maximum value until which we run the inference process. In equation 14, we write our probability path to go from alpha=1 to alpha=infinity. In practice we cut off alpha at alpha_max.')
    parser.add_argument("--cls_expanded_simplex", action="store_true")
    parser.add_argument("--simplex_encoding_dim", type=int, default=64)
    parser.add_argument("--flow_temp", type=float, default=1.0)
    parser.add_argument('--val_pred_type', type=str, choices=['argmax', 'sample'], default='argmax')
    parser.add_argument('--num_integration_steps', type=int, default=100, help='The number of integration steps used during inference.')

    # Gaussian mode: time sampling (t in [0,1]).
    # By default this is uniform (Beta(1,1)). You can bias toward cleaner inputs with a> b.
    parser.add_argument("--gaussian_t_beta_a_start", type=float, default=1.0)
    parser.add_argument("--gaussian_t_beta_b_start", type=float, default=1.0)
    parser.add_argument("--gaussian_t_beta_a_end", type=float, default=1.0)
    parser.add_argument("--gaussian_t_beta_b_end", type=float, default=1.0)
    parser.add_argument(
        "--gaussian_t_beta_anneal_steps",
        type=int,
        default=0,
        help="If >0, linearly anneal (a,b) from *_start to *_end over this many optimizer steps.",
    )
    # Gaussian mode: optional beta(t) schedule for interpolant x_t = beta(t) x1 + (1-beta(t)) x0.
    parser.add_argument(
        "--gaussian_beta_schedule",
        type=str,
        choices=["linear", "table"],
        default="linear",
        help="How to map t->beta(t) for gaussian interpolant. 'linear' uses beta=t. "
        "'table' uses a precomputed monotone lookup table.",
    )
    parser.add_argument(
        "--gaussian_beta_table_path",
        type=str,
        default=None,
        help="Path to torch-saved dict containing 't' and 'beta' 1D tensors.",
    )
    parser.add_argument(
        "--gaussian_adaptive_loss",
        action="store_true",
        help=(
            "Gaussian-only legacy alias for DFM p-adaptive VFM weighting. "
            "Uses per-token w_t = stopgrad((||Delta||^2 + c)^(-r))."
        ),
    )
    parser.add_argument("--gaussian_adaptive_loss_c", type=float, default=0.01)
    parser.add_argument("--gaussian_adaptive_loss_r", type=float, default=0.5)
    parser.add_argument(
        "--dfm_vfm_loss",
        action=BooleanOptionalAction,
        default=True,
        help="Gaussian DiT path: use DFM's exact loss_vfm construction (default: enabled).",
    )
    parser.add_argument(
        "--p_adaptive_loss_weighting",
        action="store_true",
        help="Mirror DFM p_adaptive_loss_weighting for VFM loss. Uses --grad_norm_p/c.",
    )
    parser.add_argument("--grad_norm_p", type=float, default=0.0)
    parser.add_argument("--grad_norm_c", type=float, default=1e-8)
    parser.add_argument(
        "--learnable_loss_weighting",
        action="store_true",
        help="Mirror DFM learned loss weighting. Requires a model.loss_weighting module.",
    )

    # META flow map (MFM-style) diagonal GLASS distillation for gaussian mode.
    # When enabled, we train a conditional velocity model v(s, x_s | t_cond, x_tcond)
    # by distilling a teacher checkpoint. This is internal conditioning (no rewards yet).
    parser.add_argument(
        "--gaussian_mfm_diag_distill",
        action="store_true",
        help="Gaussian-only: train META flow map by diagonal GLASS distillation (requires --mfm_teacher_ckpt).",
    )
    parser.add_argument(
        "--mfm_teacher_ckpt",
        type=str,
        default=None,
        help="Path to teacher Lightning checkpoint (.ckpt) used for META flow map distillation.",
    )
    parser.add_argument(
        "--mfm_teacher_ckpt_hparams",
        type=str,
        default=None,
        help="Optional path to teacher hparams.yaml. If omitted, will try to find it next to the checkpoint under lightning_logs/.",
    )

    # MFM diagonal training: t_cond schedule (ported from MFM repo defaults).
    parser.add_argument(
        "--mfm_t_cond_warmup_steps",
        type=int,
        default=2500000,
        help="For the first N steps, set t_cond=0 (learn unconditional/diagonal first).",
    )
    parser.add_argument(
        "--mfm_t_cond_power",
        type=float,
        default=1.0,
        help="After warmup, sample t_cond as (U^power) to bias toward smaller t_cond when power>1.",
    )
    parser.add_argument(
        "--mfm_t_cond_0_rate",
        type=float,
        default=0.0,
        help="After warmup, with this probability force t_cond=0 (drop conditioning).",
    )
    # MFM consistency (off-diagonal) schedule, ported from MFM configs.
    parser.add_argument(
        "--mfm_consistency",
        action="store_true",
        help="If set, after diagonal warmup train off-diagonal consistency with annealed time gaps.",
    )
    parser.add_argument(
        "--mfm_consistency_type",
        type=str,
        default="lsd",
        choices=["lsd", "esd", "esd_teacher"],
        help="Off-diagonal MFM consistency objective: pure LSD/JVP or upstream-style teacher-anchored ESD.",
    )
    parser.add_argument(
        "--mfm_consistency_warmup_steps",
        type=int,
        default=2500000,
        help="Number of optimizer steps to run diagonal GLASS warmup before switching to consistency.",
    )
    parser.add_argument(
        "--mfm_consistency_anneal_end_step",
        type=int,
        default=25000,
        help=(
            "Linearly increase time-gap size until this absolute step. "
            "If <= warmup, treat as an anneal duration after warmup."
        ),
    )
    parser.add_argument(
        "--mfm_consistency_step_offset",
        type=int,
        default=0,
        help="Offset applied to global_step for the (s,u) gap schedule when resuming; "
        "effective_step = max(0, global_step - offset). Use to restart gap annealing after resume.",
    )
    parser.add_argument(
        "--mfm_consistency_gap_mode",
        type=str,
        default="random",
        choices=["random", "fixed", "long", "endpoint"],
        help=(
            "How to sample off-diagonal consistency times after warmup. "
            "'random' keeps the upstream sorted-uniform schedule; 'fixed' uses a fixed delta; "
            "'long' samples delta uniformly from [min_gap,max_gap]; 'endpoint' uses s=0,u=1."
        ),
    )
    parser.add_argument(
        "--mfm_consistency_fixed_gap",
        type=float,
        default=0.75,
        help="Gap delta used when --mfm_consistency_gap_mode=fixed.",
    )
    parser.add_argument(
        "--mfm_consistency_min_gap",
        type=float,
        default=0.75,
        help="Minimum gap delta used when --mfm_consistency_gap_mode=long.",
    )
    parser.add_argument(
        "--mfm_consistency_max_gap",
        type=float,
        default=1.0,
        help="Maximum gap delta used when --mfm_consistency_gap_mode=long.",
    )
    parser.add_argument(
        "--mfm_legacy_gap_or_default",
        action="store_true",
        help=(
            "Reproduce the DNA-MFM@187fe7b gap sampling for gap_mode fixed/long, where a "
            "gap bound of 0 was silently replaced by its default (0.75/1.0). Off by default "
            "(the July fix used by the 4-step students)."
        ),
    )
    parser.add_argument(
        "--mfm_val_gaps",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.5],
        help="Validation: fixed time gaps (delta) to report consistency loss for.",
    )
    parser.add_argument(
        "--mfm_fixed_probe_batch_size",
        type=int,
        default=128,
        help="Validation: number of examples from the first val batch used for cached fixed MFM probes.",
    )
    parser.add_argument(
        "--mfm_fixed_probe_seed",
        type=int,
        default=12345,
        help="Validation: RNG seed used to build cached fixed MFM probe targets.",
    )
    parser.add_argument(
        "--mfm_checkpoint_metric",
        type=str,
        default=None,
        help=(
            "Optional validation metric for dMFM checkpoint selection. "
            "Diagonal-only runs should use val_mfm_probe_diag_mse."
        ),
    )
    parser.add_argument(
        "--mfm_encoder_depth",
        type=int,
        default=None,
        help="META DiT: number of early blocks using first-stage conditioning. Default: half of dit_blocks.",
    )
    parser.add_argument(
        "--mfm_preserve_t_cond_0",
        action=BooleanOptionalAction,
        default=True,
        help="META DiT: preserve behavior that t_cond=0 produces zero x_cond modulation (MFM-style).",
    )

    # MFM loss mixing (match upstream: diagonal GLASS always, off-diagonal consistency added after warmup).
    parser.add_argument(
        "--mfm_diag_loss_weight",
        type=float,
        default=1.0,
        help="Weight for diagonal GLASS (distilled FM) loss term.",
    )
    parser.add_argument(
        "--mfm_lsd_loss_weight",
        type=float,
        default=1.0,
        help="Weight for off-diagonal consistency loss term (LSD or ESD, applied after warmup).",
    )

    parser.add_argument(
        "--mfm_diag_atom_loss",
        choices=["velocity", "ce"],
        default="velocity",
        help=(
            "Diagonal distillation target. 'velocity' (default) regresses the GLASS "
            "posterior velocity with the adaptive loss. 'ce' instead matches the meta "
            "denoiser Psi_{s,s} to the teacher's psi_{t*}(S) with soft-label cross "
            "entropy, the discrete analogue of Prop. 1 (and of --atom-loss-type ce in "
            "the QM9 student). Equivalent optimum, different error measure: CE acts on "
            "the logits, so it does not vanish when softmax saturates."
        ),
    )
    # Match upstream MFM: use different adaptive-loss p for off-diagonal distillation.
    parser.add_argument(
        "--mfm_diag_adaptive_p",
        type=float,
        default=0.5,
        help="Adaptive-loss exponent p for diagonal GLASS (distill FM). Upstream default: 0.5.",
    )
    parser.add_argument(
        "--mfm_lsd_adaptive_p",
        type=float,
        default=1.0,
        help="Adaptive-loss exponent p for off-diagonal consistency. Upstream default: 1.0.",
    )

    # Optimizer / LR schedule (match upstream MFM: optional warmup then constant LR).
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "radam"],
        help="Optimizer to use. Upstream MFM defaults to RAdam.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (if supported).",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="If >0, linearly warm up LR per step then keep constant (MFM-style).",
    )
    parser.add_argument(
        "--lr_warmup_start_factor",
        type=float,
        default=0.1,
        help="Warmup start factor for LinearLR (start_factor*lr -> lr).",
    )

    # Logging
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--print_freq", type=int, default=100)
    # The paper runs all used --no-wandb; W&B is opt-in here.
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_name", type=str, default="default")
    return parser


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(paths.REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def parse_train_args(argv=None, *, tee_stdout: bool = True):
    """Parse trainer flags, set ``MODEL_DIR``, tee stdout to ``log.log`` and write ``args.json``."""
    parser = build_train_parser()
    args = parser.parse_args(argv)
    timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["MODEL_DIR"] = args.output_dir or str(paths.OUTPUTS / (args.run_name + "_" + timestamp))
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))

    from dmfm.utils.log_utils import Logger
    os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
    if tee_stdout:
        sys.stdout = Logger(logpath=os.path.join(os.environ["MODEL_DIR"], "log.log"), syspart=sys.stdout)
        sys.stderr = Logger(logpath=os.path.join(os.environ["MODEL_DIR"], "log.log"), syspart=sys.stderr)
    args.commit = _git_commit()
    # Keep a lightweight, human-readable run manifest next to Lightning
    # checkpoints. This mirrors the standalone DFM trainer and lets later
    # sampling/probing reconstruct the exact dMFM architecture.
    with open(os.path.join(os.environ["MODEL_DIR"], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    if args.score_free_guidance:
        assert args.cls_free_noclass_ratio == 0, 'no auxiliary class is needed for classifier free guidance if you do score free guidance. The training on the auxialiary classis is basically only data augmentation then.'
    if args.probability_tilt:
        assert args.cls_free_guidance
    assert not (args.probability_tilt and args.score_free_guidance)
    return args
