#!/usr/bin/env python
"""Train a DNA dMFM (discrete meta flow map) by distilling a base DFM teacher.

Ported from ``dirichlet-flow-matching/scripts/train_dna.py`` (Jul 26 2026;
launched by ``slurm/train_yeast_parent_dmfm_diagonal.sbatch`` for the dMFMs and
``slurm/train_yeast_parent_dmfm_4step_continue.sbatch`` for the 4-step
students). Flags are defined in :mod:`dmfm.utils.parsing`; the objective lives
in :mod:`dmfm.utils.mfm_diag_distill` and the Lightning loop in
:mod:`dmfm.lightning.dna_module`.

Changes vs. the original: package imports, argument parsing moved into
``main(argv)`` (so it can be called from Python), W&B only when ``--wandb``.
Training logic, seeding (``torch.manual_seed(0)`` after parsing), trainer and
checkpoint settings are unchanged. Example (the L=50 paper run, see
``scripts/dna/train_dmfm.sbatch`` for all flags)::

    python -m dmfm.experiments.train_dmfm --run_name L50_dmfm --no-wandb \\
        --yeast_data_pt data/dna/yeast_parent_disjoint/yeast_parent_L50.pt \\
        --yeast_split_pt data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \\
        --yeast_train_split a_train --yeast_val_split a_val \\
        --gaussian_mfm_diag_distill --mfm_teacher_ckpt checkpoints/dna/base/L50/best.pt \\
        --mfm_teacher_ckpt_hparams checkpoints/dna/base/L50/args.json ...
"""

from __future__ import annotations

import os


def main(argv=None):
    from torch.utils.data import Subset

    from dmfm.utils.parsing import parse_train_args

    args = parse_train_args(argv)

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy

    from dmfm.lightning.dna_module import DNAModule
    from dmfm.utils.dataset import YeastMiddleDataset
    from dmfm.utils.torch_io import torch_load
    from dmfm.utils.yeast_splits import subset_by_yeast_split

    torch.manual_seed(0)

    if args.wandb:
        import wandb

        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            settings=wandb.Settings(start_method="fork"),
            project=os.environ.get("WANDB_PROJECT", "dmfm"),
            name=args.run_name,
            config=args,
        )

    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    slurm_nnodes = int(os.environ.get("SLURM_NNODES", "1") or "1")
    is_ddp = slurm_ntasks > 1

    trainer = pl.Trainer(
        default_root_dir=os.environ["MODEL_DIR"],
        accelerator=args.accelerator,
        devices=1 if is_ddp or args.accelerator == "cpu" else "auto",
        num_nodes=slurm_nnodes if is_ddp else 1,
        strategy=DDPStrategy(find_unused_parameters=False) if is_ddp else "auto",
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=getattr(args, "log_every_n_steps", 50),
        enable_progress_bar=not (args.wandb or args.no_tqdm),
        gradient_clip_val=args.grad_clip,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.environ["MODEL_DIR"],
                save_top_k=5,
                save_last=True,
                monitor=(
                    "val_fxd_generated_to_allseqs"
                    if args.fid_early_stop
                    else (
                        (getattr(args, "mfm_checkpoint_metric", None) or "val_loss")
                        if getattr(args, "gaussian_mfm_diag_distill", False)
                        else "val_perplexity"
                    )
                ),
                mode="min",
            )
        ],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
    )

    if args.dataset_type != 'yeast':
        raise ValueError(f"Unsupported dataset_type={args.dataset_type!r}; this repo keeps only the yeast DNA path.")

    full_train_ds = YeastMiddleDataset(args, path=args.yeast_data_pt)
    full_val_ds = YeastMiddleDataset(args, path=args.yeast_data_pt)
    if args.yeast_split_pt:
        train_ds = subset_by_yeast_split(full_train_ds, args.yeast_split_pt, args.yeast_train_split)
        val_ds = subset_by_yeast_split(full_val_ds, args.yeast_split_pt, args.yeast_val_split)
    else:
        # Legacy default for historical checkpoints: fixed 95/5 split.
        val_len = int(0.05 * len(full_train_ds))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(full_train_ds), generator=g)
        train_ds = Subset(full_train_ds, perm[val_len:])
        val_ds = Subset(full_val_ds, perm[:val_len])

    if args.subset_train_as_val:
        val_set_size = len(val_ds) if args.constant_val_len is None else args.constant_val_len
        val_ds = Subset(train_ds, torch.randperm(len(train_ds))[:val_set_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    model = DNAModule(args, base_ds.alphabet_size, base_ds.num_cls)

    if args.init_ckpt:
        init_ckpt = torch_load(args.init_ckpt, map_location="cpu")
        init_state = init_ckpt.get("state_dict", init_ckpt)
        # These auxiliary modules are deliberately omitted from dMFM checkpoints;
        # keep the filter for compatibility with general DNA checkpoints.
        init_state = {
            key: value
            for key, value in init_state.items()
            if "cls_model" not in key and "distill_model" not in key
        }
        incompatible = model.load_state_dict(init_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Weight-only initialization from {args.init_ckpt} was incompatible: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        # Loading a learned dMFM must not trigger the first-step teacher warm start,
        # which would overwrite the learned flow-map parameters.
        if hasattr(model.model, "_mfm_teacher_warm_started"):
            model.model._mfm_teacher_warm_started = True
        print(f"Initialized model weights from {args.init_ckpt}; optimizer starts fresh.", flush=True)

    if args.validate:
        trainer.validate(model, train_loader if args.validate_on_train else val_loader, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    return os.environ["MODEL_DIR"]


if __name__ == "__main__":
    main()
