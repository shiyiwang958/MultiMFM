"""LightningModule for training the DNA dMFM (and, optionally, a Gaussian DFM).

Ported from DNA-MFM@187fe7b ``lightning_modules/dna_module.py``. Only the code
paths used with ``--model dit --mode gaussian`` are kept:

- ``--gaussian_mfm_diag_distill``: dMFM training through
  :class:`dmfm.utils.mfm_diag_distill.GaussianMfmDiagDistiller` (all paper dMFMs
  and 4-step students; launched by ``dmfm.experiments.train_dmfm``).
- otherwise: the DFM ``loss_vfm`` objective for a :class:`DiTSequenceModel`
  (the paper's base models were trained with the standalone
  ``dmfm.experiments.train_base_dfm`` instead; this branch is kept for parity).

Removed (never reachable in the DNA runs; they referenced purged modules such as
``model.dna_models_old``): Dirichlet/Riemannian/AR/distill inference,
classifier(-free) guidance, clean-classifier FXD metrics, fly-brain utilities.
The dMFM branch, validation probes, checkpoint metric and optimizer set-up are
unchanged.
"""

import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

from dmfm.lightning.general_module import GeneralModule, _ragged_log_frame, _wandb
from dmfm.models.dna_models import DiTSequenceModel, DNAMFMStudent
from dmfm.utils.flow_utils import gaussian_beta, gaussian_denoiser_flow_step, update_ema
from dmfm.utils.log_utils import get_logger
from dmfm.utils.mfm_diag_distill import GaussianMfmDiagDistiller

logger = get_logger(__name__)


class DNAModule(GeneralModule):
    def __init__(self, args, alphabet_size, num_cls):
        super().__init__(args)
        self.load_model(alphabet_size, num_cls)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.val_outputs = defaultdict(list)
        self.train_outputs = defaultdict(list)
        self.train_out_initialized = False
        self._mfm_diag_distiller = None
        self.mean_log_ema = {}
        self.inf_counter = 1
        self.nan_inf_counter = 0

    def on_fit_start(self):
        if getattr(self.args, "ckpt", None) is None:
            return
        lr = float(getattr(self.args, "lr", 0.0))
        if lr <= 0:
            return
        for optimizer in getattr(self.trainer, "optimizers", []):
            for group in optimizer.param_groups:
                group["lr"] = lr
        try:
            logger.info(f"Reset optimizer LR to args.lr={lr} after checkpoint restore")
        except Exception:
            pass

    @staticmethod
    def _bcast_like(coef, x):
        while coef.ndim < x.ndim:
            coef = coef[..., None]
        return coef.to(device=x.device, dtype=x.dtype)

    def _dfm_apply_loss_weighting(self, error, s, t, delta):
        """DFM VFM loss weighting at per-token shape [B, L]."""
        learnable = bool(getattr(self.args, "learnable_loss_weighting", False))
        p_adaptive = bool(getattr(self.args, "p_adaptive_loss_weighting", False))
        # Back-compat: the older DNA flag now aliases DFM's p-adaptive weighting.
        legacy_adaptive = bool(getattr(self.args, "gaussian_adaptive_loss", False))

        if learnable and (p_adaptive or legacy_adaptive):
            raise ValueError("Only one of learnable_loss_weighting or p-adaptive weighting can be enabled")

        if learnable:
            if not hasattr(self.model, "loss_weighting") or self.model.loss_weighting is None:
                raise ValueError("--learnable_loss_weighting requires model.loss_weighting")
            w = self.model.loss_weighting(s, t)
            w = self._bcast_like(w, error)
            return torch.exp(-w) * error + w, w

        if p_adaptive or legacy_adaptive:
            if p_adaptive:
                p = float(getattr(self.args, "grad_norm_p", 0.0))
                c = float(getattr(self.args, "grad_norm_c", 1e-8))
            else:
                p = float(getattr(self.args, "gaussian_adaptive_loss_r", 0.5))
                c = float(getattr(self.args, "gaussian_adaptive_loss_c", 0.01))
            if p == 0.0:
                return error, 1.0
            with torch.no_grad():
                w = 1.0 / ((delta.detach() + c) ** p)
            return w * error, w

        return error, 1.0

    def _dfm_vfm_loss(self, seq):
        """DFM loss_vfm for Gaussian mode: CE = -log psi_{t,t}(x_t)[x1] with DFM weighting."""
        B, L = seq.shape
        K = self.model.alphabet_size
        x1 = torch.nn.functional.one_hot(seq, num_classes=K).float()
        x0 = torch.randn((B, L, K), device=seq.device, dtype=x1.dtype)
        t = torch.rand((B,), device=seq.device)
        beta_t = gaussian_beta(self.args, t)
        xt = (1.0 - beta_t[:, None, None]) * x0 + beta_t[:, None, None] * x1

        log_psi_tt = self.model.psi_st(xt, t, t, return_log_psi=True)
        ce = -log_psi_tt.gather(-1, seq.long().unsqueeze(-1)).squeeze(-1)
        psi_tt = log_psi_tt.exp()
        q_true = psi_tt.gather(-1, seq.long().unsqueeze(-1)).squeeze(-1)
        delta = psi_tt.pow(2).sum(dim=-1) - 2.0 * q_true + 1.0

        weighted_ce, weight = self._dfm_apply_loss_weighting(ce, t, t, delta)
        losses = weighted_ce.mean(-1)

        self.lg("loss/unweight_vfm_loss", ce.mean())
        self.lg("dfm_vfm_t", t.mean())
        if isinstance(weight, torch.Tensor):
            self.lg("dfm_vfm_weight", weight.mean())
        return losses, log_psi_tt, xt, t

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {
            k: v for k, v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k
        }

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
            self.trainer.save_checkpoint(
                os.path.join(os.environ["MODEL_DIR"], f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt")
            )
        self.try_print_log()
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()

    def _get_distiller(self):
        if self._mfm_diag_distiller is None:
            self._mfm_diag_distiller = GaussianMfmDiagDistiller(
                self.args,
                alphabet_size=self.model.alphabet_size,
                device=self.device,
            )
        return self._mfm_diag_distiller

    def _mfm_step(self, seq, batch_idx):
        B, L = seq.shape
        if self.args.mode != "gaussian":
            raise ValueError("--gaussian_mfm_diag_distill requires --mode gaussian")
        distiller = self._get_distiller()
        try:
            global_step = int(self.trainer.global_step)
        except Exception:
            global_step = None

        # Diagonal GLASS always; off-diagonal consistency added after warmup.
        loss, logs = distiller.step_both(seq=seq, student_model=self.model, global_step=global_step)
        # Keep primary loss curve aligned with the actually-optimized objective.
        self.lg("loss", torch.full((B,), float(loss.detach().cpu().item()), device=self.device))

        keep = {"mfm_diag_loss_mean", "mfm_lsd_loss_mean", "mfm_esd_loss_mean", "mfm_gap", "mfm_tcond"}
        for k, v in logs.items():
            if k in keep:
                self.lg(k, v)

        pf = int(getattr(self.args, "print_freq", 200) or 200)
        it = int(getattr(self, "iter_step", 0) or 0)
        if pf > 0 and (it % pf == 0):
            def _f(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return float(x.detach().mean().cpu().item())
                try:
                    return float(x)
                except Exception:
                    return None

            msg = {
                "iter_step": it,
                "global_step": int(global_step) if global_step is not None else None,
                "mfm_s": _f(logs.get("mfm_s")),
                "mfm_u": _f(logs.get("mfm_u")),
                "mfm_gap": _f(logs.get("mfm_gap")),
                "mfm_tcond": _f(logs.get("mfm_tcond")),
                "mfm_diag_loss_mean": _f(logs.get("mfm_diag_loss_mean")),
                "mfm_lsd_loss_mean": _f(logs.get("mfm_lsd_loss_mean")),
                "mfm_lsd_unweighted_mean": _f(logs.get("mfm_lsd_loss_unweighted_mean")),
                "mfm_esd_loss_mean": _f(logs.get("mfm_esd_loss_mean")),
                "mfm_esd_unweighted_mean": _f(logs.get("mfm_esd_loss_unweighted_mean")),
                "mfm_consistency_type": str(getattr(self.args, "mfm_consistency_type", "lsd")),
            }
            print(f"MFM_DEBUG {msg}", flush=True)
            try:
                p = os.path.join(os.environ.get("MODEL_DIR", "."), "mfm_debug.txt")
                with open(p, "a") as f:
                    f.write(f"MFM_DEBUG {msg}\n")
            except Exception:
                pass

        # Fixed validation probes on the first validation batch.
        if self.stage == "val" and (batch_idx is None or int(batch_idx) == 0):
            gaps = list(getattr(self.args, "mfm_val_gaps", [0.01, 0.1, 0.5]))
            metrics = distiller.fixed_probe_metrics(seq=seq, student_model=self.model, gaps=gaps)
            for k, v in metrics.items():
                self.lg(k, v)
        self.lg("dur", torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        self.last_log_time = time.time()
        return loss

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq, cls = batch
        B, L = seq.shape

        if getattr(self.args, "gaussian_mfm_diag_distill", False):
            return self._mfm_step(seq, batch_idx)

        if not (self.args.mode == "gaussian" and hasattr(self.model, "psi_st") and bool(getattr(self.args, "dfm_vfm_loss", True))):
            raise NotImplementedError("dmfm keeps only the Gaussian DiT loss_vfm path (--mode gaussian --dfm_vfm_loss)")
        losses, logits, xt, alphas = self._dfm_vfm_loss(seq)

        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        self.lg('recovery_top1', torch.argmax(logits, dim=-1).eq(seq).float().mean(-1))

        if self.stage == "val":
            # Fixed-t denoising difficulty.
            x1 = torch.nn.functional.one_hot(seq, num_classes=self.model.alphabet_size).float()
            for t_val in [0.9, 0.7, 0.5, 0.3, 0.1]:
                t = torch.full((B,), float(t_val), device=self.device)
                x0 = torch.randn((B, L, self.model.alphabet_size), device=self.device)
                beta_t = gaussian_beta(self.args, t)
                xt_fix = beta_t[:, None, None] * x1 + (1 - beta_t[:, None, None]) * x0
                logits_fix = self.model(xt_fix, t=t)
                per_pos = torch.nn.functional.cross_entropy(logits_fix.transpose(1, 2), seq, reduction="none")
                per_ex = per_pos.mean(-1)
                self.lg(f"fixedt_loss_{t_val:.1f}", per_ex)
                self.lg(f"fixedt_ppl_{t_val:.1f}", torch.exp(per_ex))
                self.lg(f"fixedt_recovery_{t_val:.1f}", torch.argmax(logits_fix, dim=-1).eq(seq).float().mean(-1))

            logits_pred = self.gaussian_flow_inference(seq)
            seq_pred = torch.argmax(logits_pred, dim=-1)
            self.lg('seq', [''.join(['ACGT'[n] for n in s]) for s in seq_pred])
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            self.val_outputs['seqs'].append(seq_pred.cpu())
            self.val_outputs['true_seqs'].append(seq.detach().cpu())

        self.lg('alpha', alphas)
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def gaussian_flow_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.randn((B, L, K), device=self.device)
        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        last_logits = None
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt, last_logits, _ = gaussian_denoiser_flow_step(self.args, self.model, xt, s[None].expand(B), t[None].expand(B))
        if last_logits is None:
            last_logits = self.model(xt, torch.zeros(B, device=self.device))
        return last_logits

    def on_validation_epoch_start(self):
        self.inf_counter = 1
        self.nan_inf_counter = 0

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'val_nan_inf_step_fraction': self.nan_inf_counter / self.inf_counter})
        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        self.mean_log_ema = update_ema(current_dict=mean_log, prev_ema=self.mean_log_ema, gamma=0.9)
        mean_log.update(self.mean_log_ema)
        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                _wandb().log(mean_log, step=int(self.trainer.global_step))
            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            _ragged_log_frame(log).to_csv(path)

        # k-mer JS divergence of generated vs. true sequences (DFM branch only).
        try:
            if len(self.val_outputs.get('seqs', [])) > 0 and len(self.val_outputs.get('true_seqs', [])) > 0:
                seqs_gen = torch.cat(self.val_outputs['seqs'], dim=0)
                seqs_true = torch.cat(self.val_outputs['true_seqs'], dim=0)
                js1 = self._kmer_js_divergence(seqs_gen, seqs_true, k=1)
                js2 = self._kmer_js_divergence(seqs_gen, seqs_true, k=2)
                js1_ref = self._kmer_js_divergence_uniform_reference(seqs_true, k=1)
                js2_ref = self._kmer_js_divergence_uniform_reference(seqs_true, k=2)
                extra = {
                    "val_jsdiv_k1": float(js1),
                    "val_jsdiv_k2": float(js2),
                    "val_jsdiv_k1_ref": float(js1_ref),
                    "val_jsdiv_k2_ref": float(js2_ref),
                    "val_jsdiv_k1_norm": float(min(1.0, js1 / (js1_ref + 1e-8))),
                    "val_jsdiv_k2_norm": float(min(1.0, js2 / (js2_ref + 1e-8))),
                }
                if self.trainer.is_global_zero:
                    logger.info(str(extra))
                    self.log_dict(extra, batch_size=1)
        except Exception as e:
            if self.trainer.is_global_zero:
                logger.info(f"WARNING: failed k-mer JS logging: {e}")

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]
        self.val_outputs = defaultdict(list)

    @staticmethod
    def _kmer_counts(x: torch.Tensor, k: int) -> torch.Tensor:
        x = x.long()
        N, L = x.shape
        ids = torch.zeros((N, L - k + 1), dtype=torch.long, device=x.device)
        for i in range(k):
            ids = ids * 4 + x[:, i: i + (L - k + 1)]
        return torch.bincount(ids.reshape(-1), minlength=4 ** k).float()

    @classmethod
    def _kmer_js_divergence(cls, seqs_gen: torch.Tensor, seqs_true: torch.Tensor, k: int) -> float:
        cg = cls._kmer_counts(seqs_gen, k)
        ct = cls._kmer_counts(seqs_true, k)
        pg = cg / (cg.sum() + 1e-8)
        pt = ct / (ct.sum() + 1e-8)
        m = 0.5 * (pg + pt)
        kl_g = (pg * (pg.add(1e-8).log() - m.add(1e-8).log())).sum()
        kl_t = (pt * (pt.add(1e-8).log() - m.add(1e-8).log())).sum()
        return float((0.5 * (kl_g + kl_t)).detach().cpu().item())

    @classmethod
    def _kmer_js_divergence_uniform_reference(cls, seqs_true: torch.Tensor, k: int) -> float:
        ct = cls._kmer_counts(seqs_true, k)
        pt = ct / (ct.sum() + 1e-8)
        num = 4 ** k
        pu = torch.full((num,), 1.0 / num, device=pt.device, dtype=pt.dtype)
        m = 0.5 * (pu + pt)
        kl_u = (pu * (pu.add(1e-8).log() - m.add(1e-8).log())).sum()
        kl_t = (pt * (pt.add(1e-8).log() - m.add(1e-8).log())).sum()
        return float((0.5 * (kl_u + kl_t)).detach().cpu().item())

    def on_train_epoch_start(self) -> None:
        self.inf_counter = 1
        self.nan_inf_counter = 0

    def on_train_epoch_end(self):
        self.train_out_initialized = True
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                _wandb().log(mean_log)
        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().reshape(-1).tolist()
        elif isinstance(data, np.ndarray):
            data = data.reshape(-1).tolist()
        elif not isinstance(data, (list, tuple)):
            data = [data]
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def configure_optimizers(self):
        # The distiller may own a (unused, zero-weighted) weighting network; it is
        # included in the optimizer exactly as in the original code.
        params = list(self.parameters())
        if getattr(self.args, "gaussian_mfm_diag_distill", False):
            distiller = self._get_distiller()
            if distiller.teacher_wrapper is None:
                distiller.load_teacher()
            if (
                getattr(distiller, "teacher_model", None) is not None
                and hasattr(self.model, "warm_start_from_denoiser_teacher")
                and not getattr(self.model, "_mfm_teacher_warm_started", False)
                and getattr(self.args, "ckpt", None) is None
            ):
                self.model.warm_start_from_denoiser_teacher(distiller.teacher_model)
            if getattr(distiller, "weighting_model", None) is not None:
                params += list(distiller.weighting_model.parameters())

        opt_name = str(getattr(self.args, "optimizer", "adam")).lower()
        wd = float(getattr(self.args, "weight_decay", 0.0))
        if opt_name == "radam":
            optimizer = optim.RAdam(params, lr=self.args.lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=wd)

        warmup_steps = int(getattr(self.args, "lr_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            start_factor = float(getattr(self.args, "lr_warmup_start_factor", 0.1))
            warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
            const = ConstantLR(optimizer, factor=1.0, total_iters=int(1e12))
            scheduler = SequentialLR(optimizer, schedulers=[warmup, const], milestones=[warmup_steps])
        else:
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=int(1e12))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def load_model(self, alphabet_size, num_cls):
        if self.args.model != 'dit':
            raise NotImplementedError(f"dmfm keeps only --model dit (got {self.args.model!r})")
        if getattr(self.args, "gaussian_mfm_diag_distill", False):
            self.model = DNAMFMStudent(self.args, alphabet_size=alphabet_size)
        else:
            self.model = DiTSequenceModel(self.args, alphabet_size=alphabet_size)
