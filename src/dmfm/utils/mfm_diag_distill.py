"""dMFM training objective: diagonal GLASS distillation + ESD/LSD consistency.

Ported from the July 2026 ``dirichlet-flow-matching/utils/mfm_diag_distill.py``
(newer than DNA-MFM@187fe7b). The July version fixes the gap sampling of
``--mfm_consistency_gap_mode {fixed,long}``: a lower gap bound of 0 is honoured
instead of being replaced by the 0.75 default (``value or default`` bug). The
4-step students (``*_dmfm_4step_esd_gap_upto025_corrected_20260727``, Table 18)
were trained with this corrected version; the diagonal+ESD dMFMs use
``gap_mode=random`` and are unaffected. Pass ``--mfm_legacy_gap_or_default``
to the trainer (``args.mfm_legacy_gap_or_default=True``) to get the pre-fix
behaviour.

Other changes: package imports and the vendored ``dmfm._mfm`` instead of the
``sys.path``-based ``utils.mfm_lib``.
"""

import copy
import os
from types import SimpleNamespace

import torch
import yaml

from dmfm._mfm import LossWeightingNetwork, compute_loss, extract_posterior_velocity
from dmfm.models.dna_models import DiTSequenceModel, DNAMFMTeacherAdapter
from dmfm.utils.esm import upgrade_state_dict
from dmfm.utils.flow_utils import gaussian_beta
from dmfm.utils.torch_io import torch_load


def import_mfm_extract_posterior_velocity():
    return extract_posterior_velocity


def import_mfm_compute_loss():
    return compute_loss


def import_mfm_loss_weighting_network():
    return LossWeightingNetwork


def _broadcast_to_shape(tensor: torch.Tensor, shape) -> torch.Tensor:
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


def _namespace_from_mapping(value):
    if isinstance(value, dict):
        return SimpleNamespace(**copy.deepcopy(value))
    return copy.deepcopy(value)


def _checkpoint_model_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "student_state_dict"):
            state_dict = ckpt.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    return ckpt


def _best_state_dict_for_model(model: torch.nn.Module, state_dict: dict):
    candidates = [state_dict, upgrade_state_dict(state_dict, prefixes=["model."])]
    model_keys = set(model.state_dict())
    best = max(candidates, key=lambda sd: len(model_keys.intersection(sd.keys())))
    return {
        k: v
        for k, v in best.items()
        if not ("cls_model" in k or "distill_model" in k)
    }


class GaussianMfmDiagDistiller:
    """
    Encapsulates META flow map diagonal GLASS distillation:
      - loads/freeze teacher
      - wraps teacher into MFM BaseModel interface
      - computes GLASS target via mfm.losses.extract_posterior_velocity

    This keeps the LightningModule (`DNAModule`) clean and reduces brittleness.
    """

    def __init__(self, args, *, alphabet_size: int, device: torch.device):
        self.args = args
        self.alphabet_size = int(alphabet_size)
        self.device = device
        self.teacher_model = None
        self.teacher_wrapper = None
        self.weighting_model = None
        self._fixed_probe = None

    def load_teacher(self):
        ckpt_path = getattr(self.args, "mfm_teacher_ckpt", None)
        if ckpt_path is None:
            raise ValueError("--gaussian_mfm_diag_distill requires --mfm_teacher_ckpt")

        hparams_path = getattr(self.args, "mfm_teacher_ckpt_hparams", None)
        if hparams_path is None:
            base = os.path.dirname(os.path.abspath(ckpt_path))
            cand = os.path.join(base, "lightning_logs", "version_0", "hparams.yaml")
            if os.path.exists(cand):
                hparams_path = cand
        if hparams_path is None or not os.path.exists(hparams_path):
            raise FileNotFoundError(
                "Could not find teacher hparams.yaml. Provide --mfm_teacher_ckpt_hparams explicitly."
            )

        with open(hparams_path) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
        if isinstance(hparams, dict) and "model_cfg" in hparams:
            # Standalone DFM checkpoints save an args.json with both training
            # args and the exact model_cfg used to instantiate the teacher.
            teacher_args = _namespace_from_mapping(hparams["model_cfg"])
        elif isinstance(hparams, dict) and "args" in hparams:
            teacher_args = _namespace_from_mapping(hparams["args"])
        else:
            teacher_args = _namespace_from_mapping(hparams)

        # Ensure teacher is a standard DNA model (not META) and in gaussian mode.
        setattr(teacher_args, "gaussian_mfm_diag_distill", False)
        setattr(teacher_args, "mode", "gaussian")
        teacher = DiTSequenceModel(teacher_args, alphabet_size=self.alphabet_size)

        ckpt = torch_load(ckpt_path, map_location=self.device)
        state_dict = _checkpoint_model_state_dict(ckpt)
        model_dict = _best_state_dict_for_model(teacher, state_dict)
        incompat = teacher.load_state_dict(model_dict, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            print(
                "MFM teacher load incompatibility:",
                f"missing={len(incompat.missing_keys)}",
                f"unexpected={len(incompat.unexpected_keys)}",
                flush=True,
            )
        if len(incompat.missing_keys) >= len(teacher.state_dict()) // 2:
            raise RuntimeError(
                "MFM teacher checkpoint did not load correctly. "
                f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)} "
                f"ckpt={ckpt_path} hparams={hparams_path}"
            )
        teacher.eval()
        teacher.to(self.device)
        for p in teacher.parameters():
            p.requires_grad = False

        self.teacher_model = teacher
        self.teacher_wrapper = DNAMFMTeacherAdapter(teacher).to(self.device)
        self.teacher_wrapper.eval()
        for p in self.teacher_wrapper.parameters():
            p.requires_grad = False

        # Create weighting model if the dependency is available; otherwise we will
        # fall back to adaptive loss.
        try:
            LossWeightingNetwork = import_mfm_loss_weighting_network()
            self.weighting_model = LossWeightingNetwork(channels=int(getattr(self.args, "hidden_dim", 128)))
            self.weighting_model.to(self.device)
        except Exception:
            self.weighting_model = None

    def _sample_t_cond(self, B: int, *, global_step) -> torch.Tensor:
        """
        Port of MFM's t_cond schedule:
        - for warmup, t_cond = 0
        - after, t_cond = (U**power) * Bernoulli(1-0_rate)
        """
        warmup = int(getattr(self.args, "mfm_t_cond_warmup_steps", 0) or 0)
        power = float(getattr(self.args, "mfm_t_cond_power", 1.0))
        zero_rate = float(getattr(self.args, "mfm_t_cond_0_rate", 0.0))
        if global_step is None:
            gs = 0
        else:
            gs = int(global_step)
        if warmup > 0 and gs < warmup:
            return torch.zeros((B,), device=self.device)
        # after warmup: sample and optionally drop to 0
        t = torch.rand((B,), device=self.device)
        if power != 1.0:
            t = t.pow(power)
        if zero_rate > 0:
            keep = torch.bernoulli(torch.full((B,), 1.0 - zero_rate, device=self.device))
            t = t * keep
        return t

    def _sample_fixed_gap_s_u(self, B: int, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
        delta = float(max(0.0, min(1.0, delta)))
        s = torch.rand((B,), device=self.device) * (1.0 - delta)
        u = (s + delta).clamp_max(1.0)
        return s, u

    def _sample_s_u(self, B: int, *, global_step) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Port of MFM's s,u schedule (anneal gap size after warmup).
        - during warmup: s=u (diagonal)
        - then: start with tiny gaps and linearly increase to full random gaps
        Optional non-random modes intentionally bias consistency training toward
        long jumps / one-step maps.
        """
        warmup = int(getattr(self.args, "mfm_consistency_warmup_steps", 2500) or 0)
        anneal_end = int(getattr(self.args, "mfm_consistency_anneal_end_step", warmup) or warmup)
        if anneal_end <= warmup:
            # Treat small values as an anneal duration after warmup. This matches
            # how the SLURM configs use e.g. warmup=1M, anneal_end=25k.
            anneal_end = warmup + max(anneal_end, 1)
        step_offset = int(getattr(self.args, "mfm_consistency_step_offset", 0) or 0)
        if global_step is None:
            gs = 0
        else:
            gs = int(global_step)
        gs_eff = max(0, gs - step_offset)

        t1 = torch.rand((B,), device=self.device)
        t2 = torch.rand((B,), device=self.device)
        t_min = torch.minimum(t1, t2)
        t_max = torch.maximum(t1, t2)
        mid = 0.5 * (t_min + t_max)
        dist = (t_max - t_min)

        if gs_eff < warmup:
            return t1, t1

        gap_mode = str(getattr(self.args, "mfm_consistency_gap_mode", "random") or "random").lower()
        if gap_mode == "endpoint":
            return torch.zeros((B,), device=self.device), torch.ones((B,), device=self.device)
        legacy = bool(getattr(self.args, "mfm_legacy_gap_or_default", False))
        if gap_mode == "fixed":
            delta_value = getattr(self.args, "mfm_consistency_fixed_gap", 0.75)
            if legacy:  # DNA-MFM@187fe7b behaviour
                delta = float(delta_value or 0.75)
            else:
                delta = float(0.75 if delta_value is None else delta_value)
            return self._sample_fixed_gap_s_u(B, delta)
        if gap_mode == "long":
            # Zero is a valid lower bound, so do not use ``value or default``
            # here: it would silently turn [0, 0.25] into [0.25, 0.75].
            min_gap_value = getattr(self.args, "mfm_consistency_min_gap", 0.75)
            max_gap_value = getattr(self.args, "mfm_consistency_max_gap", 1.0)
            if legacy:  # DNA-MFM@187fe7b behaviour
                min_gap = float(min_gap_value or 0.75)
                max_gap = float(max_gap_value or 1.0)
            else:
                min_gap = float(0.75 if min_gap_value is None else min_gap_value)
                max_gap = float(1.0 if max_gap_value is None else max_gap_value)
            lo = float(max(0.0, min(1.0, min_gap)))
            hi = float(max(0.0, min(1.0, max_gap)))
            if hi < lo:
                lo, hi = hi, lo
            delta = lo + (hi - lo) * torch.rand((B,), device=self.device)
            s = torch.rand((B,), device=self.device) * (1.0 - delta)
            u = (s + delta).clamp_max(1.0)
            return s, u
        if gap_mode != "random":
            raise ValueError(
                f"Unknown --mfm_consistency_gap_mode={gap_mode!r}; "
                "expected 'random', 'fixed', 'long', or 'endpoint'."
            )

        if gs_eff < anneal_end:
            denom = max(int(anneal_end - warmup), 1)
            progress = float(gs_eff - warmup) / float(denom)
            max_step_size = float(max(0.0, min(1.0, progress)))
            s = mid - 0.5 * max_step_size * dist
            u = mid + 0.5 * max_step_size * dist
            return s, u
        return t_min, t_max

    def step_both(self, *, seq: torch.Tensor, student_model, global_step=None):
        """
        Upstream-style training step:
          - Always compute diagonal GLASS distillation loss (on the diagonal).
          - If enabled + past warmup, also compute off-diagonal consistency loss.
        Both terms share the same conditioning variables (t_cond, x_tcond) like upstream.
        """
        if self.teacher_wrapper is None:
            self.load_teacher()
        gs_for_warm_start = int(global_step) if global_step is not None else 0
        if (
            self.teacher_model is not None
            and hasattr(student_model, "warm_start_from_denoiser_teacher")
            and not getattr(student_model, "_mfm_teacher_warm_started", False)
            and gs_for_warm_start == 0
        ):
            student_model.warm_start_from_denoiser_teacher(self.teacher_model)

        B, L = seq.shape
        K = self.alphabet_size
        x1 = torch.nn.functional.one_hot(seq, num_classes=K).float()

        # --- 1) Conditioning variables (shared across terms) ---
        t_cond = self._sample_t_cond(B, global_step=global_step)
        beta_tcond = gaussian_beta(self.args, t_cond)
        x0_cond = torch.randn((B, L, K), device=self.device)
        x_tcond = beta_tcond[:, None, None] * x1 + (1 - beta_tcond[:, None, None]) * x0_cond

        # --- 2) Diagonal GLASS (distilled FM) ---
        s = torch.rand((B,), device=self.device)
        beta_s, _ = gaussian_beta(self.args, s, return_deriv=True)
        x0 = torch.randn((B, L, K), device=self.device)
        I_s = beta_s[:, None, None] * x1 + (1 - beta_s[:, None, None]) * x0

        extract_posterior_velocity = import_mfm_extract_posterior_velocity()
        with torch.no_grad():
            v_teacher = extract_posterior_velocity(
                s,
                I_s,
                x_tcond,
                t_cond,
                labels=None,
                cfg_scales=torch.ones((B,), device=self.device),
                teacher_model=self.teacher_wrapper,
                checkpoint_type="sit",
            ).detach()

        compute_loss = import_mfm_compute_loss()
        # Match upstream MFM defaults: p=0.5 for diagonal, p=1.0 for off-diagonal distillation.
        p_diag = float(getattr(self.args, "mfm_diag_adaptive_p", 0.5))
        p_lsd = float(getattr(self.args, "mfm_lsd_adaptive_p", 1.0))
        c = float(getattr(self.args, "gaussian_adaptive_loss_c", 0.01))
        weighting = torch.zeros((B,), device=self.device)

        diag_atom_loss = str(getattr(self.args, "mfm_diag_atom_loss", "velocity"))
        ce_per = kl_per = None
        if diag_atom_loss == "ce":
            diag_loss_mean, ce_per, kl_per, diag_sqerr = self._diag_ce_loss(
                student_model=student_model, I_s=I_s, s=s, t_cond=t_cond,
                x_tcond=x_tcond, v_teacher=v_teacher,
            )
            diag_loss_unweighted = diag_loss_mean
        else:
            v_pred_diag = student_model.v(s, s, I_s, t_cond, x_tcond)
            diag_loss_mean, diag_loss_unweighted = compute_loss(
                v_pred_diag,
                v_teacher,
                weighting,
                loss_type="adaptive",
                adaptive_p=p_diag,
                adaptive_c=c,
                stop_gradient=True,
            )
            diag_sqerr = (v_pred_diag - v_teacher).pow(2).mean(dim=(1, 2))

        logs = {
            "mfm_diag_loss": diag_sqerr,
            "mfm_diag_loss_mean": torch.full(
                (B,), float(diag_loss_mean.detach().cpu().item()), device=self.device
            ),
            "mfm_diag_loss_unweighted_mean": torch.full(
                (B,), float(diag_loss_unweighted.detach().cpu().item()), device=self.device
            ),
            "mfm_s": s,
            "mfm_tcond": t_cond,
        }
        if ce_per is not None:
            logs["mfm_diag_ce"] = ce_per
            logs["mfm_diag_kl"] = kl_per

        # Weighted sum like upstream (terms always present, off-diagonal gated by warmup).
        w_diag = float(getattr(self.args, "mfm_diag_loss_weight", 1.0))
        total = w_diag * diag_loss_mean

        # --- 3) Off-diagonal consistency (optional) ---
        do_consistency = bool(getattr(self.args, "mfm_consistency", False))
        warmup = int(getattr(self.args, "mfm_consistency_warmup_steps", 2500) or 0)
        gs = int(global_step) if global_step is not None else 0
        if do_consistency and gs >= warmup:
            consistency_type = str(getattr(self.args, "mfm_consistency_type", "lsd")).lower()
            s2, u2 = self._sample_s_u(B, global_step=global_step)
            beta_s2 = gaussian_beta(self.args, s2)
            x0_2 = torch.randn((B, L, K), device=self.device)
            I_s2 = beta_s2[:, None, None] * x1 + (1 - beta_s2[:, None, None]) * x0_2

            def vsu_fn(s_in, u_in, x_in):
                return student_model.v(s_in, u_in, x_in, t_cond, x_tcond)

            def Xsu_fn(s_in, u_in, x_in):
                v_su = student_model.v(s_in, u_in, x_in, t_cond, x_tcond)
                return student_model.X(s_in, u_in, x_in, v_su)

            if consistency_type == "lsd":
                primals = (s2, u2, I_s2)
                tangents = (torch.zeros_like(s2), torch.ones_like(u2), torch.zeros_like(I_s2))
                Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
                student = student_model.v(u2, u2, Xsu, t_cond, x_tcond)
                target = dXdu
                log_prefix = "mfm_lsd"
            elif consistency_type in {"esd", "esd_teacher"}:
                with torch.no_grad():
                    vss_teacher = extract_posterior_velocity(
                        s2,
                        I_s2,
                        x_tcond,
                        t_cond,
                        labels=None,
                        cfg_scales=torch.ones((B,), device=self.device),
                        teacher_model=self.teacher_wrapper,
                        checkpoint_type="sit",
                    ).detach()
                primals = (s2, u2, I_s2)
                tangents = (torch.ones_like(s2), torch.zeros_like(u2), vss_teacher)
                student, jvp = torch.func.jvp(vsu_fn, primals, tangents)
                target = vss_teacher + _broadcast_to_shape(u2 - s2, jvp.shape) * jvp
                log_prefix = "mfm_esd"
            else:
                raise ValueError(
                    f"Unknown --mfm_consistency_type={consistency_type!r}; "
                    "expected 'lsd' or 'esd_teacher'."
                )

            consistency_loss_mean, consistency_loss_unweighted = compute_loss(
                student,
                target,
                weighting,
                loss_type="adaptive",
                adaptive_p=p_lsd,
                adaptive_c=c,
                stop_gradient=True,
            )
            consistency_sqerr = (student - target).pow(2).mean(dim=(1, 2))
            student_norm = student.pow(2).mean(dim=(1, 2)).sqrt()
            target_norm = target.pow(2).mean(dim=(1, 2)).sqrt()
            logs.update(
                {
                    f"{log_prefix}_loss": consistency_sqerr,
                    f"{log_prefix}_loss_mean": torch.full(
                        (B,),
                        float(consistency_loss_mean.detach().cpu().item()),
                        device=self.device,
                    ),
                    # Normalize the unweighted mean per element to avoid dimension-scale artifacts.
                    f"{log_prefix}_loss_unweighted_mean": torch.full(
                        (B,),
                        float(
                            (consistency_loss_unweighted / float(L * K))
                            .detach()
                            .cpu()
                            .item()
                        ),
                        device=self.device,
                    ),
                    f"{log_prefix}_student_norm": student_norm,
                    f"{log_prefix}_target_norm": target_norm,
                    "mfm_u": u2,
                    "mfm_gap": (u2 - s2),
                }
            )
            if consistency_type in {"esd", "esd_teacher"}:
                logs["mfm_esd_teacher_diag_norm"] = vss_teacher.pow(2).mean(dim=(1, 2)).sqrt()
                logs["mfm_esd_jvp_norm"] = jvp.pow(2).mean(dim=(1, 2)).sqrt()
            w_lsd = float(getattr(self.args, "mfm_lsd_loss_weight", 1.0))
            total = total + (w_lsd * consistency_loss_mean)

        return total, logs

    def _diag_ce_loss(self, *, student_model, I_s, s, t_cond, x_tcond, v_teacher):
        """Soft-label CE between the student meta denoiser and the teacher's psi_{t*}(S).

        Discrete analogue of Prop. 1: on the diagonal Psi_{s,s}(x_s) = psi_{t*}(S). For
        the linear interpolant the teacher velocity carries exactly that denoiser,
        psi = x_s + (1-s) v_teacher -- the same relation DNAMFMStudent.v inverts -- so
        the soft labels are read off v_teacher instead of querying the teacher twice.

        Returns (loss_mean, ce_per_example, kl_per_example, sqerr_per_example).
        """
        # psi = x_s + (1-s) v holds because DNAMFMStudent.v divides by (1-s), which
        # matches the interpolant only when beta(s) = s. Fail loudly otherwise.
        schedule = str(getattr(self.args, "gaussian_beta_schedule", "linear"))
        if schedule != "linear":
            raise ValueError(
                "--mfm_diag_atom_loss ce assumes the linear interpolant beta(s)=s, "
                f"got --gaussian_beta_schedule {schedule!r}"
            )
        one_minus_s = (1.0 - s).clamp_min(1e-6)
        psi_teacher = I_s + _broadcast_to_shape(one_minus_s, I_s.shape) * v_teacher
        # A probability vector up to float error; renormalise for clean soft labels.
        psi_teacher = psi_teacher.clamp_min(0.0)
        psi_teacher = psi_teacher / psi_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        psi_teacher = psi_teacher.detach()

        logits = student_model.forward_logits(I_s, s, s, t_cond=t_cond, x_cond=x_tcond)
        log_q = torch.log_softmax(logits.float(), dim=-1)
        ce_per = -(psi_teacher * log_q).sum(dim=-1).mean(dim=-1)        # [B]
        loss_mean = ce_per.mean()
        with torch.no_grad():
            ent = -torch.special.xlogy(psi_teacher, psi_teacher).sum(dim=-1).mean(dim=-1)
            kl_per = ce_per - ent                                       # zero at optimum
            psi_student = log_q.exp()
            v_student = (psi_student - I_s) / _broadcast_to_shape(one_minus_s, I_s.shape)
            sqerr_per = (v_student - v_teacher).pow(2).mean(dim=(1, 2))
        return loss_mean, ce_per, kl_per, sqerr_per

    def diag_step(self, *, seq: torch.Tensor, student_model, global_step=None):
        """
        Returns (loss_mean, logs_dict) where logs are per-example tensors.
        """
        if self.teacher_wrapper is None:
            self.load_teacher()

        B, L = seq.shape
        x1 = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).float()
        K = x1.shape[-1]

        # Conditioning sample (t_cond, x_tcond)
        t_cond = self._sample_t_cond(B, global_step=global_step)
        beta_tcond = gaussian_beta(self.args, t_cond)
        x0_cond = torch.randn((B, L, K), device=self.device)
        xt_cond = beta_tcond[:, None, None] * x1 + (1 - beta_tcond[:, None, None]) * x0_cond

        # Diagonal point (s, x_s)
        s = torch.rand((B,), device=self.device)
        beta_s, _ = gaussian_beta(self.args, s, return_deriv=True)
        x0 = torch.randn((B, L, K), device=self.device)
        xs = beta_s[:, None, None] * x1 + (1 - beta_s[:, None, None]) * x0

        extract_posterior_velocity = import_mfm_extract_posterior_velocity()
        with torch.no_grad():
            v_teacher = extract_posterior_velocity(
                s,
                xs,
                xt_cond,
                t_cond,
                labels=None,
                cfg_scales=torch.ones((B,), device=self.device),
                teacher_model=self.teacher_wrapper,
                checkpoint_type="none",
            ).detach()

        if str(getattr(self.args, "mfm_diag_atom_loss", "velocity")) == "ce":
            loss_mean, ce_per, kl_per, sqerr_per = self._diag_ce_loss(
                student_model=student_model, I_s=xs, s=s, t_cond=t_cond,
                x_tcond=xt_cond, v_teacher=v_teacher,
            )
            mean_like = lambda v: torch.full((B,), float(v.detach().cpu().item()), device=self.device)
            return loss_mean, {
                "mfm_diag_loss": sqerr_per,
                "mfm_diag_loss_mean": mean_like(loss_mean),
                "mfm_diag_loss_unweighted_mean": mean_like(loss_mean),
                "mfm_diag_ce": ce_per,
                "mfm_diag_kl": kl_per,
                "mfm_s": s,
                "mfm_tcond": t_cond,
                "mfm_beta_s": beta_s,
                "mfm_beta_tcond": beta_tcond,
            }

        v_pred = student_model.v(s, s, xs, t_cond, xt_cond)

        compute_loss = import_mfm_compute_loss()
        # Per-example squared error for logging (stable scale).
        sqerr_per = (v_pred - v_teacher).pow(2).mean(dim=(1, 2))  # [B]
        # Force adaptive loss for stability (downweights rare huge GLASS targets).
        p = float(getattr(self.args, "gaussian_adaptive_loss_r", 0.5))
        c = float(getattr(self.args, "gaussian_adaptive_loss_c", 0.01))
        weighting = torch.zeros((B,), device=self.device)
        loss_mean, loss_unweighted = compute_loss(
            v_pred,
            v_teacher,
            weighting,
            loss_type="adaptive",
            adaptive_p=p,
            adaptive_c=c,
            stop_gradient=True,
        )
        loss_unweighted_mean = torch.full(
            (B,), float(loss_unweighted.detach().cpu().item()), device=self.device
        )

        logs = {
            "mfm_diag_loss": sqerr_per,
            "mfm_diag_loss_mean": torch.full(
                (B,), float(loss_mean.detach().cpu().item()), device=self.device
            ),
            "mfm_diag_loss_unweighted_mean": loss_unweighted_mean,
            "mfm_s": s,
            "mfm_tcond": t_cond,
            "mfm_beta_s": beta_s,
            "mfm_beta_tcond": beta_tcond,
        }
        return loss_mean, logs

    def consistency_step(self, *, seq: torch.Tensor, student_model, global_step=None):
        """
        Off-diagonal consistency loss for debugging. Supports:
          - lsd: compare v(u,u,X(s,u)) to d/du X(s,u)
          - esd_teacher: upstream MFM's teacher-anchored ESD branch
        """
        if self.teacher_wrapper is None:
            self.load_teacher()

        B, L = seq.shape
        K = self.alphabet_size
        x1 = torch.nn.functional.one_hot(seq, num_classes=K).float()

        # conditioning sample
        t_cond = self._sample_t_cond(B, global_step=global_step)
        beta_tcond = gaussian_beta(self.args, t_cond)
        x0_cond = torch.randn((B, L, K), device=self.device)
        x_tcond = beta_tcond[:, None, None] * x1 + (1 - beta_tcond[:, None, None]) * x0_cond

        # sample s<u with annealed gap
        s, u = self._sample_s_u(B, global_step=global_step)
        beta_s = gaussian_beta(self.args, s)
        x0 = torch.randn((B, L, K), device=self.device)
        I_s = beta_s[:, None, None] * x1 + (1 - beta_s[:, None, None]) * x0

        consistency_type = str(getattr(self.args, "mfm_consistency_type", "lsd")).lower()

        def vsu_fn(s_in, u_in, x_in):
            return student_model.v(s_in, u_in, x_in, t_cond, x_tcond)

        def Xsu_fn(s_in, u_in, x_in):
            v_su = student_model.v(s_in, u_in, x_in, t_cond, x_tcond)
            return student_model.X(s_in, u_in, x_in, v_su)

        if consistency_type == "lsd":
            primals = (s, u, I_s)
            tangents = (torch.zeros_like(s), torch.ones_like(u), torch.zeros_like(I_s))
            Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
            student = student_model.v(u, u, Xsu, t_cond, x_tcond)
            target = dXdu
            log_prefix = "mfm_lsd"
        elif consistency_type in {"esd", "esd_teacher"}:
            extract_posterior_velocity = import_mfm_extract_posterior_velocity()
            with torch.no_grad():
                vss_teacher = extract_posterior_velocity(
                    s,
                    I_s,
                    x_tcond,
                    t_cond,
                    labels=None,
                    cfg_scales=torch.ones((B,), device=self.device),
                    teacher_model=self.teacher_wrapper,
                    checkpoint_type="sit",
                ).detach()
            primals = (s, u, I_s)
            tangents = (torch.ones_like(s), torch.zeros_like(u), vss_teacher)
            student, jvp = torch.func.jvp(vsu_fn, primals, tangents)
            target = vss_teacher + _broadcast_to_shape(u - s, jvp.shape) * jvp
            log_prefix = "mfm_esd"
        else:
            raise ValueError(
                f"Unknown --mfm_consistency_type={consistency_type!r}; "
                "expected 'lsd' or 'esd_teacher'."
            )

        compute_loss = import_mfm_compute_loss()
        # adaptive loss forced
        p = float(getattr(self.args, "gaussian_adaptive_loss_r", 0.5))
        c = float(getattr(self.args, "gaussian_adaptive_loss_c", 0.01))
        weighting = torch.zeros((B,), device=self.device)
        loss_mean, loss_unweighted = compute_loss(
            student,
            target,
            weighting,
            loss_type="adaptive",
            adaptive_p=p,
            adaptive_c=c,
            stop_gradient=True,
        )
        sqerr_per = (student - target).pow(2).mean(dim=(1, 2))
        logs = {
            f"{log_prefix}_loss": sqerr_per,
            f"{log_prefix}_loss_mean": torch.full((B,), float(loss_mean.detach().cpu().item()), device=self.device),
            f"{log_prefix}_loss_unweighted_mean": torch.full((B,), float(loss_unweighted.detach().cpu().item()), device=self.device),
            f"{log_prefix}_student_norm": student.pow(2).mean(dim=(1, 2)).sqrt(),
            f"{log_prefix}_target_norm": target.pow(2).mean(dim=(1, 2)).sqrt(),
            "mfm_s": s,
            "mfm_u": u,
            "mfm_tcond": t_cond,
        }
        if consistency_type in {"esd", "esd_teacher"}:
            logs["mfm_esd_teacher_diag_norm"] = vss_teacher.pow(2).mean(dim=(1, 2)).sqrt()
            logs["mfm_esd_jvp_norm"] = jvp.pow(2).mean(dim=(1, 2)).sqrt()
        return loss_mean, logs

    def consistency_fixed_gap_metrics(self, *, seq: torch.Tensor, student_model, gaps: list[float]):
        """
        Validation-only: compute consistency loss for fixed gaps delta in `gaps`.
        """
        if self.teacher_wrapper is None:
            self.load_teacher()

        B, L = seq.shape
        K = self.alphabet_size
        x1 = torch.nn.functional.one_hot(seq, num_classes=K).float()
        t_cond = torch.zeros((B,), device=self.device)  # deterministic for val
        beta_tcond = gaussian_beta(self.args, t_cond)
        x0_cond = torch.randn((B, L, K), device=self.device)
        x_tcond = beta_tcond[:, None, None] * x1 + (1 - beta_tcond[:, None, None]) * x0_cond
        consistency_type = str(getattr(self.args, "mfm_consistency_type", "lsd")).lower()
        extract_posterior_velocity = import_mfm_extract_posterior_velocity()

        out = {}
        for delta in gaps:
            s = torch.rand((B,), device=self.device) * (1.0 - float(delta))
            u = (s + float(delta)).clamp_max(1.0)
            beta_s = gaussian_beta(self.args, s)
            x0 = torch.randn((B, L, K), device=self.device)
            I_s = beta_s[:, None, None] * x1 + (1 - beta_s[:, None, None]) * x0

            def vsu_fn(s_in, u_in, x_in):
                return student_model.v(s_in, u_in, x_in, t_cond, x_tcond)

            def Xsu_fn(s_in, u_in, x_in):
                v_su = student_model.v(s_in, u_in, x_in, t_cond, x_tcond)
                return student_model.X(s_in, u_in, x_in, v_su)

            if consistency_type == "lsd":
                primals = (s, u, I_s)
                tangents = (torch.zeros_like(s), torch.ones_like(u), torch.zeros_like(I_s))
                Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
                student = student_model.v(u, u, Xsu, t_cond, x_tcond)
                target = dXdu
                prefix = "mfm_lsd"
            elif consistency_type in {"esd", "esd_teacher"}:
                with torch.no_grad():
                    vss_teacher = extract_posterior_velocity(
                        s,
                        I_s,
                        x_tcond,
                        t_cond,
                        labels=None,
                        cfg_scales=torch.ones((B,), device=self.device),
                        teacher_model=self.teacher_wrapper,
                        checkpoint_type="sit",
                    ).detach()
                primals = (s, u, I_s)
                tangents = (torch.ones_like(s), torch.zeros_like(u), vss_teacher)
                student, jvp = torch.func.jvp(vsu_fn, primals, tangents)
                target = vss_teacher + _broadcast_to_shape(u - s, jvp.shape) * jvp
                prefix = "mfm_esd"
            else:
                raise ValueError(
                    f"Unknown --mfm_consistency_type={consistency_type!r}; "
                    "expected 'lsd' or 'esd_teacher'."
                )
            loss = (student - target).pow(2).mean(dim=(1, 2))  # per-example
            # Key is prefixed by stage ("val_") by DNAModule.lg; avoid double "val_" in W&B.
            out[f"{prefix}_gap_{delta:g}"] = loss
        return out

    def fixed_probe_metrics(self, *, seq: torch.Tensor, student_model, gaps: list[float]):
        """
        Validation-only fixed objective.  The first validation batch seeds a
        cached diagonal target and fixed-gap consistency probes; subsequent
        validations reuse exactly the same tensors, so curves reflect learning
        rather than fresh stochastic targets.
        """
        if self.teacher_wrapper is None:
            self.load_teacher()

        if self._fixed_probe is None:
            probe_n = int(getattr(self.args, "mfm_fixed_probe_batch_size", 128) or 128)
            probe_seed = int(getattr(self.args, "mfm_fixed_probe_seed", 12345) or 12345)
            seq_fixed = seq[: min(int(seq.shape[0]), probe_n)].detach().clone()
            B, L = seq_fixed.shape
            K = self.alphabet_size
            cuda_devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(probe_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(probe_seed)

                x1 = torch.nn.functional.one_hot(seq_fixed, num_classes=K).float()
                t_cond = torch.zeros((B,), device=self.device)
                beta_tcond = gaussian_beta(self.args, t_cond)
                x0_cond = torch.randn((B, L, K), device=self.device)
                x_tcond = beta_tcond[:, None, None] * x1 + (1 - beta_tcond[:, None, None]) * x0_cond

                s_diag = torch.rand((B,), device=self.device)
                beta_s_diag = gaussian_beta(self.args, s_diag)
                x0_diag = torch.randn((B, L, K), device=self.device)
                I_s_diag = beta_s_diag[:, None, None] * x1 + (1 - beta_s_diag[:, None, None]) * x0_diag

                extract_posterior_velocity = import_mfm_extract_posterior_velocity()
                with torch.no_grad():
                    v_teacher_diag = extract_posterior_velocity(
                        s_diag,
                        I_s_diag,
                        x_tcond,
                        t_cond,
                        labels=None,
                        cfg_scales=torch.ones((B,), device=self.device),
                        teacher_model=self.teacher_wrapper,
                        checkpoint_type="sit",
                    ).detach()

                gap_cache = {}
                for delta in gaps:
                    delta = float(delta)
                    s_gap = torch.rand((B,), device=self.device) * (1.0 - delta)
                    u_gap = (s_gap + delta).clamp_max(1.0)
                    beta_s_gap = gaussian_beta(self.args, s_gap)
                    x0_gap = torch.randn((B, L, K), device=self.device)
                    I_s_gap = beta_s_gap[:, None, None] * x1 + (1 - beta_s_gap[:, None, None]) * x0_gap
                    gap_cache[delta] = {"s": s_gap, "u": u_gap, "I_s": I_s_gap}

            self._fixed_probe = {
                "t_cond": t_cond,
                "x_tcond": x_tcond,
                "s_diag": s_diag,
                "I_s_diag": I_s_diag,
                "v_teacher_diag": v_teacher_diag,
                "gaps": gap_cache,
            }

        p = self._fixed_probe
        out = {}

        v_pred_diag = student_model.v(
            p["s_diag"], p["s_diag"], p["I_s_diag"], p["t_cond"], p["x_tcond"]
        )
        diag_sqerr = (v_pred_diag - p["v_teacher_diag"]).pow(2).mean(dim=(1, 2))
        out["mfm_probe_diag_mse"] = diag_sqerr

        if bool(getattr(self.args, "mfm_consistency", False)):
            consistency_type = str(getattr(self.args, "mfm_consistency_type", "lsd")).lower()
            extract_posterior_velocity = import_mfm_extract_posterior_velocity()
            for delta, gap in p["gaps"].items():
                s = gap["s"]
                u = gap["u"]
                I_s = gap["I_s"]

                def vsu_fn(s_in, u_in, x_in):
                    return student_model.v(s_in, u_in, x_in, p["t_cond"], p["x_tcond"])

                def Xsu_fn(s_in, u_in, x_in):
                    v_su = student_model.v(s_in, u_in, x_in, p["t_cond"], p["x_tcond"])
                    return student_model.X(s_in, u_in, x_in, v_su)

                if consistency_type == "lsd":
                    primals = (s, u, I_s)
                    tangents = (torch.zeros_like(s), torch.ones_like(u), torch.zeros_like(I_s))
                    Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
                    student = student_model.v(u, u, Xsu, p["t_cond"], p["x_tcond"])
                    target = dXdu
                    prefix = "mfm_probe_lsd"
                elif consistency_type in {"esd", "esd_teacher"}:
                    with torch.no_grad():
                        vss_teacher = extract_posterior_velocity(
                            s,
                            I_s,
                            p["x_tcond"],
                            p["t_cond"],
                            labels=None,
                            cfg_scales=torch.ones_like(s),
                            teacher_model=self.teacher_wrapper,
                            checkpoint_type="sit",
                        ).detach()
                    primals = (s, u, I_s)
                    tangents = (torch.ones_like(s), torch.zeros_like(u), vss_teacher)
                    student, jvp = torch.func.jvp(vsu_fn, primals, tangents)
                    target = vss_teacher + _broadcast_to_shape(u - s, jvp.shape) * jvp
                    prefix = "mfm_probe_esd"
                else:
                    raise ValueError(
                        f"Unknown --mfm_consistency_type={consistency_type!r}; "
                        "expected 'lsd' or 'esd_teacher'."
                    )
                out[f"{prefix}_gap_{delta:g}"] = (student - target).pow(2).mean(dim=(1, 2))
        return out
