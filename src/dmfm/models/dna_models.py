"""DNA denoiser (base DFM) and dMFM student models.

Ported from DNA-MFM@187fe7b ``model/dna_models.py`` (reconstruction in
``scratch_dfm_recon_20260925/model/dna_models.py``). Changes: package imports
instead of flat ``model.``/``utils.`` imports, and the vendored ``dmfm._mfm``
instead of the ``sys.path``-based ``utils.mfm_lib`` import of ``mfm``.

- :class:`DiTSequenceModel`   base DFM mean denoiser psi_{s,t}(x) (Table 9 models)
- :class:`DNAMFMTeacherAdapter` wraps it as an MFM ``BaseModel`` teacher
- :class:`DNAMFMStudent`      dMFM X_{s,t}(x; t_cond, x_cond) (dMFM and 4-step students)
"""

import torch
from torch import nn
import torch.nn.functional as F

from dmfm._mfm import BaseModel as MFMBaseModel
from dmfm._mfm import extract_posterior_velocity as _mfm_extract_posterior_velocity
from dmfm.models.dit_seq import DiT1D, DiT1DConfig, EmbeddingLayer, TimestepEmbedder, modulate
from dmfm.utils.flow_utils import gaussian_beta


def _posterior_velocity(
    s,
    Is,
    xt_cond,
    t_cond,
    *,
    labels=None,
    cfg_scales=None,
    teacher_model=None,
    eps=1e-6,
    omega=0.6,
    checkpoint_type="sit",
):
    if teacher_model is None:
        raise ValueError("teacher_model is required for MFM posterior velocity extraction.")
    if cfg_scales is None:
        cfg_scales = torch.ones_like(s)
    return _mfm_extract_posterior_velocity(
        s,
        Is,
        xt_cond,
        t_cond,
        labels,
        cfg_scales,
        teacher_model,
        eps=eps,
        omega=omega,
        checkpoint_type=checkpoint_type,
    )


def _bcast(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a batch scalar coefficient like DFM's simplex_flow_map.utils._bcast."""
    while coef.ndim < x.ndim:
        coef = coef[..., None]
    return coef.to(device=x.device, dtype=x.dtype)


def _get_optional_dim(args, *names, default):
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return int(value)
    return int(default)


def _assert_diagonal_times(s, t, caller: str, *, atol: float = 1e-6) -> None:
    """Guard model.v calls that are only valid for instantaneous diagonal drift."""
    if t is None:
        return
    if not torch.is_tensor(s):
        s = torch.as_tensor(s)
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=s.device)
    t = t.to(device=s.device)
    gap = (s.float() - t.float()).abs().max()
    if bool(gap > atol):
        raise ValueError(
            f"{caller} is a diagonal teacher velocity and requires s == t; "
            f"got max |s-t|={gap.item():.3e}."
        )


def _rename_state_dict_keys(state_dict, prefix_map):
    upgraded = {}
    replace_map = {
        ".qkv.": ".attn_qkv.",
        ".proj.": ".attn_out.",
        ".wqkv.": ".attn_qkv.",
        ".wo.": ".attn_out.",
        ".ffn.w1.": ".mlp.0.",
        ".ffn.w2.": ".mlp.2.",
        ".mlp.3.": ".mlp.2.",
        ".final_norm.": ".norm_final.",
        ".final_ada.1.": ".adaLN_modulation.",
        ".adaLN_modulation.1.": ".adaLN_modulation.",
        ".output_proj.": ".output_layer.linear.",
    }
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_map.items():
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
        for old, new in replace_map.items():
            new_key = new_key.replace(old, new)
        if ".ffn.w3." in new_key:
            continue

        if new_key.startswith(("time_emb.", "delta_emb.", "t_cond_proj.", "s_proj_", "delta_proj_")):
            continue
        if new_key.startswith(("time_embedder.", "delta_embedder.", "t_cond_embedder.", "s_embedder_", "t_embedder_", "gap_embedder_")):
            continue
        if new_key.startswith("model.norm_final."):
            new_key = "model.output_layer.norm_final." + new_key[len("model.norm_final."):]
        elif new_key.startswith("model.adaLN_modulation."):
            new_key = "model.output_layer.adaLN_modulation." + new_key[len("model.adaLN_modulation."):]

        if new_key == "model.input_proj.weight":
            new_key = "model.vocab_embed.embedding"
            if value.ndim == 2 and value.shape[0] > value.shape[1]:
                value = value.T.contiguous()
        elif new_key == "model.input_proj.bias":
            continue
        elif new_key == "model.output_proj.weight":
            new_key = "model.output_layer.linear.weight"
        elif new_key == "model.output_proj.bias":
            new_key = "model.output_layer.linear.bias"
        elif new_key == "input_proj.weight":
            new_key = "model.vocab_embed.embedding"
            if value.ndim == 2 and value.shape[0] > value.shape[1]:
                value = value.T.contiguous()
        elif new_key == "input_proj.bias":
            continue
        elif new_key == "x_embed.weight":
            new_key = "x_embed.embedding"
            if value.ndim == 2 and value.shape[0] > value.shape[1]:
                value = value.T.contiguous()
        elif new_key == "x_embed.bias":
            continue
        elif new_key == "x_cond_embed.weight":
            new_key = "x_cond_embed.embedding"
            if value.ndim == 2 and value.shape[0] > value.shape[1]:
                value = value.T.contiguous()
        elif new_key == "x_cond_embed.bias":
            continue
        elif new_key == "output_proj.weight":
            new_key = "model.output_layer.linear.weight"
        elif new_key == "output_proj.bias":
            new_key = "model.output_layer.linear.bias"
        upgraded[new_key] = value
    return upgraded


class DiTSequenceModel(nn.Module):
    """
    DFM-style DNA mean denoiser psi_theta(x, s, t).

    Matches the DFM denoiser API:
      - ``forward_logits(x, s, t)`` returns raw z_{s,t}(x)
      - ``psi_st(x, s, t)`` returns softmax(z_{s,t}(x))
      - ``X_st(x, s, t)`` returns Gamma_{s,t} x + Delta_{s,t} psi_st

    ``forward(x, t=...)`` remains as a compatibility wrapper for the existing
    Lightning loop and is interpreted as the diagonal call s=t.
    """

    def __init__(self, args, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args

        dim = int(args.hidden_dim)
        heads = int(getattr(args, "transformer_heads", 8))
        mlp_ratio = float(getattr(args, "transformer_ff_mult", 4))
        cond_dim = _get_optional_dim(args, "dit_cond_dim", "cond_dim", default=dim)

        self.clean_data = bool(getattr(args, "clean_data", False))
        self.use_self_condition = (not self.clean_data) and float(getattr(args, "self_condition_ratio", 0.0)) > 0.0
        self.input_dim = self.alphabet_size * (2 if self.use_self_condition else 1)
        self.time_delta_param = bool(getattr(args, "time_delta_param", True))

        self.model = DiT1D(
            DiT1DConfig(
                dim=dim,
                heads=heads,
                blocks=int(getattr(args, "dit_blocks", 12)),
                mlp_ratio=mlp_ratio,
                dropout=float(args.dropout),
                cond_dim=cond_dim,
                input_dim=self.input_dim,
                out_dim=self.alphabet_size,
                double_temb=bool(getattr(args, "double_temb", True)),
                scale_by_sigma=bool(getattr(args, "scale_by_sigma", True)),
                preserve_denoiser=bool(getattr(args, "preserve_denoiser", False)),
                use_flash_attn=bool(getattr(args, "use_flash_attn", getattr(args, "dit_use_flash_attn", True))),
                softcap=float(getattr(args, "softcap", 50.0)),
            )
        )

    def beta(self, t):
        return gaussian_beta(self.args, t)

    def beta_dot(self, t):
        _, deriv = gaussian_beta(self.args, t, return_deriv=True)
        return deriv

    def alpha(self, t):
        return 1.0 - self.beta(t)

    def alpha_dot(self, t):
        return -self.beta_dot(t)

    def ell(self, t):
        """DFM interpolant ell_t = alpha'(t) / alpha(t)."""
        return self.alpha_dot(t) / self.alpha(t).clamp_min(1e-8)

    def lam(self, t):
        """DFM interpolant lambda_t = beta'(t) - beta(t) * ell_t."""
        return self.beta_dot(t) - self.beta(t) * self.ell(t)

    def Gamma(self, s, t):
        """Gamma_{s,t} = alpha(t) / alpha(s)."""
        return self.alpha(t) / self.alpha(s).clamp_min(1e-8)

    def Delta(self, s, t):
        """Delta_{s,t} = beta(t) - Gamma_{s,t} beta(s)."""
        return self.beta(t) - self.Gamma(s, t) * self.beta(s)

    def C(self, s, t):
        """DFM Lagrangian coefficient C_{s,t} = Delta_{s,t} / lambda_t."""
        numer = self.beta(t) * self.alpha(s) - self.beta(s) * self.alpha(t)
        denom = self.alpha(s) * self.lam(t)
        return numer / denom.clamp_min(1e-8)

    def kappa_inv(self, s, t):
        """DFM ESD coefficient kappa^{-1}_{s,t}."""
        numer = self.beta(t) * self.alpha(s) - self.beta(s) * self.alpha(t)
        denom = self.alpha(t) * self.lam(s)
        return numer / denom.clamp_min(1e-8)

    def omega_psd(self, s, u, t):
        """DFM PSD semigroup weight, using the same cross-product form."""
        alpha_s = self.alpha(s)
        beta_s = self.beta(s)
        numer = self.beta(u) * alpha_s - beta_s * self.alpha(u)
        denom = self.beta(t) * alpha_s - beta_s * self.alpha(t)
        return self.Gamma(u, t) * numer / denom.clamp_min(1e-8)

    def b_t(self, x, denoiser, t):
        """DFM probability-flow drift b_t(x) = ell_t x + lambda_t denoiser."""
        return _bcast(self.ell(t), x) * x + _bcast(self.lam(t), x) * denoiser

    def forward_logits(self, x, s, t=None, x_sc=None, **kwargs):
        """Raw logits z_{s,t}(x), matching the DFM denoiser convention."""
        _ = kwargs
        if t is None:
            t = s
        if self.use_self_condition:
            if x_sc is None:
                x_sc = torch.zeros_like(x)
            x = torch.cat([x, x_sc], dim=-1)
        sigma_prime = (t - s) if self.time_delta_param else t
        return self.model(x, sigma=s, sigma_prime=sigma_prime)

    def psi_st(self, x, s, t=None, *, return_logits=False, return_log_psi=False, x_sc=None, flow_temp=1.0, **kwargs):
        """Mean denoiser psi_{s,t}(x), with DFM-compatible return flags."""
        logits = self.forward_logits(x, s, t, x_sc=x_sc, **kwargs)
        if return_logits:
            return logits
        log_psi = F.log_softmax(logits / float(flow_temp), dim=-1)
        if return_log_psi:
            return log_psi
        return log_psi.exp()

    def _X_st_psi(self, x, s, t, psi):
        return _bcast(self.Gamma(s, t), x) * x + _bcast(self.Delta(s, t), x) * psi

    def X_st(self, x, s, t, *, x_sc=None, flow_temp=1.0, **kwargs):
        """X_{s,t}(x) = Gamma_{s,t} x + Delta_{s,t} psi_{s,t}(x)."""
        psi = self.psi_st(x, s, t, x_sc=x_sc, flow_temp=flow_temp, **kwargs)
        return self._X_st_psi(x, s, t, psi)

    def v(
        self,
        s,
        t,
        x,
        t_cond=None,
        x_cond=None,
        class_labels=None,
        cfg_scale=None,
        x_sc=None,
        flow_temp=1.0,
        **kwargs,
    ):
        """
        MFM-compatible diagonal teacher velocity.

        The vendored MFM posterior extractor only queries the teacher on the
        diagonal, ``v(t_star, t_star, x_star, ...)``.  For the DNA denoiser this
        is exactly DFM's instantaneous drift b_s(x) = ell_s x + lambda_s psi_s(x).
        """
        _assert_diagonal_times(s, t, "DiTSequenceModel.v")
        _ = (t_cond, x_cond, class_labels, cfg_scale)
        psi = self.psi_st(x, s, s, x_sc=x_sc, flow_temp=flow_temp, **kwargs)
        return self.b_t(x, psi, s)

    def score(self, x, t, *, u=None, x_sc=None, flow_temp=1.0, eps=1e-8, **kwargs):
        """
        Diagonal Gaussian score from the learned flow velocity.

        Uses
            grad_x log p_t(x) =
                (u_t(x) - (beta_dot_t / beta_t) x)
                / (alpha_t^2 beta_dot_t / beta_t - alpha_dot_t alpha_t).

        For the default alpha=1-t, beta=t schedule this reduces to
            (t * u_t(x) - x) / (1 - t).
        """
        if u is None:
            u = self.v(t, t, x, x_sc=x_sc, flow_temp=flow_temp, **kwargs)

        beta = self.beta(t).clamp_min(float(eps))
        beta_dot_over_beta = self.beta_dot(t) / beta
        denom = self.alpha(t).pow(2) * beta_dot_over_beta - self.alpha_dot(t) * self.alpha(t)
        return (u - _bcast(beta_dot_over_beta, x) * x) / _bcast(denom.clamp_min(float(eps)), x)

    def sde_half_sigma_sq(self, t, eps=1e-8):
        """Return sigma_t^2 / 2 for the stochastic interpolant SDE."""
        beta = self.beta(t).clamp_min(float(eps))
        beta_dot_over_beta = self.beta_dot(t) / beta
        half_sigma_sq = self.alpha(t).pow(2) * beta_dot_over_beta - self.alpha_dot(t) * self.alpha(t)
        return half_sigma_sq.clamp_min(float(eps))

    def sde_sigma_sq(self, t, eps=1e-8):
        """Return sigma_t^2. For linear beta this is 2 * (1 / t - 1)."""
        return 2.0 * self.sde_half_sigma_sq(t, eps=eps)

    def sde_drift(
        self,
        x,
        t,
        *,
        u=None,
        score=None,
        x_sc=None,
        flow_temp=1.0,
        eps=1e-8,
        return_parts=False,
        **kwargs,
    ):
        """SDE drift b_t(x) + (sigma_t^2 / 2) score_t(x)."""
        if u is None:
            u = self.v(t, t, x, x_sc=x_sc, flow_temp=flow_temp, **kwargs)
        if score is None:
            score = self.score(x, t, u=u, x_sc=x_sc, flow_temp=flow_temp, eps=eps, **kwargs)
        half_sigma_sq = self.sde_half_sigma_sq(t, eps=eps)
        drift = u + _bcast(half_sigma_sq, x) * score
        if return_parts:
            return drift, {"u": u, "score": score, "sigma_sq": 2.0 * half_sigma_sq}
        return drift

    @torch.no_grad()
    def sample_sde(
        self,
        x0,
        *,
        t_start=1e-3,
        t_end=1.0,
        n_steps=100,
        x_sc=None,
        flow_temp=1.0,
        eps=1e-8,
        return_traj=False,
        generator=None,
        **kwargs,
    ):
        """Euler-Maruyama sampler for dX = [b + sigma^2/2 score] dt + sigma dB."""
        x = x0
        times = torch.linspace(
            float(t_start),
            float(t_end),
            int(n_steps) + 1,
            device=x.device,
            dtype=x.dtype,
        )
        traj = [x] if return_traj else None
        for t_cur, t_next in zip(times[:-1], times[1:]):
            dt = t_next - t_cur
            t = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
            drift = self.sde_drift(x, t, x_sc=x_sc, flow_temp=flow_temp, eps=eps, **kwargs)
            sigma = self.sde_sigma_sq(t, eps=eps).sqrt()
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + drift * dt + _bcast(sigma, x) * dt.sqrt() * noise
            if return_traj:
                traj.append(x)
        if return_traj:
            return x, torch.stack(traj)
        return x

    def extract_posterior_velocity(
        self,
        s,
        Is,
        xt_cond,
        t_cond,
        *,
        labels=None,
        cfg_scales=None,
        eps=1e-6,
        omega=0.6,
        checkpoint_type="sit",
    ):
        """Use mfm.losses.losses.extract_posterior_velocity with this model as the default teacher."""
        return _posterior_velocity(
            s,
            Is,
            xt_cond,
            t_cond,
            labels=labels,
            cfg_scales=cfg_scales,
            teacher_model=self,
            eps=eps,
            omega=omega,
            checkpoint_type=checkpoint_type,
        )

    def forward(self, x, s=None, t=None, x_sc=None, **kwargs):
        if s is None:
            if t is None:
                raise TypeError("DiTSequenceModel.forward requires s or t")
            s = t
        if t is None:
            t = s
        return self.forward_logits(x, s, t, x_sc=x_sc, **kwargs)

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        upgraded = _rename_state_dict_keys(
            state_dict,
            {
                "embedder.": "input_proj.",
                "input_proj.": "input_proj.",
                "time_embedder.": "time_emb.",
                "delta_embedder.": "delta_emb.",
                "logits_head.": "output_proj.",
                "out.": "output_proj.",
                "backbone.": "model.",
                "denoiser.": "model.",
            },
        )
        return super().load_state_dict(upgraded, strict=strict, *args, **kwargs)


def _zero_timestep_embedder_last_layer(embedder: TimestepEmbedder) -> None:
    if hasattr(embedder, "mlp") and len(embedder.mlp) >= 3 and isinstance(embedder.mlp[2], nn.Linear):
        nn.init.constant_(embedder.mlp[2].weight, 0)
        nn.init.constant_(embedder.mlp[2].bias, 0)


class DNAMFMTeacherAdapter(MFMBaseModel):
    """
    Wrap a DFM mean denoiser so it behaves like an MFM ``BaseModel`` teacher.

    The MFM posterior extractor only needs access to ``v(s,t,x,t_cond,x_cond)``
    and in practice queries it on the diagonal.  The wrapped denoiser already
    exposes exactly the corresponding DFM drift through ``DiTSequenceModel.v``.
    """

    def __init__(self, denoiser: DiTSequenceModel):
        super().__init__()
        self.denoiser = denoiser
        self.args = denoiser.args
        self.alphabet_size = denoiser.alphabet_size

    def v(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        _assert_diagonal_times(s, t, "DNAMFMTeacherAdapter.v")
        return self.denoiser.v(
            s,
            s,
            x,
            t_cond=t_cond,
            x_cond=x_cond,
            class_labels=class_labels,
            **kwargs,
        )


class DNAMFMStudent(MFMBaseModel):
    """
    DNA MFM student model with upstream-MFM conditioning semantics on top of the
    DFM sequence backbone.

    Interface:
      v = model.v(s, t, x_s, t_cond, x_tcond)
    where x_s and x_tcond are continuous [B, L, K] tensors.  The DiT backbone
    predicts conditional denoiser logits; the velocity is induced from the
    simplex denoiser, matching discrete MFM:
        Psi_{s,t} = softmax(logits_{s,t})
        v_{s,t}(x) = (Psi_{s,t}(x) - x) / (1 - s).
    """

    def __init__(self, args, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args

        dim = int(args.hidden_dim)
        heads = int(getattr(args, "transformer_heads", 8))
        mlp_ratio = float(getattr(args, "transformer_ff_mult", 4))
        blocks = int(getattr(args, "dit_blocks", 12))
        cond_dim = _get_optional_dim(args, "dit_cond_dim", "cond_dim", default=dim)

        # Upstream MFM keeps separate x and x_cond embedding tables.
        self.x_embed = EmbeddingLayer(dim, self.alphabet_size)
        self.x_cond_embed = EmbeddingLayer(dim, self.alphabet_size)

        # MFM-style x_cond gating driven by t_cond.
        self.x_cond_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.preserve_t_cond_0 = bool(getattr(args, "mfm_preserve_t_cond_0", True))

        # Match upstream MFM conditioning semantics: stage 1 sees s and stage 2
        # sees t, with t_cond shared across both stages.
        self.s_embedder = TimestepEmbedder(cond_dim)
        self.t_embedder = TimestepEmbedder(cond_dim)
        self.t_cond_embedder = TimestepEmbedder(cond_dim)
        self.s_embedder_second = TimestepEmbedder(cond_dim)
        self.t_embedder_second = TimestepEmbedder(cond_dim)

        enc_depth = getattr(args, "mfm_encoder_depth", None)
        self.encoder_depth = int(enc_depth) if enc_depth is not None else (blocks // 2)

        self.model = DiT1D(
            DiT1DConfig(
                dim=dim,
                heads=heads,
                blocks=blocks,
                mlp_ratio=mlp_ratio,
                dropout=float(args.dropout),
                cond_dim=cond_dim,
                input_dim=self.alphabet_size,
                out_dim=self.alphabet_size,
                double_temb=bool(getattr(args, "double_temb", True)),
                scale_by_sigma=bool(getattr(args, "scale_by_sigma", True)),
                preserve_denoiser=bool(getattr(args, "preserve_denoiser", False)),
                use_flash_attn=bool(getattr(args, "use_flash_attn", getattr(args, "dit_use_flash_attn", True))),
                softcap=float(getattr(args, "softcap", 50.0)),
            )
        )
        self._mfm_teacher_warm_started = False
        self.initialize_weights_mfm_like()

    def initialize_weights_mfm_like(self):
        # Gate off the conditioning branch at initialization, like upstream MFM.
        nn.init.constant_(self.x_cond_embed.embedding, 0)
        nn.init.constant_(self.x_cond_adaLN[-1].weight, 0)
        nn.init.constant_(self.x_cond_adaLN[-1].bias, 0)
        half_dim = self.x_cond_adaLN[-1].bias.shape[0] // 2
        nn.init.constant_(self.x_cond_adaLN[-1].bias[half_dim:], -1)

        # Stage layout mirrors upstream MFM:
        # - stage 1 uses s with t branch off
        # - stage 2 uses t with s branch off
        _zero_timestep_embedder_last_layer(self.t_cond_embedder)
        _zero_timestep_embedder_last_layer(self.t_embedder)
        _zero_timestep_embedder_last_layer(self.s_embedder_second)

    @torch.no_grad()
    def warm_start_from_denoiser_teacher(self, teacher: DiTSequenceModel):
        """Mirror upstream MFM warm-starts from the base denoiser checkpoint."""
        teacher_vocab = teacher.model.vocab_embed.embedding
        student_vocab = self.x_embed.embedding
        if teacher_vocab.shape == student_vocab.shape:
            student_vocab.copy_(teacher_vocab)
        elif teacher_vocab.shape[0] >= student_vocab.shape[0] and teacher_vocab.shape[1] == student_vocab.shape[1]:
            # Self-conditioned denoisers can have a 2K-wide vocab table; use the
            # first K rows corresponding to the current state input.
            student_vocab.copy_(teacher_vocab[: student_vocab.shape[0]])
        else:
            raise ValueError(
                "Cannot warm-start DNAMFMStudent.x_embed from teacher vocab embedding: "
                f"teacher shape={tuple(teacher_vocab.shape)}, student shape={tuple(student_vocab.shape)}"
            )
        self.x_cond_embed.embedding.copy_(self.x_embed.embedding)

        # Initialize all external time-conditioning branches from the denoiser's
        # learned time embedder.  This is not a semantic one-to-one mapping, but
        # it gives the student a meaningful time basis instead of random weights.
        sigma_sd = teacher.model.sigma_map.state_dict()
        self.s_embedder.load_state_dict(sigma_sd)
        self.t_embedder.load_state_dict(sigma_sd)
        self.t_cond_embedder.load_state_dict(sigma_sd)
        self.s_embedder_second.load_state_dict(sigma_sd)
        if teacher.model.sigma_map_prime is not None:
            sigma_prime_sd = teacher.model.sigma_map_prime.state_dict()
            self.t_embedder_second.load_state_dict(sigma_prime_sd)
        else:
            self.t_embedder_second.load_state_dict(sigma_sd)

        # Copy as much of the actual DiT backbone as possible.  This is the
        # important part for stability: blocks/output layer should not remain
        # random if we want a real teacher initialization.
        teacher_sd = teacher.model.state_dict()
        student_sd = self.model.state_dict()
        for key, value in teacher_sd.items():
            if key in student_sd and student_sd[key].shape == value.shape:
                student_sd[key] = value.detach().clone()
        self.model.load_state_dict(student_sd, strict=False)

        self._mfm_teacher_warm_started = True

    def _conditioned_tokens(self, s, t, x, t_cond, x_cond):
        if x_cond is None:
            raise ValueError("DNAMFMStudent requires x_cond (x_tcond).")
        if t_cond is None:
            raise ValueError("DNAMFMStudent requires t_cond.")
        B, L, _K = x.shape

        x_emb = self.x_embed(x)
        x_cond_emb = self.x_cond_embed(x_cond)

        t_cond_embedded = self.t_cond_embedder(t_cond).reshape(B, 1, -1)
        shift_cond, scale_cond = self.x_cond_adaLN(t_cond_embedded).chunk(2, dim=-1)
        if self.preserve_t_cond_0:
            tc = t_cond.reshape(B, 1, 1)
            shift_cond = shift_cond * tc
            scale_cond = (scale_cond * tc) - 1.0
        x_cond_emb = modulate(x_cond_emb, shift_cond, scale_cond)
        x_tok = x_emb + x_cond_emb

        s_first = self.s_embedder(s).reshape(B, -1)
        t_first = self.t_embedder(t).reshape(B, -1)
        c_first = F.silu(s_first + t_first + t_cond_embedded.reshape(B, -1))

        s_second = self.s_embedder_second(s).reshape(B, -1)
        t_second = self.t_embedder_second(t).reshape(B, -1)
        c_second = F.silu(s_second + t_second + t_cond_embedded.reshape(B, -1))

        return x_tok, c_first, c_second

    def forward_logits(self, x, s, t, *, t_cond, x_cond, class_labels=None, **kwargs):
        """Return conditional denoiser logits z_{s,t}(x; t_cond, x_cond)."""
        _ = (class_labels, kwargs)
        x_tok, c_first, c_second = self._conditioned_tokens(s, t, x, t_cond, x_cond)
        return self.model.forward_hidden_two_stage(
            x_tok, c_first, c_second, encoder_depth=self.encoder_depth
        )

    def Psi_st(
        self,
        x,
        s,
        t,
        *,
        t_cond,
        x_cond,
        class_labels=None,
        flow_temp=1.0,
        return_logits=False,
        return_log_psi=False,
        **kwargs,
    ):
        """Return the conditional meta denoiser Psi_{s,t}(x; t_cond, x_cond)."""
        logits = self.forward_logits(
            x,
            s,
            t,
            t_cond=t_cond,
            x_cond=x_cond,
            class_labels=class_labels,
            **kwargs,
        )
        if return_logits:
            return logits
        log_psi = F.log_softmax(logits / flow_temp, dim=-1)
        if return_log_psi:
            return log_psi
        return log_psi.exp()

    # Lowercase alias to match DiTSequenceModel.psi_st naming when convenient.
    psi_st = Psi_st

    def v(
        self,
        s,
        t,
        x,
        t_cond,
        x_cond,
        class_labels=None,
        flow_temp=1.0,
        eps=1e-8,
        **kwargs,
    ):
        """
        Induced discrete MFM velocity from the conditional denoiser.

        The DiT head is a mean denoiser/logit head, not a velocity head.  With
        the linear discrete interpolant, BaseModel.X(s,t,x,v) then recovers
            X_{s,t}(x) = ((1-t)/(1-s)) x + ((t-s)/(1-s)) Psi_{s,t}(x).
        """
        psi = self.Psi_st(
            x,
            s,
            t,
            t_cond=t_cond,
            x_cond=x_cond,
            class_labels=class_labels,
            flow_temp=flow_temp,
            **kwargs,
        )
        denom = (1.0 - s).clamp_min(float(eps))
        return (psi - x) / _bcast(denom, x)

    def extract_posterior_velocity(
        self,
        s,
        Is,
        xt_cond,
        t_cond,
        *,
        labels=None,
        cfg_scales=None,
        teacher_model=None,
        eps=1e-6,
        omega=0.6,
        checkpoint_type="sit",
    ):
        """Use mfm.losses.losses.extract_posterior_velocity with this model as the default teacher."""
        return _posterior_velocity(
            s,
            Is,
            xt_cond,
            t_cond,
            labels=labels,
            cfg_scales=cfg_scales,
            teacher_model=self if teacher_model is None else teacher_model,
            eps=eps,
            omega=omega,
            checkpoint_type=checkpoint_type,
        )


# Name the intended roles explicitly.  Existing training keeps using
# DiTSequenceModel / MetaDiTSequenceModel, while notebooks can import the clearer
# aliases without changing Lightning wiring.
DNADiagonalDenoiser = DiTSequenceModel
DNADMFMTeacherAdapter = DNAMFMTeacherAdapter
DNADMFMVelocityModel = DNAMFMStudent
MetaDiTSequenceModel = DNAMFMStudent
