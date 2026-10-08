"""DFM-style 1D DiT backbone for DNA sequences.

Ported verbatim from DNA-MFM@187fe7b ``model/dit_seq.py`` (reconstruction in
``scratch_dfm_recon_20260925/model/dit_seq.py``). FlashAttention-3 was never
available in the paper runs (``fa3_func = None``), so attention always uses the
softcap/math path; with ``use_flash_attn=True`` in the saved args the model warns
once and falls back.
"""

import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

fa3_func = None


def _is_dual(t: torch.Tensor) -> bool:
    """Detect torch.func dual tensors; Flash/SDPA kernels do not support JVP reliably."""
    try:
        from torch._C._functorch import is_gradtrackingtensor

        return is_gradtrackingtensor(t)
    except (ImportError, AttributeError):
        pass
    return "FunctionalTensorWrapper" in type(t).__name__ or "dual" in str(type(t)).lower()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    return x * freqs_cos + _rotate_half(x) * freqs_sin


class Rotary1D(nn.Module):
    """DFM DiT-style rotary cache."""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached = {}

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, str(device), str(dtype))
        if key in self._cached:
            return self._cached[key]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(dtype)
        sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(dtype)
        cos[:, :, 2, :, :].fill_(1.0)
        sin[:, :, 2, :, :].fill_(0.0)
        self._cached[key] = (cos, sin)
        return cos, sin


class LayerNorm(nn.Module):
    """DFM DiT LayerNorm: no bias, one learned scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x.to(dtype) * self.weight[None, None, :].to(dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """DFM DiT sinusoidal timestep embedder."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t.view(-1), self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    """DFM DiT vocab embedding that accepts integer tokens or simplex/logit vectors."""

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.embedding[x]
        if x.ndim != 3:
            raise ValueError(f"EmbeddingLayer expects [B,L] or [B,L,V], got shape={tuple(x.shape)}")
        return torch.einsum("blv,ve->ble", x.float(), self.embedding.float()).to(x.dtype)


def _softcap_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softcap: float = -1.0) -> torch.Tensor:
    B, H, S, D = q.shape
    q = q / (D**0.5)
    attn_weights = torch.einsum("bhid,bhjd->bhij", q, k)
    if softcap > 0.0:
        attn_weights = softcap * torch.tanh(attn_weights / softcap)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", attn_probs, v)


class DDiTBlock1D(nn.Module):
    """DFM DDiTBlock adapted to sequence tensors."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
        softcap: float = 50.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "hidden_dim must be divisible by transformer_heads"
        assert self.head_dim % 2 == 0, "RoPE requires an even per-head dimension"
        self.dropout = float(dropout)
        self.use_flash_attn = bool(use_flash_attn)
        self.softcap = float(softcap)

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(mlp_ratio * dim), dim, bias=True),
        )
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _attention(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.unbind(dim=2)
        if self.use_flash_attn and fa3_func is not None and not _is_dual(q):
            return fa3_func(q, k, v, softcap=self.softcap)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if _is_dual(q) or self.softcap > 0.0:
            out = _softcap_attention(q, k, v, softcap=self.softcap)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=False,
            )
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        B, L, D = x.shape
        qkv = self.attn_qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        cos, sin = rotary_cos_sin
        cos = cos.to(device=qkv.device, dtype=qkv.dtype)
        sin = sin.to(device=qkv.device, dtype=qkv.dtype)
        qkv = qkv * cos + _rotate_half(qkv) * sin

        x = self._attention(qkv).contiguous().view(B, L, D)
        x = F.dropout(self.attn_out(x), p=self.dropout, training=self.training)
        x = x_skip + gate_msa * x

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = F.dropout(self.mlp(h), p=self.dropout, training=self.training)
        return x + gate_mlp * h


class DDiTFinalLayer(nn.Module):
    """DFM DDiT final layer: final norm, AdaLN, zero-initialized output."""

    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int, adaLN: bool = True, bias: bool = True):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=bias)
        self.linear.weight.data.zero_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate(x, shift, scale)
        return self.linear(x)


@dataclass
class DiT1DConfig:
    dim: int
    heads: int
    blocks: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    cond_dim: Optional[int] = None
    input_dim: Optional[int] = None
    out_dim: Optional[int] = None
    double_temb: bool = True
    scale_by_sigma: bool = True
    preserve_denoiser: bool = False
    use_flash_attn: bool = True
    softcap: float = 50.0
    frequency_embedding_size: int = 256


class DiT1D(nn.Module):
    """
    DFM DiT-style sequence model.

    This owns the vocab embedding, timestep embedders, DDiT blocks, and final
    output layer adapted to 1D DNA sequences.
    """

    def __init__(self, config: DiT1DConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.dim)
        self.cond_dim = int(config.cond_dim or config.dim)
        self.input_dim = int(config.input_dim or config.out_dim or config.dim)
        self.out_dim = int(config.out_dim or self.input_dim)
        self.double_temb = bool(config.double_temb)
        self.scale_by_sigma = bool(config.scale_by_sigma)
        self.preserve_denoiser = bool(config.preserve_denoiser)
        self.use_flash_attn = bool(config.use_flash_attn)
        if self.use_flash_attn and fa3_func is None:
            warnings.warn(
                "FlashAttention v3 backend is unavailable; falling back to softcap/math attention.",
                RuntimeWarning,
            )
            self.use_flash_attn = False

        self.vocab_embed = EmbeddingLayer(self.hidden_size, self.input_dim)
        self.sigma_map = TimestepEmbedder(self.cond_dim, frequency_embedding_size=config.frequency_embedding_size)
        self.sigma_map_prime = (
            TimestepEmbedder(self.cond_dim, frequency_embedding_size=config.frequency_embedding_size)
            if self.double_temb
            else None
        )
        self.rotary_emb = Rotary1D(dim=self.hidden_size // int(config.heads))
        self.blocks = nn.ModuleList(
            [
                DDiTBlock1D(
                    dim=self.hidden_size,
                    n_heads=int(config.heads),
                    cond_dim=self.cond_dim,
                    mlp_ratio=float(config.mlp_ratio),
                    dropout=float(config.dropout),
                    use_flash_attn=self.use_flash_attn,
                    softcap=float(config.softcap),
                )
                for _ in range(int(config.blocks))
            ]
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size=self.hidden_size,
            out_channels=self.out_dim,
            cond_dim=self.cond_dim,
            adaLN=True,
        )

    def time_cond(self, sigma: torch.Tensor, sigma_prime: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.sigma_map(sigma)
        if sigma_prime is not None:
            if self.sigma_map_prime is not None:
                t_prime_emb = self.sigma_map_prime(sigma_prime)
                if self.preserve_denoiser:
                    t_prime_emb = t_prime_emb + self.sigma_map(torch.zeros_like(sigma))
            else:
                t_prime_emb = self.sigma_map(sigma_prime)
            t_emb = t_emb + t_prime_emb
        return F.silu(t_emb)

    def _run_blocks(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        rotary_cos_sin = self.rotary_emb(x.shape[1], device=x.device, dtype=x.dtype)
        amp_dtype = torch.bfloat16 if self.use_flash_attn else torch.float32
        autocast_enabled = amp_dtype in (torch.float16, torch.bfloat16)
        amp_context = (
            torch.amp.autocast(device_type=x.device.type, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            return self.output_layer(x, c)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prime: Optional[torch.Tensor] = None,
        *,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if c is None:
            c = self.time_cond(sigma, sigma_prime)
        x = self.vocab_embed(x)
        return self._run_blocks(x, c)

    def forward_hidden(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self._run_blocks(x, c)

    def forward_hidden_two_stage(
        self,
        x: torch.Tensor,
        c_first: torch.Tensor,
        c_second: torch.Tensor,
        *,
        encoder_depth: int,
    ) -> torch.Tensor:
        rotary_cos_sin = self.rotary_emb(x.shape[1], device=x.device, dtype=x.dtype)
        ed = max(0, min(int(encoder_depth), len(self.blocks)))
        for i, block in enumerate(self.blocks):
            c = c_first if i < ed else c_second
            x = block(x, rotary_cos_sin, c)
        return self.output_layer(x, c_second)
