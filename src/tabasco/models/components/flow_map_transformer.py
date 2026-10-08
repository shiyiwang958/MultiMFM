"""Flow-map (Meta Flow Map / MFM) student transformer for TABASCO.

A *flow map* over two inner times ``s -> u``, conditioned on a noisy observation
``x_cond`` at level ``t_cond``. The default parametrization predicts an endpoint
and derives velocity:

    v_su(x) = (x1_hat(s, u, x, t_cond, x_cond) - x) / (1 - s)

It also supports direct velocity and endpoint-plus-residual velocity heads for
parametrization experiments.

Inductive biases ported from ``norway/mfm_model.py`` (critical for the flow map):
  * Off-diagonal time enters ONLY through ``du = u - s`` via a zero-at-zero MLP,
    so at ``u == s`` the off-diagonal pathway is *exactly zero* and the model is
    identically the diagonal endpoint predictor (anchors the diagonal under
    off-diagonal training).
  * ``x_cond`` is gated by the ``t_cond`` embedding (zero at ``t_cond == 0``), so
    ``t_cond == 0`` is exactly the unconditional flow map.

JVP note (Stage-2 ESD): run ``torch.func.jvp`` under
``torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`` (fused SDPA lacks forward AD).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tabasco.models.components.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from tabasco.models.components.transformer import Transformer
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


class DiTBlock(nn.Module):
    """adaLN-Zero transformer block (DiT/SiT/norway style).

    The conditioning vector ``c`` produces per-block scale/shift/gate for both the
    attention and MLP sublayers; the modulation projection is zero-initialized so
    the block starts as a near-identity (stable training, conditioning opens up).
    Uses nn.MultiheadAttention (JVP-safe under the MATH SDPA backend).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: Tensor, c: Tensor, key_padding_mask: Tensor) -> Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class ZeroAtZeroTimeMLP(nn.Module):
    """Time embedding that is structurally zero when its scalar input is 0.

    Uses a centered Fourier feature (value minus value-at-0) through a bias-free
    MLP with a zero-initialized last layer, so the whole output starts at zero and
    is exactly zero at input 0 forever (matching norway's ZeroAtZeroTimeMLP).
    """

    def __init__(self, hidden_dim: int, max_len: int = 200):
        super().__init__()
        self.fourier = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=max_len)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.net[2].weight)

    def forward(self, t: Tensor) -> Tensor:
        centered = self.fourier(t) - self.fourier(torch.zeros_like(t))
        return self.net(centered)


class TabascoFlowMap(nn.Module):
    """MFM flow-map student over (coords, atom-type) molecular states."""

    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        cross_attention: bool = True,
        add_sinusoid_posenc: bool = True,
        max_num_atoms: int = 90,
        velocity_parametrization: str = "endpoint",
        eps: float = 1e-3,
        block_conditioning: str = "additive",
        encoder_depth: int | None = None,
        time_encoding_max_len: int = 200,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        if velocity_parametrization not in (
            "endpoint",
            "direct",
            "endpoint_residual",
            "coord_direct_atom_endpoint",
            "coord_endpoint_residual_atom_endpoint",
        ):
            raise ValueError(f"Invalid velocity_parametrization: {velocity_parametrization!r}")
        self.velocity_parametrization = velocity_parametrization
        if block_conditioning not in (
            "additive",
            "adaln",
            "gated",
            "gated_delta",
            "split_gated",
        ):
            raise ValueError(f"Invalid block_conditioning: {block_conditioning!r}")
        self.block_conditioning = block_conditioning
        self.use_delta_conditioning = block_conditioning == "gated_delta"
        self.encoder_depth = (
            max(1, min(num_layers - 1, encoder_depth))
            if encoder_depth is not None and num_layers > 1
            else max(1, min(num_layers - 1, (2 * num_layers) // 3))
            if block_conditioning == "split_gated" and num_layers > 1
            else num_layers
        )
        self.eps = eps

        # inner state x (at time s)
        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_linear_embed = nn.Linear(atom_dim, hidden_dim, bias=False)
        # conditioning observation x_cond (at level t_cond)
        self.cond_coord_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.cond_atom_embed = nn.Linear(atom_dim, hidden_dim, bias=False)
        if self.use_delta_conditioning:
            self.cond_delta_coord_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
            self.cond_delta_atom_embed = nn.Linear(atom_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.cond_delta_coord_embed.weight)
            nn.init.zeros_(self.cond_delta_atom_embed.weight)

        if add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=max_num_atoms
            )

        # time conditioning: standard source-time s; zero-at-zero du=(u-s) and t_cond.
        # ``time_encoding_max_len`` sets the highest Fourier frequency of the s and
        # (u - s) features (angular frequencies span 1..max_len); ESD's d/ds JVP
        # scales with it. Parameter-free, so it does not change the state dict.
        self.time_encoding_max_len = time_encoding_max_len
        self.s_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=time_encoding_max_len)
        self.du_mlp = ZeroAtZeroTimeMLP(hidden_dim, max_len=time_encoding_max_len)
        self.tcond_mlp = ZeroAtZeroTimeMLP(hidden_dim)
        # gate x_cond by the (zero-at-zero) t_cond embedding -> off at t_cond=0.
        self.x_cond_gate = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)

        if block_conditioning in ("additive", "gated", "gated_delta", "split_gated"):
            self.transformer = Transformer(
                dim=hidden_dim, num_heads=num_heads, depth=num_layers
            )
            if block_conditioning in ("gated", "gated_delta", "split_gated"):
                self.block_cond_gates = nn.ModuleList(
                    [nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(num_layers)]
                )
                for gate in self.block_cond_gates:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
        else:  # adaln: conditioning modulates every block (DiT/SiT/norway style)
            self.dit_blocks = nn.ModuleList(
                [DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)]
            )
            self.adaln_final = nn.Linear(hidden_dim, 2 * hidden_dim)
            nn.init.zeros_(self.adaln_final.weight)
            nn.init.zeros_(self.adaln_final.bias)
            self.adaln_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )
        if velocity_parametrization in ("direct", "coord_direct_atom_endpoint"):
            nn.init.zeros_(self.out_coord_linear[-1].weight)
        if velocity_parametrization == "direct":
            nn.init.zeros_(self.out_atom_type_linear[-1].weight)
            nn.init.zeros_(self.out_atom_type_linear[-1].bias)

        if cross_attention:
            self.coord_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                batch_first=True, norm_first=True,
            )
            self.atom_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                batch_first=True, norm_first=True,
            )

        if velocity_parametrization == "endpoint_residual":
            self.residual_coord_linear = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, spatial_dim, bias=False),
            )
            self.residual_atom_type_linear = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(inplace=False),
                nn.Linear(hidden_dim, atom_dim),
            )
            nn.init.zeros_(self.residual_coord_linear[-1].weight)
            nn.init.zeros_(self.residual_atom_type_linear[-1].weight)
            nn.init.zeros_(self.residual_atom_type_linear[-1].bias)
        elif velocity_parametrization == "coord_endpoint_residual_atom_endpoint":
            self.residual_coord_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.SiLU(inplace=False),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.SiLU(inplace=False),
                nn.Linear(hidden_dim, spatial_dim, bias=False),
            )
            nn.init.zeros_(self.residual_coord_mlp[-1].weight)

    @torch.no_grad()
    def _copy_module_expanded(self, dst_module: nn.Module, src_module: nn.Module) -> bool:
        """Copy matching source weights into a same-size or wider destination module.

        This lets a wider student keep the teacher-initialized subnetwork instead of
        training from scratch. Non-overlapping destination weights are zeroed, except
        LayerNorm scales are initialized to 1, so extra hidden channels start inert.
        """
        src_state = src_module.state_dict()
        dst_state = dst_module.state_dict()
        copied_any = False

        def reset_dst(name: str, tensor: Tensor) -> Tensor:
            out = tensor.detach().clone()
            if out.dtype.is_floating_point:
                if name.endswith("norm.weight") or ".norm." in name and name.endswith(".weight"):
                    out.fill_(1.0)
                else:
                    out.zero_()
            return out

        def copy_overlap(name: str, dst: Tensor, src: Tensor) -> Tensor:
            out = reset_dst(name, dst)
            if not out.dtype.is_floating_point or src.ndim != out.ndim:
                if src.shape == out.shape:
                    return src.detach().clone()
                return out

            # MultiheadAttention packs q/k/v along dim 0. Copy each packed block
            # into the corresponding wider block instead of naive top-left copy.
            if name.endswith("in_proj_weight") and src.ndim == 2:
                src_dim = src.shape[1]
                dst_dim = out.shape[1]
                if src.shape[0] == 3 * src_dim and out.shape[0] == 3 * dst_dim:
                    rows = min(src_dim, dst_dim)
                    cols = min(src_dim, dst_dim)
                    for block in range(3):
                        out[block * dst_dim : block * dst_dim + rows, :cols].copy_(
                            src[block * src_dim : block * src_dim + rows, :cols]
                        )
                    return out
            if name.endswith("in_proj_bias") and src.ndim == 1:
                src_dim = src.shape[0] // 3
                dst_dim = out.shape[0] // 3
                if src.shape[0] == 3 * src_dim and out.shape[0] == 3 * dst_dim:
                    width = min(src_dim, dst_dim)
                    for block in range(3):
                        out[block * dst_dim : block * dst_dim + width].copy_(
                            src[block * src_dim : block * src_dim + width]
                        )
                    return out

            slices = tuple(slice(0, min(d, s)) for d, s in zip(out.shape, src.shape))
            out[slices].copy_(src[slices])
            return out

        new_state = {}
        for name, dst_tensor in dst_state.items():
            src_tensor = src_state.get(name)
            if src_tensor is None:
                new_state[name] = reset_dst(name, dst_tensor)
                continue
            new_state[name] = copy_overlap(name, dst_tensor, src_tensor.to(dst_tensor.device))
            copied_any = True
        dst_module.load_state_dict(new_state, strict=False)
        return copied_any

    @torch.no_grad()
    def init_from_base(
        self, teacher_net: nn.Module, *, copy_output_heads: bool = True
    ) -> "TabascoFlowMap":
        """Copy the teacher's endpoint-predictor weights (norway's init_from_base).

        The backbone mirrors the teacher ``TransformerModule`` (continuous_linear atom
        mode, cross-attention), so the student starts as the teacher's unconditional
        endpoint predictor; training then opens the (zero-init) conditioning + du
        pathways. The new modules (cond embeds, du/tcond MLPs, x_cond gate) stay at init.
        """
        pairs = [
            ("linear_embed", "linear_embed"),
            ("atom_linear_embed", "atom_linear_embed"),
            ("cond_coord_embed", "linear_embed"),
            ("cond_atom_embed", "atom_linear_embed"),
            ("positional_encoding", "positional_encoding"),
            ("transformer", "transformer"),
            ("coord_cross_attention", "coord_cross_attention"),
            ("atom_cross_attention", "atom_cross_attention"),
        ]
        if copy_output_heads:
            pairs.extend(
                [
                    ("out_coord_linear", "out_coord_linear"),
                    ("out_atom_type_linear", "out_atom_type_linear"),
                ]
            )
        copied = []
        for dst, src in pairs:
            if hasattr(self, dst) and hasattr(teacher_net, src):
                if self._copy_module_expanded(getattr(self, dst), getattr(teacher_net, src)):
                    copied.append(dst)
        if self.velocity_parametrization in ("direct", "coord_direct_atom_endpoint"):
            nn.init.zeros_(self.out_coord_linear[-1].weight)
        if self.velocity_parametrization == "direct":
            nn.init.zeros_(self.out_atom_type_linear[-1].weight)
            nn.init.zeros_(self.out_atom_type_linear[-1].bias)
        print(f"init_from_base copied: {copied}")
        return self

    def forward(
        self,
        coords_x: Tensor,
        atomics_x: Tensor,
        coords_cond: Tensor,
        atomics_cond: Tensor,
        padding_mask: Tensor,
        s: Tensor,
        u: Tensor,
        t_cond: Tensor,
        return_atom_logits: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Return (coord_velocity, atom_velocity) for the flow map ``s -> u``.

        With ``return_atom_logits`` also return the meta-denoiser logits
        ``z_{s,u}`` (``Psi_{s,u} = softmax(z_{s,u})``); only defined for
        parametrizations whose atom velocity is a pure softmax endpoint.
        """
        if return_atom_logits and self.velocity_parametrization in ("direct", "endpoint_residual"):
            raise ValueError(
                "atom logits are not the meta denoiser for "
                f"velocity_parametrization={self.velocity_parametrization!r}"
            )
        real_mask = 1 - padding_mask.int()
        n_atoms = coords_x.shape[1]

        embed_coords = self.linear_embed(coords_x)
        embed_atoms = self.atom_linear_embed(atomics_x.float())

        # time conditioning (per-sample -> broadcast per token).
        embed_s = self.s_encoding(s)
        embed_du = self.du_mlp(u - s)            # zero when u == s
        embed_tc = self.tcond_mlp(t_cond)        # zero when t_cond == 0
        cond_vec = embed_s + embed_du + embed_tc  # (B, hidden)
        cond_vec_enc = embed_s + embed_tc
        cond_vec_dec = self.s_encoding(u) + embed_du + embed_tc

        # x_cond gated by t_cond embedding (off at t_cond == 0; bias-free gate).
        coord_gate, atom_gate = self.x_cond_gate(embed_tc).chunk(2, dim=-1)
        embed_cond_coords = coord_gate.unsqueeze(1) * self.cond_coord_embed(coords_cond)
        embed_cond_atoms = atom_gate.unsqueeze(1) * self.cond_atom_embed(atomics_cond.float())
        if self.use_delta_conditioning:
            coord_delta = coords_cond - coords_x
            atom_delta = atomics_cond.float() - atomics_x.float()
            embed_cond_coords = embed_cond_coords + coord_gate.unsqueeze(
                1
            ) * self.cond_delta_coord_embed(coord_delta)
            embed_cond_atoms = embed_cond_atoms + atom_gate.unsqueeze(
                1
            ) * self.cond_delta_atom_embed(atom_delta)

        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(
                batch_size=coords_x.shape[0], seq_len=n_atoms
            )
        else:
            embed_posenc = torch.zeros_like(embed_coords)

        # Additive/gated modes add time conditioning to tokens. Gated mode also
        # opens zero-init per-block conditional residual scales after teacher init.
        h_in = embed_coords + embed_atoms + embed_cond_coords + embed_cond_atoms + embed_posenc
        if self.block_conditioning in ("additive", "gated", "gated_delta"):
            h_in = h_in + cond_vec.unsqueeze(1)
        h_in = h_in * real_mask.unsqueeze(-1)

        if self.block_conditioning == "additive":
            h_out = self.transformer(h_in, padding_mask=padding_mask)
        elif self.block_conditioning in ("gated", "gated_delta"):
            h_out = h_in
            for layer, gate in zip(self.transformer.layers, self.block_cond_gates):
                attn_scale, ff_scale = gate(cond_vec).chunk(2, dim=-1)
                attn_output = layer.attn_block(
                    h_out, key_padding_mask=padding_mask
                )
                h_out = h_out + (1.0 + attn_scale.unsqueeze(1)) * attn_output
                ff_output = layer.ff_block(h_out)
                h_out = h_out + (1.0 + ff_scale.unsqueeze(1)) * ff_output
            h_out = self.transformer.norm(h_out)
        elif self.block_conditioning == "split_gated":
            h_out = h_in
            for idx, (layer, gate) in enumerate(
                zip(self.transformer.layers, self.block_cond_gates)
            ):
                layer_cond = cond_vec_enc if idx < self.encoder_depth else cond_vec_dec
                attn_scale, ff_scale = gate(layer_cond).chunk(2, dim=-1)
                h_attn = (h_out + layer_cond.unsqueeze(1)) * real_mask.unsqueeze(-1)
                attn_output = layer.attn_block(
                    h_attn, key_padding_mask=padding_mask
                )
                h_out = (
                    h_out + (1.0 + attn_scale.unsqueeze(1)) * attn_output
                ) * real_mask.unsqueeze(-1)
                h_ff = (h_out + layer_cond.unsqueeze(1)) * real_mask.unsqueeze(-1)
                ff_output = layer.ff_block(h_ff)
                h_out = (
                    h_out + (1.0 + ff_scale.unsqueeze(1)) * ff_output
                ) * real_mask.unsqueeze(-1)
            h_out = self.transformer.norm(h_out)
        else:
            h_out = h_in
            for block in self.dit_blocks:
                h_out = block(h_out, cond_vec, padding_mask) * real_mask.unsqueeze(-1)
            scale, shift = self.adaln_final(cond_vec).chunk(2, dim=-1)
            h_out = self.adaln_norm(h_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h_out = h_out * real_mask.unsqueeze(-1)

        if self.cross_attention:
            h_coord = self.coord_cross_attention(
                h_out, h_in, tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            coord_head = self.out_coord_linear(h_coord)
            h_atom = self.atom_cross_attention(
                h_out, h_in, tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            atom_head = self.out_atom_type_linear(h_atom)
        else:
            h_coord = h_out
            h_atom = h_out
            coord_head = self.out_coord_linear(h_out)
            atom_head = self.out_atom_type_linear(h_out)

        if self.velocity_parametrization == "direct":
            coords_v = mask_and_zero_com(coord_head, padding_mask)
            atomics_v = apply_mask(atom_head, padding_mask)
            return coords_v, atomics_v

        # v(I_s) = (E[x1 | I_s, cond] - I_s) / (1 - s); softmax keeps the atom
        # endpoint on the simplex (matches the DFM teacher's parametrization).
        coord_end = mask_and_zero_com(coord_head, padding_mask)
        atom_end = apply_mask(torch.softmax(atom_head, dim=-1), padding_mask)
        one_minus_s = (1.0 - s).clamp_min(self.eps).view(
            -1, *([1] * (coords_x.ndim - 1))
        )
        if self.velocity_parametrization == "coord_direct_atom_endpoint":
            coords_v = mask_and_zero_com(coord_head, padding_mask)
        else:
            coords_v = mask_and_zero_com(
                (coord_end - coords_x) / one_minus_s, padding_mask
            )
        atomics_v = apply_mask((atom_end - atomics_x) / one_minus_s, padding_mask)

        if self.velocity_parametrization == "endpoint_residual":
            coords_res = mask_and_zero_com(self.residual_coord_linear(h_coord), padding_mask)
            atomics_res = apply_mask(self.residual_atom_type_linear(h_atom), padding_mask)
            coords_v = mask_and_zero_com(coords_v + coords_res, padding_mask)
            atomics_v = apply_mask(atomics_v + atomics_res, padding_mask)
        elif self.velocity_parametrization == "coord_endpoint_residual_atom_endpoint":
            coords_res = mask_and_zero_com(self.residual_coord_mlp(h_coord), padding_mask)
            coords_v = mask_and_zero_com(coords_v + coords_res, padding_mask)

        if return_atom_logits:
            return coords_v, atomics_v, atom_head
        return coords_v, atomics_v
