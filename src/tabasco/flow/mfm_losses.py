"""Stage-1 diagonal GLASS-distillation loss for the TABASCO MFM student.

The student flow map ``v(s, u, x, t_cond, x_cond)`` is trained, on the diagonal
``s == u``, to match the GLASS posterior velocity of the teacher DFM:

    target = extract_glass_velocity(teacher, I_s, Y_{t_cond}, s, t_cond)

over the full ``t_cond`` range (no data-FM term). States are built with the
teacher's own interpolant (``_create_path``) so they sit exactly on the linear SI
that the GLASS closed form assumes.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabasco.sample.glass import extract_glass_velocity
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com


def _masked_mse(
    pred: Tensor, target: Tensor, real_mask: Tensor, sample_weight: Tensor | None = None
) -> Tensor:
    """Mean squared error over non-padded entries (mask is 1 for real atoms).

    ``sample_weight`` (shape ``(B,)``) optionally reweights each molecule. With the
    endpoint parametrization, passing ``(1 - s)**2`` turns velocity-MSE into
    endpoint-space MSE (cancels the ``1/(1-s)`` amplification near ``s -> 1``).
    """
    w = real_mask.unsqueeze(-1)
    if sample_weight is not None:
        w = w * sample_weight.view(-1, *([1] * (pred.ndim - 1)))
    se = ((pred - target) ** 2) * w
    return se.sum() / w.sum().clamp_min(1.0) / pred.shape[-1]


def _masked_soft_cross_entropy(
    logits: Tensor, target_probs: Tensor, real_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Soft-label cross entropy ``-sum_k p_k log softmax(z)_k``, mean over real atoms.

    Returns ``(ce, kl)``; ``kl = ce - H(p)`` has the same gradient and is the
    diagnostic that reaches zero at the optimum.
    """
    log_q = torch.log_softmax(logits.float(), dim=-1)
    p = target_probs.float()
    ce_atom = -(p * log_q).sum(dim=-1)
    ent_atom = -torch.special.xlogy(p, p).sum(dim=-1)
    w = real_mask.to(ce_atom.dtype)
    denom = w.sum().clamp_min(1.0)
    ce = (ce_atom * w).sum() / denom
    kl = ((ce_atom - ent_atom).detach() * w).sum() / denom
    return ce, kl


def _student_has_atom_logits(student) -> bool:
    return getattr(student, "velocity_parametrization", "endpoint") not in (
        "direct",
        "endpoint_residual",
    )


def esd_logit_space_target(
    teacher_logits: Tensor,
    student_logits: Tensor,
    jvp_logits: Tensor,
    s: Tensor,
    u: Tensor,
    real_mask: Tensor,
    *,
    shift_clip: float = 5.0,
) -> tuple[Tensor, dict]:
    """Simplex-valued ESD target for the atom meta denoiser (paper App. C.2).

    The Eulerian identity (37) reads ``Psi_{s,u} = psi + kinv * D_s Psi_{s,u}`` with
    ``kinv = (u - s)(1 - s)/(1 - u)`` and ``D_s = d_s + b_s . grad``. For
    ``Psi = softmax(z)``, ``D_s Psi = Psi * delta`` with
    ``delta = D_s z - <Psi, D_s z>``, so ``Psi * (1 - kinv * delta) = psi`` and

        T_ESD = softmax(z_teacher - log(1 - kinv * delta)).

    The raw right side of (37) sums to one but can have negative entries for an
    imperfect student; the log shift is clipped to ``[-shift_clip, shift_clip]``
    (which also covers ``1 - kinv * delta <= 0``).
    """
    with torch.no_grad():
        psi_su = torch.softmax(student_logits.float(), dim=-1)
        dz = jvp_logits.float()
        delta = dz - (psi_su * dz).sum(dim=-1, keepdim=True)
        kinv = ((u - s) * (1.0 - s) / (1.0 - u).clamp_min(1e-6)).view(-1, 1, 1)
        r = kinv * delta
        one_minus_r = 1.0 - r
        bound = math.exp(shift_clip)
        shift = -torch.log(one_minus_r.clamp(min=1.0 / bound, max=bound))
        target = torch.softmax(teacher_logits.float() + shift, dim=-1)

        teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
        raw = teacher_probs + r * psi_su
        w = real_mask.to(r.dtype)
        n_entries = (w.sum() * r.shape[-1]).clamp_min(1.0)
        n_atoms = w.sum().clamp_min(1.0)
        clipped = (one_minus_r < 1.0 / bound) | (one_minus_r > bound)
        stats = {
            "esd_target_nonpos_frac": float(((one_minus_r <= 0) * w.unsqueeze(-1)).sum() / n_entries),
            "esd_target_clip_frac": float((clipped * w.unsqueeze(-1)).sum() / n_entries),
            # Probability mass of the (clipped) target on clipped classes, and the
            # negative mass of the raw Eq.(37) target, per atom.
            "esd_target_clip_mass": float(((target * clipped).sum(-1) * w).sum() / n_atoms),
            "esd_raw_target_neg_mass": float(((-raw).clamp_min(0).sum(-1) * w).sum() / n_atoms),
            "esd_raw_target_neg_atom_frac": float(
                (((raw.min(dim=-1).values < -1e-6).to(w.dtype)) * w).sum() / n_atoms
            ),
            "esd_raw_target_min": float(raw.masked_fill(w.unsqueeze(-1) == 0, 1.0).min()),
            "esd_kinv_delta_absmax": float((r.abs() * w.unsqueeze(-1)).max()),
        }
    return target, stats


def _masked_adaptive_loss(
    pred: Tensor,
    target: Tensor,
    real_mask: Tensor,
    *,
    p: float = 0.5,
    c: float = 1.0,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Adaptive pseudo-Huber loss (ported from MFM ``adaptive_loss``), masked.

    Per molecule: ``delta_sq = sum_atoms ||pred - target||^2`` (masked), then
    ``w = 1/(delta_sq + c)^p`` (detached) down-weights large-error outliers — for
    ``p=0.5`` the loss grows like ``|delta|`` in the tail (Huber-like), stabilizing
    the high-magnitude JVP targets in the ESD term. ``sample_weight`` (e.g.
    ``(1-s)**2``) folds endpoint-space weighting into the per-molecule error.
    """
    w = real_mask.unsqueeze(-1)
    se = ((pred - target) ** 2) * w
    delta_sq = se.flatten(1).sum(dim=1)  # (B,)
    if sample_weight is not None:
        delta_sq = delta_sq * sample_weight
    weight = (1.0 / (delta_sq + c) ** p).detach()
    return (delta_sq * weight).mean()


def _masked_delta_sq(
    pred: Tensor,
    target: Tensor,
    real_mask: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Per-molecule masked squared error summed over atoms/features."""
    se = ((pred - target) ** 2) * real_mask.unsqueeze(-1)
    delta_sq = se.flatten(1).sum(dim=1)
    if sample_weight is not None:
        delta_sq = delta_sq * sample_weight
    return delta_sq


def _masked_direction_loss(
    pred: Tensor,
    target: Tensor,
    real_mask: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Mean ``1 - cosine`` over molecules, with padding removed."""
    w = real_mask.unsqueeze(-1).to(pred.dtype)
    pred_flat = (pred * w).flatten(1)
    target_flat = (target * w).flatten(1)
    pred_norm = pred_flat.norm(dim=1)
    target_norm = target_flat.norm(dim=1)
    valid = (pred_norm > eps) & (target_norm > eps)
    if not torch.any(valid):
        zero = pred.sum() * 0.0
        return zero, zero
    cosine = (pred_flat[valid] * target_flat[valid]).sum(dim=1) / (
        pred_norm[valid] * target_norm[valid]
    ).clamp_min(eps)
    return (1.0 - cosine).mean(), cosine.mean()


def _masked_rms_per_sample(x: Tensor, real_mask: Tensor) -> Tensor:
    w = real_mask.unsqueeze(-1).to(x.dtype)
    denom = (real_mask.sum(dim=1) * x.shape[-1]).clamp_min(1.0)
    return (((x.square() * w).flatten(1).sum(dim=1) / denom).clamp_min(1e-12)).sqrt()


def _normalized_random_condition_delta(
    cond: TensorDict,
    real_mask: Tensor,
    eps: float,
    *,
    perturb_coords: bool = True,
    perturb_atomics: bool = True,
) -> TensorDict:
    """Random masked condition perturbation with per-active-modality RMS ``eps``."""
    if perturb_coords:
        dcoords = torch.randn_like(cond["coords"]) * real_mask.unsqueeze(-1).to(
            cond["coords"].dtype
        )
        dcoords = mask_and_zero_com(dcoords, cond["padding_mask"])
        dcoords = dcoords / _masked_rms_per_sample(dcoords, real_mask).view(
            -1, *([1] * (dcoords.ndim - 1))
        ) * eps
    else:
        dcoords = torch.zeros_like(cond["coords"])
    if perturb_atomics:
        datomics = torch.randn_like(cond["atomics"]) * real_mask.unsqueeze(-1).to(
            cond["atomics"].dtype
        )
        datomics = apply_mask(datomics, cond["padding_mask"])
        datomics = datomics / _masked_rms_per_sample(datomics, real_mask).view(
            -1, *([1] * (datomics.ndim - 1))
        ) * eps
    else:
        datomics = torch.zeros_like(cond["atomics"])
    return TensorDict(
        {
            "coords": dcoords,
            "atomics": datomics,
            "padding_mask": cond["padding_mask"],
        },
        batch_size=cond.batch_size,
    )


def _add_condition_delta(cond: TensorDict, delta: TensorDict) -> TensorDict:
    return TensorDict(
        {
            "coords": mask_and_zero_com(
                cond["coords"] + delta["coords"], cond["padding_mask"]
            ),
            "atomics": apply_mask(
                cond["atomics"] + delta["atomics"], cond["padding_mask"]
            ),
            "padding_mask": cond["padding_mask"],
        },
        batch_size=cond.batch_size,
    )


def _multimodal_masked_adaptive_loss(
    pred_coords: Tensor,
    pred_atomics: Tensor,
    target_coords: Tensor,
    target_atomics: Tensor,
    real_mask: Tensor,
    *,
    p: float = 0.5,
    c: float = 0.01,
    sample_weight: Tensor | None = None,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
) -> Tensor:
    """MFM adaptive loss over coordinates and atom states jointly."""
    delta_sq = coord_weight * _masked_delta_sq(
        pred_coords, target_coords, real_mask, sample_weight
    ) + atom_weight * _masked_delta_sq(
        pred_atomics, target_atomics, real_mask, sample_weight
    )
    weight = (1.0 / (delta_sq + c) ** p).detach()
    return (delta_sq * weight).mean()


def _clip_sample_l2(x: Tensor, real_mask: Tensor, max_norm: float) -> Tensor:
    """Clip each molecule's masked L2 norm to ``max_norm``."""
    if max_norm <= 0.0:
        return x
    w = real_mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * w
    flat_norm = x_masked.flatten(1).norm(dim=1).clamp_min(1e-12)
    scale = (max_norm / flat_norm).clamp(max=1.0)
    return x_masked * scale.view(-1, *([1] * (x.ndim - 1)))


def build_diagonal_batch(
    teacher,
    x1: TensorDict,
    s: Tensor,
    t_cond: Tensor,
    *,
    noise_s: TensorDict | None = None,
    noise_cond: TensorDict | None = None,
) -> tuple[TensorDict, TensorDict]:
    """Build the inner state ``I_s`` and conditioning ``Y_{t_cond}`` for endpoints x1.

    Both use the teacher's interpolant via ``_create_path`` so the linear-SI / GLASS
    assumptions hold. ``noise_s`` (for I_s) and ``noise_cond`` are independent.
    """
    path_s = teacher._create_path(x1, t=s, noise_batch=noise_s)
    path_cond = teacher._create_path(x1, t=t_cond, noise_batch=noise_cond)
    return path_s.x_t, path_cond.x_t


def _clone_state(state: TensorDict) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"].clone(),
            "atomics": state["atomics"].clone(),
            "padding_mask": state["padding_mask"].clone(),
        },
        batch_size=state.batch_size,
    )


def _repeat_state_batch(state: TensorDict, repeats: int) -> TensorDict:
    return TensorDict(
        {
            "coords": state["coords"].repeat_interleave(repeats, dim=0),
            "atomics": state["atomics"].repeat_interleave(repeats, dim=0),
            "padding_mask": state["padding_mask"].repeat_interleave(repeats, dim=0),
        },
        batch_size=state["padding_mask"].shape[0] * repeats,
    )


def _coord_spread_by_group(
    coords: Tensor,
    real_mask: Tensor,
    group_size: int,
) -> Tensor:
    """Coordinate spread per repeated condition group.

    Matches the diagnostic's coordinate-spread notion: average coordinate std
    over atoms/dimensions for each condition, with repeated posterior samples
    adjacent in the batch.
    """
    if group_size <= 1:
        raise ValueError("group_size must be > 1 for coordinate spread")
    if coords.shape[0] % group_size != 0:
        raise ValueError(
            f"batch={coords.shape[0]} must be divisible by group_size={group_size}"
        )
    n_groups = coords.shape[0] // group_size
    grouped = coords.reshape(n_groups, group_size, *coords.shape[1:])
    grouped_mask = real_mask.reshape(n_groups, group_size, -1)[:, 0].to(coords.dtype)
    std = grouped.std(dim=1, correction=0)
    denom = grouped_mask.sum(dim=1).clamp_min(1.0) * coords.shape[-1]
    return (std * grouped_mask.unsqueeze(-1)).sum(dim=(1, 2)) / denom


def diagonal_distill_loss(
    student,
    teacher,
    x1: TensorDict,
    s: Tensor,
    t_cond: Tensor,
    *,
    noise_s: TensorDict | None = None,
    noise_cond: TensorDict | None = None,
    endpoint_space_loss: bool = False,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 0.01,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    direction_weight: float = 0.0,
    condition_jvp_weight: float = 0.0,
    condition_jvp_direction_weight: float = 0.0,
    condition_jvp_eps: float = 0.03,
    condition_jvp_mode: str = "both",
    atom_loss_type: str = "mse",
) -> tuple[Tensor, dict]:
    """Diagonal GLASS-distillation loss for one batch.

    Args:
        student: ``TabascoFlowMap``.
        teacher: trained ``FlowMatchingModel`` (linear DFM atoms).
        x1: clean endpoints ``{coords, atomics, padding_mask}``.
        s, t_cond: shape ``(B,)`` inner time and conditioning level.
        endpoint_space_loss: weight by ``(1-s)**2`` so the velocity-MSE becomes
            endpoint-space MSE (recommended with the endpoint parametrization).
        atom_loss_type: ``"mse"`` matches atom velocities; ``"ce"`` matches the
            atom meta denoiser to the teacher's ``psi_{t*}(S)`` with soft-label
            cross entropy (Prop. 1 / Eq. 13). Coordinates always use MSE.

    Returns:
        (loss, components) where components has coord/atom L2 for logging.
    """
    i_s, cond = build_diagonal_batch(
        teacher, x1, s, t_cond, noise_s=noise_s, noise_cond=noise_cond
    )

    return diagonal_state_distill_loss(
        student,
        teacher,
        i_s,
        cond,
        s,
        t_cond,
        endpoint_space_loss=endpoint_space_loss,
        loss_type=loss_type,
        adaptive_p=adaptive_p,
        adaptive_c=adaptive_c,
        coord_weight=coord_weight,
        atom_weight=atom_weight,
        direction_weight=direction_weight,
        condition_jvp_weight=condition_jvp_weight,
        condition_jvp_direction_weight=condition_jvp_direction_weight,
        condition_jvp_eps=condition_jvp_eps,
        condition_jvp_mode=condition_jvp_mode,
        atom_loss_type=atom_loss_type,
    )


def diagonal_state_distill_loss(
    student,
    teacher,
    i_s: TensorDict,
    cond: TensorDict,
    s: Tensor,
    t_cond: Tensor,
    *,
    target: TensorDict | None = None,
    endpoint_space_loss: bool = False,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 0.01,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    direction_weight: float = 0.0,
    condition_jvp_weight: float = 0.0,
    condition_jvp_direction_weight: float = 0.0,
    condition_jvp_eps: float = 0.03,
    condition_jvp_mode: str = "both",
    atom_loss_type: str = "mse",
) -> tuple[Tensor, dict]:
    """Diagonal GLASS-distillation loss on an explicit auxiliary state.

    This is the same supervised diagonal target as ``diagonal_distill_loss``, but
    lets callers choose the conditioning state and the query state ``i_s``. It is
    useful for generated-condition training where there is no clean endpoint
    paired with ``x_cond``.
    """
    if atom_loss_type not in ("mse", "ce"):
        raise ValueError(f"Unknown atom_loss_type={atom_loss_type!r}.")
    if atom_loss_type == "ce" and loss_type != "mse":
        raise ValueError("atom_loss_type='ce' requires loss_type='mse' for coordinates")
    # Teacher/student atom logits are always fetched when available so the
    # denoiser KL is logged for both the MSE and CE objectives.
    with_logits = target is None and _student_has_atom_logits(student)
    if atom_loss_type == "ce" and not with_logits:
        raise ValueError(
            "atom_loss_type='ce' needs GLASS teacher logits (target=None) and a "
            "softmax-endpoint atom parametrization"
        )
    teacher_logits = None
    if target is None:
        if with_logits:
            target, teacher_logits = extract_glass_velocity(
                teacher, i_s, cond, s, t_cond, return_atom_logits=True
            )
        else:
            target = extract_glass_velocity(teacher, i_s, cond, s, t_cond)

    # Student diagonal prediction v(s, s, I_s, t_cond, Y_tcond).
    student_out = student(
        i_s["coords"],
        i_s["atomics"],
        cond["coords"],
        cond["atomics"],
        i_s["padding_mask"],
        s,
        s,
        t_cond,
        return_atom_logits=with_logits,
    )
    coords_v, atomics_v = student_out[0], student_out[1]

    real_mask = (1 - i_s["padding_mask"].int()).to(coords_v.dtype)
    sample_weight = ((1.0 - s) ** 2) if endpoint_space_loss else None
    coord_loss = _masked_mse(coords_v, target["coords"], real_mask, sample_weight)
    atom_loss = _masked_mse(atomics_v, target["atomics"], real_mask, sample_weight)
    zero = coord_loss.detach() * 0.0
    atom_ce, atom_kl = zero, zero
    if with_logits:
        atom_ce, atom_kl = _masked_soft_cross_entropy(
            student_out[2], torch.softmax(teacher_logits.float(), dim=-1), real_mask
        )
    if atom_loss_type == "ce":
        loss = coord_weight * coord_loss + atom_weight * atom_ce
    elif loss_type == "adaptive":
        loss = _multimodal_masked_adaptive_loss(
            coords_v,
            atomics_v,
            target["coords"],
            target["atomics"],
            real_mask,
            p=adaptive_p,
            c=adaptive_c,
            sample_weight=sample_weight,
            coord_weight=coord_weight,
            atom_weight=atom_weight,
        )
    elif loss_type == "mse":
        loss = coord_weight * coord_loss + atom_weight * atom_loss
    else:
        raise ValueError(f"Unknown diagonal loss_type={loss_type!r}.")

    coord_direction_loss, coord_cosine = _masked_direction_loss(
        coords_v, target["coords"], real_mask
    )
    atom_direction_loss, atom_cosine = _masked_direction_loss(
        atomics_v, target["atomics"], real_mask
    )
    direction_loss = coord_weight * coord_direction_loss + atom_weight * atom_direction_loss
    if direction_weight > 0.0:
        loss = loss + direction_weight * direction_loss

    zero = loss.detach() * 0.0
    condition_jvp_loss = zero
    condition_jvp_direction_loss = zero
    condition_jvp_coord_cosine = zero
    condition_jvp_atom_cosine = zero
    condition_jvp_coord_l2 = zero
    condition_jvp_atom_l2 = zero
    if condition_jvp_weight > 0.0 or condition_jvp_direction_weight > 0.0:
        if condition_jvp_eps <= 0.0:
            raise ValueError("--diag-condition-jvp-eps must be positive")
        if condition_jvp_mode not in ("both", "coords", "atomics"):
            raise ValueError(
                "condition_jvp_mode must be one of: both, coords, atomics"
            )
        delta_cond = _normalized_random_condition_delta(
            cond,
            real_mask,
            condition_jvp_eps,
            perturb_coords=condition_jvp_mode in ("both", "coords"),
            perturb_atomics=condition_jvp_mode in ("both", "atomics"),
        )
        pert_cond = _add_condition_delta(cond, delta_cond)
        target_pert = extract_glass_velocity(teacher, i_s, pert_cond, s, t_cond)
        coords_v_pert, atomics_v_pert = student(
            i_s["coords"],
            i_s["atomics"],
            pert_cond["coords"],
            pert_cond["atomics"],
            i_s["padding_mask"],
            s,
            s,
            t_cond,
        )
        target_dc = (target_pert["coords"] - target["coords"]) / condition_jvp_eps
        target_da = (target_pert["atomics"] - target["atomics"]) / condition_jvp_eps
        pred_dc = (coords_v_pert - coords_v) / condition_jvp_eps
        pred_da = (atomics_v_pert - atomics_v) / condition_jvp_eps
        condition_jvp_coord_l2 = _masked_mse(pred_dc, target_dc, real_mask)
        condition_jvp_atom_l2 = _masked_mse(pred_da, target_da, real_mask)
        condition_jvp_loss = (
            coord_weight * condition_jvp_coord_l2
            + atom_weight * condition_jvp_atom_l2
        )
        condition_jvp_coord_dir, condition_jvp_coord_cosine = _masked_direction_loss(
            pred_dc, target_dc, real_mask
        )
        condition_jvp_atom_dir, condition_jvp_atom_cosine = _masked_direction_loss(
            pred_da, target_da, real_mask
        )
        condition_jvp_direction_loss = (
            coord_weight * condition_jvp_coord_dir
            + atom_weight * condition_jvp_atom_dir
        )
        loss = (
            loss
            + condition_jvp_weight * condition_jvp_loss
            + condition_jvp_direction_weight * condition_jvp_direction_loss
        )

    components = {
        "coord_l2": float(coord_loss.detach()),
        "atom_l2": float(atom_loss.detach()),
        "atom_ce": float(atom_ce.detach()),
        "atom_kl": float(atom_kl.detach()),
        "coord_direction_loss": float(coord_direction_loss.detach()),
        "atom_direction_loss": float(atom_direction_loss.detach()),
        "coord_cosine": float(coord_cosine.detach()),
        "atom_cosine": float(atom_cosine.detach()),
        "direction_loss": float(direction_loss.detach()),
        "condition_jvp_coord_l2": float(condition_jvp_coord_l2.detach()),
        "condition_jvp_atom_l2": float(condition_jvp_atom_l2.detach()),
        "condition_jvp_loss": float(condition_jvp_loss.detach()),
        "condition_jvp_direction_loss": float(condition_jvp_direction_loss.detach()),
        "condition_jvp_coord_cosine": float(condition_jvp_coord_cosine.detach()),
        "condition_jvp_atom_cosine": float(condition_jvp_atom_cosine.detach()),
        "diag_loss": float(loss.detach()),
        "target_coord_norm": float(target["coords"].norm().detach()),
        "target_atom_norm": float(target["atomics"].norm().detach()),
    }
    return loss, components


def glass_posterior_diagonal_distill_loss(
    student,
    teacher,
    cond: TensorDict,
    t_cond: Tensor,
    s: Tensor,
    *,
    posterior_steps: int = 64,
    noise_init: TensorDict | None = None,
    endpoint_space_loss: bool = False,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 0.01,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    direction_weight: float = 0.0,
    condition_jvp_weight: float = 0.0,
    condition_jvp_direction_weight: float = 0.0,
    condition_jvp_eps: float = 0.03,
    condition_jvp_mode: str = "both",
    atom_loss_type: str = "mse",
) -> tuple[Tensor, dict]:
    """Distill diagonal velocity on a GLASS posterior-path state.

    ``cond`` may be a generated production state. We draw a prior state, run the
    teacher GLASS ODE only up to the requested inner time ``s``, and supervise the
    instantaneous diagonal velocity at that point. The loss itself contains no
    final rollout matching term.
    """
    if posterior_steps <= 0:
        raise ValueError("posterior_steps must be positive")
    if s.ndim != 1 or s.numel() == 0:
        raise ValueError("s must be a non-empty 1D tensor")
    s_target = float(s[0].detach().cpu())
    if not torch.allclose(s, torch.full_like(s, s_target), atol=1e-6):
        raise ValueError("glass_posterior_diagonal_distill_loss expects scalar s per batch")

    mask = cond["padding_mask"]
    state = _clone_state(noise_init) if noise_init is not None else teacher._sample_noise_like_batch(cond)
    if s_target > 0.0:
        n_steps = max(1, int(round(posterior_steps * s_target)))
        ds_val = s_target / float(n_steps)
        with torch.no_grad():
            for i in range(n_steps):
                s_cur = torch.full_like(t_cond, i * ds_val)
                vel = extract_glass_velocity(teacher, state, cond, s_cur, t_cond)
                state["coords"] = mask_and_zero_com(
                    state["coords"] + ds_val * vel["coords"], mask
                )
                state["atomics"] = apply_mask(
                    state["atomics"] + ds_val * vel["atomics"], mask
                )

    return diagonal_state_distill_loss(
        student,
        teacher,
        state,
        cond,
        s,
        t_cond,
        endpoint_space_loss=endpoint_space_loss,
        loss_type=loss_type,
        adaptive_p=adaptive_p,
        adaptive_c=adaptive_c,
        coord_weight=coord_weight,
        atom_weight=atom_weight,
        direction_weight=direction_weight,
        condition_jvp_weight=condition_jvp_weight,
        condition_jvp_direction_weight=condition_jvp_direction_weight,
        condition_jvp_eps=condition_jvp_eps,
        condition_jvp_mode=condition_jvp_mode,
        atom_loss_type=atom_loss_type,
    )


def esd_consistency_loss(
    student,
    teacher,
    x1: TensorDict,
    s: Tensor,
    u: Tensor,
    t_cond: Tensor,
    *,
    noise_s: TensorDict | None = None,
    noise_cond: TensorDict | None = None,
    stop_grad_target: bool = True,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 1.0,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    endpoint_space_loss: bool = False,
    target_clip: float = 0.0,
    delta_clip: float = 0.0,
    atom_loss_type: str = "mse",
    ce_shift_clip: float = 5.0,
) -> tuple[Tensor, dict]:
    """Off-diagonal ESD (Eulerian self-distillation) consistency loss.

    Enforces the flow-map PDE with the GLASS teacher providing the diagonal velocity
    ``vss``. With tangent ``(ds=1, du=0, dI_s=vss)``::

        jvp = d/de [ v(s+e, u, I_s + e*vss) ]            (forward-mode AD)
        target = vss + clip((u - s) * jvp)               (stop-grad)
        student = v(s, u, I_s)
        loss = MSE(student, target)

    The JVP runs under the MATH SDPA backend (fused SDPA lacks forward-mode AD).

    With ``atom_loss_type="ce"`` the atom term is instead the soft-label cross
    entropy between ``softmax(z_{s,u})`` and the simplex-valued logit-space target
    of ``esd_logit_space_target`` (paper Eq. 38); coordinates keep the MSE above.
    """
    i_s, cond = build_diagonal_batch(
        teacher, x1, s, t_cond, noise_s=noise_s, noise_cond=noise_cond
    )
    mask = i_s["padding_mask"]
    cc, ca = cond["coords"], cond["atomics"]

    if atom_loss_type not in ("mse", "ce"):
        raise ValueError(f"Unknown atom_loss_type={atom_loss_type!r}.")
    with_logits = _student_has_atom_logits(student)
    if atom_loss_type == "ce" and not with_logits:
        raise ValueError("atom_loss_type='ce' needs a softmax-endpoint atom parametrization")
    # Diagonal GLASS velocity vss = v(s, s, I_s) target (teacher, no grad).
    teacher_logits = None
    if with_logits:
        vss, teacher_logits = extract_glass_velocity(
            teacher, i_s, cond, s, t_cond, return_atom_logits=True
        )
    else:
        vss = extract_glass_velocity(teacher, i_s, cond, s, t_cond)
    if target_clip > 0.0:
        vss["coords"] = vss["coords"].clamp(-target_clip, target_clip)
        vss["atomics"] = vss["atomics"].clamp(-target_clip, target_clip)

    def vsu_fn(s_in, coords_in, atoms_in):
        with sdpa_kernel(SDPBackend.MATH):
            return student(
                coords_in, atoms_in, cc, ca, mask, s_in, u, t_cond,
                return_atom_logits=with_logits,
            )

    primals = (s, i_s["coords"], i_s["atomics"])
    tangents = (torch.ones_like(s), vss["coords"], vss["atomics"])
    outs, jvps = torch.func.jvp(vsu_fn, primals, tangents)
    (vsu_c, vsu_a), (jvp_c, jvp_a) = outs[:2], jvps[:2]

    du = (u - s).view(-1, *([1] * (vsu_c.ndim - 1)))
    real_mask = (1 - mask.int()).to(vsu_c.dtype)
    delta_c = du * jvp_c
    delta_a = du * jvp_a
    delta_c_raw_norm = delta_c.norm()
    delta_a_raw_norm = delta_a.norm()
    if delta_clip > 0.0:
        delta_c = _clip_sample_l2(delta_c, real_mask, delta_clip)
        delta_a = _clip_sample_l2(delta_a, real_mask, delta_clip)
    tgt_c = vss["coords"] + delta_c
    tgt_a = vss["atomics"] + delta_a
    if stop_grad_target:
        tgt_c, tgt_a = tgt_c.detach(), tgt_a.detach()

    sw = ((1.0 - s) ** 2) if endpoint_space_loss else None
    if loss_type == "adaptive":
        coord_loss = _masked_adaptive_loss(
            vsu_c, tgt_c, real_mask, p=adaptive_p, c=adaptive_c, sample_weight=sw
        )
        atom_loss = _masked_adaptive_loss(
            vsu_a, tgt_a, real_mask, p=adaptive_p, c=adaptive_c, sample_weight=sw
        )
    else:
        coord_loss = _masked_mse(vsu_c, tgt_c, real_mask, sw)
        atom_loss = _masked_mse(vsu_a, tgt_a, real_mask, sw)
    zero = coord_loss.detach() * 0.0
    atom_ce, atom_kl, target_stats = zero, zero, {}
    if with_logits:
        ce_target, target_stats = esd_logit_space_target(
            teacher_logits, outs[2], jvps[2], s, u, real_mask,
            shift_clip=ce_shift_clip,
        )
        atom_ce, atom_kl = _masked_soft_cross_entropy(outs[2], ce_target, real_mask)
    if atom_loss_type == "ce":
        loss = coord_weight * coord_loss + atom_weight * atom_ce
    else:
        loss = coord_weight * coord_loss + atom_weight * atom_loss
    # report raw (unweighted) endpoint-space L2 for comparability across runs
    with torch.no_grad():
        ep = (1.0 - s) ** 2
        raw_c = _masked_mse(vsu_c, tgt_c, real_mask, ep)
        raw_a = _masked_mse(vsu_a, tgt_a, real_mask, ep)
    return loss, {
        "esd_coord_l2": float(raw_c.detach()),
        "esd_atom_l2": float(raw_a.detach()),
        "esd_atom_ce": float(atom_ce.detach()),
        "esd_atom_kl": float(atom_kl.detach()),
        **target_stats,
        "jvp_coord_norm": float(jvp_c.norm().detach()),
        "jvp_atom_norm": float(jvp_a.norm().detach()),
        "delta_coord_norm": float(delta_c_raw_norm.detach()),
        "delta_atom_norm": float(delta_a_raw_norm.detach()),
        "delta_coord_norm_clipped": float(delta_c.norm().detach()),
        "delta_atom_norm_clipped": float(delta_a.norm().detach()),
        "target_coord_norm": float(tgt_c.norm().detach()),
        "target_atom_norm": float(tgt_a.norm().detach()),
        "student_coord_norm": float(vsu_c.norm().detach()),
        "student_atom_norm": float(vsu_a.norm().detach()),
    }


def psd_consistency_loss(
    student,
    teacher,
    x1: TensorDict,
    s: Tensor,
    u: Tensor,
    w: Tensor,
    t_cond: Tensor,
    *,
    noise_s: TensorDict | None = None,
    noise_cond: TensorDict | None = None,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 1.0,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    endpoint_space_loss: bool = False,
    atom_loss_type: str = "mse",
) -> tuple[Tensor, dict]:
    """Progressive self-distillation (PSD) consistency loss (paper Eqs. 39-41).

    Enforces the semigroup property ``X_{s,w} = X_{u,w}(X_{s,u})`` for
    ``s <= u <= w``. In velocity form (``X_{s,u}(x) = x + (u - s) v_{s,u}(x)``)::

        x_u    = I_s + (u - s) * v(s, u, I_s)                          (stop-grad)
        target = [(u - s) v(s, u, I_s) + (w - u) v(u, w, x_u)] / (w - s)
        loss   = MSE(v(s, w, I_s), target)

    which, in denoiser form, is the convex combination
    ``T = alpha * Psi_{s,u}(I_s) + beta * Psi_{u,w}(x_u)`` with
    ``alpha = (u-s)(1-w)/((w-s)(1-u))``, ``beta = (w-u)(1-s)/((w-s)(1-u))``.
    With ``atom_loss_type="ce"`` the atom term is the soft-label cross entropy of
    ``softmax(z_{s,w})`` against ``T``; since ``alpha, beta >= 0`` sum to one, T is
    on the simplex and needs no clipping. Coordinates always use the velocity loss.
    No derivatives are taken, so no JVP is needed.
    """
    if atom_loss_type not in ("mse", "ce"):
        raise ValueError(f"Unknown atom_loss_type={atom_loss_type!r}.")
    with_logits = _student_has_atom_logits(student)
    if atom_loss_type == "ce" and not with_logits:
        raise ValueError("atom_loss_type='ce' needs a softmax-endpoint atom parametrization")
    i_s, cond = build_diagonal_batch(
        teacher, x1, s, t_cond, noise_s=noise_s, noise_cond=noise_cond
    )
    mask = i_s["padding_mask"]
    cc, ca = cond["coords"], cond["atomics"]

    def bcast(x: Tensor) -> Tensor:
        return x.view(-1, 1, 1)

    w_minus_s = (w - s).clamp_min(1e-6)
    with torch.no_grad():
        out_su = student(
            i_s["coords"], i_s["atomics"], cc, ca, mask, s, u, t_cond,
            return_atom_logits=with_logits,
        )
        x_u_c = mask_and_zero_com(i_s["coords"] + bcast(u - s) * out_su[0], mask)
        x_u_a = apply_mask(i_s["atomics"] + bcast(u - s) * out_su[1], mask)
        out_uw = student(
            x_u_c, x_u_a, cc, ca, mask, u, w, t_cond,
            return_atom_logits=with_logits,
        )
        lam_su = bcast((u - s) / w_minus_s)
        lam_uw = bcast((w - u) / w_minus_s)
        tgt_c = mask_and_zero_com(lam_su * out_su[0] + lam_uw * out_uw[0], mask)
        tgt_a = apply_mask(lam_su * out_su[1] + lam_uw * out_uw[1], mask)

    out_sw = student(
        i_s["coords"], i_s["atomics"], cc, ca, mask, s, w, t_cond,
        return_atom_logits=with_logits,
    )
    vsw_c, vsw_a = out_sw[0], out_sw[1]
    real_mask = (1 - mask.int()).to(vsw_c.dtype)
    sw = ((1.0 - s) ** 2) if endpoint_space_loss else None
    if loss_type == "adaptive":
        coord_loss = _masked_adaptive_loss(
            vsw_c, tgt_c, real_mask, p=adaptive_p, c=adaptive_c, sample_weight=sw
        )
        atom_loss = _masked_adaptive_loss(
            vsw_a, tgt_a, real_mask, p=adaptive_p, c=adaptive_c, sample_weight=sw
        )
    elif loss_type == "mse":
        coord_loss = _masked_mse(vsw_c, tgt_c, real_mask, sw)
        atom_loss = _masked_mse(vsw_a, tgt_a, real_mask, sw)
    else:
        raise ValueError(f"Unknown PSD loss_type={loss_type!r}.")

    zero = coord_loss.detach() * 0.0
    atom_ce, atom_kl = zero, zero
    if with_logits:
        with torch.no_grad():
            one_minus_u = bcast(1.0 - u).clamp_min(1e-6)
            alpha = bcast((u - s) * (1.0 - w)) / (bcast(w_minus_s) * one_minus_u)
            beta = bcast((w - u) * (1.0 - s)) / (bcast(w_minus_s) * one_minus_u)
            ce_target = (
                alpha * torch.softmax(out_su[2].float(), dim=-1)
                + beta * torch.softmax(out_uw[2].float(), dim=-1)
            )
        atom_ce, atom_kl = _masked_soft_cross_entropy(out_sw[2], ce_target, real_mask)
    if atom_loss_type == "ce":
        loss = coord_weight * coord_loss + atom_weight * atom_ce
    else:
        loss = coord_weight * coord_loss + atom_weight * atom_loss

    with torch.no_grad():
        ep = (1.0 - s) ** 2
        raw_c = _masked_mse(vsw_c, tgt_c, real_mask, ep)
        raw_a = _masked_mse(vsw_a, tgt_a, real_mask, ep)
    return loss, {
        "psd_coord_l2": float(raw_c.detach()),
        "psd_atom_l2": float(raw_a.detach()),
        "psd_atom_ce": float(atom_ce.detach()),
        "psd_atom_kl": float(atom_kl.detach()),
        "target_coord_norm": float(tgt_c.norm().detach()),
        "target_atom_norm": float(tgt_a.norm().detach()),
        "student_coord_norm": float(vsw_c.norm().detach()),
        "student_atom_norm": float(vsw_a.norm().detach()),
    }


def glass_transition_distill_loss(
    student,
    teacher,
    x1: TensorDict,
    t_cond: Tensor,
    *,
    n_flow_steps: int,
    interval_idx: int,
    teacher_steps: int = 64,
    noise_cond: TensorDict | None = None,
    noise_init: TensorDict | None = None,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 1.0,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
) -> tuple[Tensor, dict]:
    """Directly distill a GLASS transition for a target few-step sampler grid.

    Instead of using the local ESD/JVP target, this runs a fine GLASS trajectory
    from prior noise and trains the student flow map on exactly one coarse
    transition ``s -> u``:

        target_v = (x_u^GLASS - x_s^GLASS) / (u - s)

    If ``n_flow_steps=1`` this is direct one-step distillation. If
    ``n_flow_steps=2`` or ``4``, the intervals match the two-/four-step sampler
    used at inference.
    """
    if n_flow_steps <= 0:
        raise ValueError("n_flow_steps must be positive")
    if not (0 <= interval_idx < n_flow_steps):
        raise ValueError(
            f"interval_idx={interval_idx} must be in [0, {n_flow_steps})"
        )
    if teacher_steps % n_flow_steps != 0:
        raise ValueError(
            f"teacher_steps={teacher_steps} must be divisible by n_flow_steps={n_flow_steps}"
        )

    path_cond = teacher._create_path(x1, t=t_cond, noise_batch=noise_cond)
    cond = path_cond.x_t
    state = _clone_state(noise_init) if noise_init is not None else teacher._sample_noise_like_batch(cond)
    mask = cond["padding_mask"]

    start_idx = interval_idx * teacher_steps // n_flow_steps
    end_idx = (interval_idx + 1) * teacher_steps // n_flow_steps
    state_s = _clone_state(state) if start_idx == 0 else None

    with torch.no_grad():
        for i in range(end_idx):
            if i == start_idx:
                state_s = _clone_state(state)
            s_cur = torch.full_like(t_cond, float(i / teacher_steps))
            ds = 1.0 / float(teacher_steps)
            vel = extract_glass_velocity(teacher, state, cond, s_cur, t_cond)
            state["coords"] = mask_and_zero_com(
                state["coords"] + ds * vel["coords"], mask
            )
            state["atomics"] = apply_mask(state["atomics"] + ds * vel["atomics"], mask)
        state_u = _clone_state(state)

    if state_s is None:
        raise RuntimeError("Failed to capture GLASS transition start state.")

    s_val = start_idx / float(teacher_steps)
    u_val = end_idx / float(teacher_steps)
    s = torch.full_like(t_cond, s_val)
    u = torch.full_like(t_cond, u_val)
    denom = max(u_val - s_val, 1e-8)
    target_c = mask_and_zero_com((state_u["coords"] - state_s["coords"]) / denom, mask)
    target_a = apply_mask((state_u["atomics"] - state_s["atomics"]) / denom, mask)

    coords_v, atomics_v = student(
        state_s["coords"],
        state_s["atomics"],
        cond["coords"],
        cond["atomics"],
        mask,
        s,
        u,
        t_cond,
    )

    real_mask = (1 - mask.int()).to(coords_v.dtype)
    if loss_type == "adaptive":
        coord_loss = _masked_adaptive_loss(
            coords_v, target_c, real_mask, p=adaptive_p, c=adaptive_c
        )
        atom_loss = _masked_adaptive_loss(
            atomics_v, target_a, real_mask, p=adaptive_p, c=adaptive_c
        )
    else:
        coord_loss = _masked_mse(coords_v, target_c, real_mask)
        atom_loss = _masked_mse(atomics_v, target_a, real_mask)
    loss = coord_weight * coord_loss + atom_weight * atom_loss
    with torch.no_grad():
        raw_c = _masked_mse(coords_v, target_c, real_mask)
        raw_a = _masked_mse(atomics_v, target_a, real_mask)
    return loss, {
        "transition_coord_l2": float(raw_c.detach()),
        "transition_atom_l2": float(raw_a.detach()),
        "transition_target_coord_norm": float(target_c.norm().detach()),
        "transition_target_atom_norm": float(target_a.norm().detach()),
        "transition_student_coord_norm": float(coords_v.norm().detach()),
        "transition_student_atom_norm": float(atomics_v.norm().detach()),
        "transition_n_flow_steps": float(n_flow_steps),
        "transition_interval_idx": float(interval_idx),
    }


def glass_rollout_distill_loss(
    student,
    teacher,
    x1: TensorDict,
    t_cond: Tensor,
    *,
    n_flow_steps: int = 4,
    teacher_steps: int = 64,
    noise_cond: TensorDict | None = None,
    noise_init: TensorDict | None = None,
    cond_state: TensorDict | None = None,
    loss_type: str = "mse",
    adaptive_p: float = 0.5,
    adaptive_c: float = 1.0,
    coord_weight: float = 1.0,
    atom_weight: float = 1.0,
    final_weight: float = 2.0,
    velocity_weight: float = 0.0,
    teacher_forced_velocity_weight: float = 0.0,
    spread_weight: float = 0.0,
    spread_samples: int = 1,
    diagonal: bool = False,
) -> tuple[Tensor, dict]:
    """Distill the realized few-step student rollout against GLASS coarse nodes.

    Stage-3 trains isolated GLASS transitions ``x_s -> x_u``. This objective
    instead starts teacher and student from the same prior noise, runs the
    student for the actual ``n_flow_steps`` jump sampler, and matches the
    resulting states to GLASS64 states at the same coarse nodes. The loss is
    backpropagated through the whole student rollout, exposing compounding error.
    With ``diagonal=True``, query ``v(s,s,.)`` at each coarse node and still
    advance by the coarse step. This trains the multi-step diagonal GLASS
    integrator used when collapsing to fewer steps is not the primary target.
    """
    if n_flow_steps <= 0:
        raise ValueError("n_flow_steps must be positive")
    if teacher_steps % n_flow_steps != 0:
        raise ValueError(
            f"teacher_steps={teacher_steps} must be divisible by n_flow_steps={n_flow_steps}"
        )
    if spread_samples <= 0:
        raise ValueError("spread_samples must be positive")
    if spread_weight > 0.0 and spread_samples < 2:
        raise ValueError("spread_weight requires spread_samples >= 2")

    if cond_state is None:
        path_cond = teacher._create_path(x1, t=t_cond, noise_batch=noise_cond)
        cond = path_cond.x_t
    else:
        cond = _clone_state(cond_state)
    if spread_samples > 1:
        cond = _repeat_state_batch(cond, spread_samples)
        t_cond = t_cond.repeat_interleave(spread_samples)
    mask = cond["padding_mask"]
    init_state = _clone_state(noise_init) if noise_init is not None else teacher._sample_noise_like_batch(cond)

    node_stride = teacher_steps // n_flow_steps
    target_nodes: list[TensorDict] = []
    with torch.no_grad():
        state = _clone_state(init_state)
        target_nodes.append(_clone_state(state))
        for i in range(teacher_steps):
            s_cur = torch.full_like(t_cond, float(i / teacher_steps))
            ds = 1.0 / float(teacher_steps)
            vel = extract_glass_velocity(teacher, state, cond, s_cur, t_cond)
            state["coords"] = mask_and_zero_com(
                state["coords"] + ds * vel["coords"], mask
            )
            state["atomics"] = apply_mask(state["atomics"] + ds * vel["atomics"], mask)
            if (i + 1) % node_stride == 0:
                target_nodes.append(_clone_state(state))

    if len(target_nodes) != n_flow_steps + 1:
        raise RuntimeError(
            f"Expected {n_flow_steps + 1} GLASS nodes, got {len(target_nodes)}"
        )

    real_mask = (1 - mask.int()).to(cond["coords"].dtype)
    coords = init_state["coords"]
    atomics = init_state["atomics"]
    coord_losses: list[Tensor] = []
    atom_losses: list[Tensor] = []
    velocity_coord_losses: list[Tensor] = []
    velocity_atom_losses: list[Tensor] = []
    teacher_forced_velocity_coord_losses: list[Tensor] = []
    teacher_forced_velocity_atom_losses: list[Tensor] = []
    spread_losses: list[Tensor] = []
    spread_ratios: list[Tensor] = []
    spread_student_means: list[Tensor] = []
    spread_target_means: list[Tensor] = []
    weighted_terms: list[Tensor] = []

    for i in range(n_flow_steps):
        s_val = i / float(n_flow_steps)
        u_val = (i + 1) / float(n_flow_steps)
        s = torch.full_like(t_cond, s_val)
        u = torch.full_like(t_cond, u_val)
        u_query = s if diagonal else u
        coords_v, atomics_v = student(
            coords,
            atomics,
            cond["coords"],
            cond["atomics"],
            mask,
            s,
            u_query,
            t_cond,
        )
        step_size = u_val - s_val
        target_prev = target_nodes[i]
        target = target_nodes[i + 1]
        if velocity_weight > 0.0 or teacher_forced_velocity_weight > 0.0:
            target_v_c = mask_and_zero_com(
                (target["coords"] - target_prev["coords"]) / step_size, mask
            )
            target_v_a = apply_mask(
                (target["atomics"] - target_prev["atomics"]) / step_size, mask
            )
        if velocity_weight > 0.0:
            if loss_type == "adaptive":
                velocity_coord_loss = _masked_adaptive_loss(
                    coords_v,
                    target_v_c,
                    real_mask,
                    p=adaptive_p,
                    c=adaptive_c,
                )
                velocity_atom_loss = _masked_adaptive_loss(
                    atomics_v,
                    target_v_a,
                    real_mask,
                    p=adaptive_p,
                    c=adaptive_c,
                )
            else:
                velocity_coord_loss = _masked_mse(coords_v, target_v_c, real_mask)
                velocity_atom_loss = _masked_mse(atomics_v, target_v_a, real_mask)
            velocity_coord_losses.append(velocity_coord_loss)
            velocity_atom_losses.append(velocity_atom_loss)
        if teacher_forced_velocity_weight > 0.0:
            tf_coords_v, tf_atomics_v = student(
                target_prev["coords"],
                target_prev["atomics"],
                cond["coords"],
                cond["atomics"],
                mask,
                s,
                u_query,
                t_cond,
            )
            if loss_type == "adaptive":
                tf_velocity_coord_loss = _masked_adaptive_loss(
                    tf_coords_v,
                    target_v_c,
                    real_mask,
                    p=adaptive_p,
                    c=adaptive_c,
                )
                tf_velocity_atom_loss = _masked_adaptive_loss(
                    tf_atomics_v,
                    target_v_a,
                    real_mask,
                    p=adaptive_p,
                    c=adaptive_c,
                )
            else:
                tf_velocity_coord_loss = _masked_mse(
                    tf_coords_v, target_v_c, real_mask
                )
                tf_velocity_atom_loss = _masked_mse(
                    tf_atomics_v, target_v_a, real_mask
                )
            teacher_forced_velocity_coord_losses.append(tf_velocity_coord_loss)
            teacher_forced_velocity_atom_losses.append(tf_velocity_atom_loss)

        coords = mask_and_zero_com(coords + step_size * coords_v, mask)
        atomics = apply_mask(atomics + step_size * atomics_v, mask)

        if loss_type == "adaptive":
            coord_loss = _masked_adaptive_loss(
                coords,
                target["coords"],
                real_mask,
                p=adaptive_p,
                c=adaptive_c,
            )
            atom_loss = _masked_adaptive_loss(
                atomics,
                target["atomics"],
                real_mask,
                p=adaptive_p,
                c=adaptive_c,
            )
        else:
            coord_loss = _masked_mse(coords, target["coords"], real_mask)
            atom_loss = _masked_mse(atomics, target["atomics"], real_mask)
        node_weight = final_weight if i == n_flow_steps - 1 else 1.0
        coord_losses.append(coord_loss)
        atom_losses.append(atom_loss)
        node_loss = node_weight * (coord_weight * coord_loss + atom_weight * atom_loss)
        if velocity_weight > 0.0:
            node_loss = node_loss + velocity_weight * (
                coord_weight * velocity_coord_losses[-1]
                + atom_weight * velocity_atom_losses[-1]
            )
        if teacher_forced_velocity_weight > 0.0:
            node_loss = node_loss + teacher_forced_velocity_weight * (
                coord_weight * teacher_forced_velocity_coord_losses[-1]
                + atom_weight * teacher_forced_velocity_atom_losses[-1]
            )
        if spread_weight > 0.0:
            student_spread = _coord_spread_by_group(coords, real_mask, spread_samples)
            target_spread = _coord_spread_by_group(
                target["coords"], real_mask, spread_samples
            ).detach()
            spread_ratio = student_spread / target_spread.clamp_min(1e-4)
            spread_loss = ((student_spread - target_spread) / target_spread.clamp_min(1e-4)).pow(2).mean()
            spread_losses.append(spread_loss)
            spread_ratios.append(spread_ratio.mean())
            spread_student_means.append(student_spread.mean())
            spread_target_means.append(target_spread.mean())
            node_loss = node_loss + node_weight * spread_weight * spread_loss
        weighted_terms.append(node_loss)

    loss = torch.stack(weighted_terms).mean()
    coord_stack = torch.stack(coord_losses)
    atom_stack = torch.stack(atom_losses)
    if velocity_coord_losses:
        velocity_coord_stack = torch.stack(velocity_coord_losses)
        velocity_atom_stack = torch.stack(velocity_atom_losses)
        velocity_coord_l2 = float(velocity_coord_stack.mean().detach())
        velocity_atom_l2 = float(velocity_atom_stack.mean().detach())
        velocity_final_coord_l2 = float(velocity_coord_stack[-1].detach())
        velocity_final_atom_l2 = float(velocity_atom_stack[-1].detach())
    else:
        velocity_coord_l2 = 0.0
        velocity_atom_l2 = 0.0
        velocity_final_coord_l2 = 0.0
        velocity_final_atom_l2 = 0.0
    if teacher_forced_velocity_coord_losses:
        tf_velocity_coord_stack = torch.stack(teacher_forced_velocity_coord_losses)
        tf_velocity_atom_stack = torch.stack(teacher_forced_velocity_atom_losses)
        tf_velocity_coord_l2 = float(tf_velocity_coord_stack.mean().detach())
        tf_velocity_atom_l2 = float(tf_velocity_atom_stack.mean().detach())
        tf_velocity_final_coord_l2 = float(tf_velocity_coord_stack[-1].detach())
        tf_velocity_final_atom_l2 = float(tf_velocity_atom_stack[-1].detach())
    else:
        tf_velocity_coord_l2 = 0.0
        tf_velocity_atom_l2 = 0.0
        tf_velocity_final_coord_l2 = 0.0
        tf_velocity_final_atom_l2 = 0.0
    if spread_losses:
        spread_stack = torch.stack(spread_losses)
        spread_ratio_stack = torch.stack(spread_ratios)
        spread_student_stack = torch.stack(spread_student_means)
        spread_target_stack = torch.stack(spread_target_means)
        spread_l2 = float(spread_stack.mean().detach())
        spread_final_l2 = float(spread_stack[-1].detach())
        spread_ratio = float(spread_ratio_stack.mean().detach())
        spread_final_ratio = float(spread_ratio_stack[-1].detach())
        spread_student_final = float(spread_student_stack[-1].detach())
        spread_target_final = float(spread_target_stack[-1].detach())
    else:
        spread_l2 = 0.0
        spread_final_l2 = 0.0
        spread_ratio = 0.0
        spread_final_ratio = 0.0
        spread_student_final = 0.0
        spread_target_final = 0.0
    return loss, {
        "rollout_coord_l2": float(coord_stack.mean().detach()),
        "rollout_atom_l2": float(atom_stack.mean().detach()),
        "rollout_final_coord_l2": float(coord_stack[-1].detach()),
        "rollout_final_atom_l2": float(atom_stack[-1].detach()),
        "rollout_velocity_coord_l2": velocity_coord_l2,
        "rollout_velocity_atom_l2": velocity_atom_l2,
        "rollout_velocity_final_coord_l2": velocity_final_coord_l2,
        "rollout_velocity_final_atom_l2": velocity_final_atom_l2,
        "rollout_teacher_forced_velocity_coord_l2": tf_velocity_coord_l2,
        "rollout_teacher_forced_velocity_atom_l2": tf_velocity_atom_l2,
        "rollout_teacher_forced_velocity_final_coord_l2": tf_velocity_final_coord_l2,
        "rollout_teacher_forced_velocity_final_atom_l2": tf_velocity_final_atom_l2,
        "rollout_coord_spread_l2": spread_l2,
        "rollout_final_coord_spread_l2": spread_final_l2,
        "rollout_coord_spread_ratio": spread_ratio,
        "rollout_final_coord_spread_ratio": spread_final_ratio,
        "rollout_final_coord_spread_student": spread_student_final,
        "rollout_final_coord_spread_target": spread_target_final,
        "rollout_steps": float(n_flow_steps),
        "rollout_final_weight": float(final_weight),
        "rollout_velocity_weight": float(velocity_weight),
        "rollout_teacher_forced_velocity_weight": float(teacher_forced_velocity_weight),
        "rollout_spread_weight": float(spread_weight),
        "rollout_spread_samples": float(spread_samples),
        "rollout_diagonal": float(diagonal),
        "rollout_target_coord_norm": float(target_nodes[-1]["coords"].norm().detach()),
        "rollout_target_atom_norm": float(target_nodes[-1]["atomics"].norm().detach()),
        "rollout_student_coord_norm": float(coords.norm().detach()),
        "rollout_student_atom_norm": float(atomics.norm().detach()),
    }
