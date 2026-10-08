"""GLASS posterior-velocity target and adaptive losses from the vendored MFM.

``extract_posterior_velocity`` is copied from ``mfm/losses/losses.py`` and the
loss helpers from ``mfm/losses/utils.py`` of the MFM copy vendored in
DNA-MFM@187fe7b (see ``dmfm/_mfm/__init__.py``). Only the ``labels is None``
branch of ``extract_posterior_velocity`` is used by the DNA code; the
class-conditional (CFG) branches are image-only and are not carried over.
"""

import torch


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


def l2_loss(
    pred,
    target,
    weighting,
    stop_gradient=False,
):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta_sq = (pred - actual_target) ** 2
    # sum over data dimensions
    delta_sq = torch.sum(delta_sq, dim=list(range(1, len(delta_sq.shape))))
    weighting = broadcast_to_shape(weighting, delta_sq.shape)
    weighted_delta_sq = (1 / weighting.exp()) * delta_sq + weighting
    # mean over batch
    return torch.mean(weighted_delta_sq), torch.mean(delta_sq)


def log_lv_loss(pred, target, weighting, stop_gradient=False):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta_sq = (pred - actual_target) ** 2
    mean_loss = torch.mean(delta_sq, dim=list(range(1, len(pred.shape))))
    # Reshape mean_loss to match weighting for broadcasting: (B,) -> (B, 1, 1, 1)
    mean_loss = broadcast_to_shape(mean_loss, weighting.shape)
    log_loss = torch.log((1 / weighting.exp()) * mean_loss + 1.0) + 0.5 * weighting
    return torch.mean(log_loss), torch.mean(mean_loss)


def adaptive_loss(pred, target, weighting, p, c, stop_gradient=False):
    """Computes the adaptively weighted squared L2 loss.
    Loss = w * ||pred - target||^2, where w = 1 / (||pred - target||^2 + c)^p.
    """
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target

    delta_sq = (pred - actual_target) ** 2
    # sum over data dimensions
    delta_sq = torch.sum(delta_sq, dim=tuple(range(1, len(delta_sq.shape))))
    weight = 1.0 / (delta_sq + c) ** p
    weight = torch.detach(weight)
    weight = broadcast_to_shape(weight, delta_sq.shape)
    delta_sq = delta_sq * weight
    weighted_delta_sq = delta_sq / weighting.exp() + weighting
    # mean over batch
    return torch.mean(weighted_delta_sq), torch.mean(delta_sq).detach()


def compute_loss(
    pred,
    target,
    weighting,
    loss_type,
    adaptive_p=None,
    adaptive_c=None,
    stop_gradient=False,
):
    if loss_type == "l2":
        return l2_loss(pred, target, weighting, stop_gradient=stop_gradient)
    elif loss_type == "lv":
        return log_lv_loss(pred, target, weighting, stop_gradient=stop_gradient)
    elif loss_type == "adaptive":
        return adaptive_loss(
            pred, target, weighting, adaptive_p, adaptive_c, stop_gradient=stop_gradient
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


@torch.no_grad()
def extract_posterior_velocity(
    s,
    Is,
    xt_cond,
    t_cond,
    labels,
    cfg_scales,
    teacher_model,
    eps=1e-6,
    omega=0.6,
    checkpoint_type="dmf",
):
    device = s.device
    N = s.shape[0]
    s = broadcast_to_shape(s, xt_cond.shape)
    t = broadcast_to_shape(t_cond, xt_cond.shape)
    one_minus_s = 1 - s  # 1

    denom = (t**2 * one_minus_s**2 + (1 - t) ** 2 * s**2).clamp_min(eps)  # t**2
    P_norm = (1 - t) ** 2 / denom  # ((1-t)**2 / t**2)
    sqrt_P_norm = torch.sqrt(P_norm)  # (1-t)/t

    t_star = 1 / (1 + one_minus_s * sqrt_P_norm)  # t

    coeff_cond = t_star * one_minus_s**2 * t / denom  # 1
    coeff_Is = t_star * s * P_norm  # 0
    x_star = coeff_cond * xt_cond + coeff_Is * Is  # x_t_cond

    with torch.no_grad():
        if labels is None:
            v_star = teacher_model.v(
                t_star.view(N),
                t_star.view(N),
                x_star,
                t_cond.view(N),
                xt_cond,
            )
        else:
            raise NotImplementedError(
                "Class-conditional (CFG) posterior velocity is image-only and not vendored in dmfm."
            )

        term2 = t_star * sqrt_P_norm * v_star  # (1-t)b_t(x_t)
        diff_div_x = (
            (1 - t) ** 2 * (1 + s) - t**2 * one_minus_s
        ) / denom  # (1-2t)/t**2
        B_minus_1_div_x = (diff_div_x - (P_norm + sqrt_P_norm)) / (
            1 + one_minus_s * sqrt_P_norm
        )  # -1
        A_div_x = t_star * one_minus_s * t / denom  # 1
        # x_t - I_s
        term1 = A_div_x * xt_cond + B_minus_1_div_x * Is
        # x_t - I_s + (1-t)b_t(x_t) -> x_t + (1-t)b_t(x_t) -> E[x_1|x_t]
        dIsds_distill = term1 + term2

    return dIsds_distill
