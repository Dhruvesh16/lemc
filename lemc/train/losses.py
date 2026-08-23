from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """pred_xy, target_xy: [B, T_pred, 2]. target_mask: [B, T_pred] bool. Masked MSE."""
    diff2 = (pred_xy - target_xy) ** 2
    per_frame = diff2.sum(dim=-1)  # [B, T]
    mask = target_mask.float()
    return (per_frame * mask).sum() / mask.sum().clamp(min=1)


def intent_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: [B, 1]. labels: [B] float, NaN where unknown (excluded)."""
    valid = ~torch.isnan(labels)
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[valid].squeeze(-1), labels[valid])


def residual_l2_penalty(residual: torch.Tensor | None) -> torch.Tensor:
    """Small anti-explosion penalty on LEMC's residual magnitude (distinct from
    anti-collapse -- see lemc/train/loop.py's collapse-detection logging)."""
    if residual is None:
        return torch.zeros(())
    return (residual**2).sum(dim=-1).mean()


def obd_correlation_loss(residual: torch.Tensor | None, ego_speed: torch.Tensor) -> torch.Tensor:
    """Auxiliary, PIE-only, training-time-only supervision (JAAD has no OBD --
    LEMC still takes zero sensor input at inference on either dataset, this
    just shapes what it learns to output during training on PIE).

    residual: [B, T_obs-1, 2]. ego_speed: [B, T_obs], NaN where unknown.
    Returns 1 - Pearson r between per-window mean residual magnitude and mean
    OBD speed -- correlation (not an absolute-scale MSE) because it's scale/
    shift invariant, directly optimizing the same statistic
    scripts/05_validate_lemc.py reports as the trust gate. 0 (no gradient)
    when there's not enough valid-speed signal in the batch to compute it."""
    if residual is None:
        return torch.zeros(())
    mean_mag = residual.norm(dim=-1).mean(dim=1)  # [B]
    mean_speed = ego_speed.nanmean(dim=1)  # [B]
    valid = ~torch.isnan(mean_speed)
    if valid.sum() < 2:
        return torch.zeros((), device=residual.device)
    x = mean_mag[valid] - mean_mag[valid].mean()
    y = mean_speed[valid] - mean_speed[valid].mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    r = (x * y).sum() / denom
    return 1.0 - r
