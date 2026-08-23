from __future__ import annotations

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def horizon_indices(fps: int, t_pred: int) -> dict[float, int]:
    """Frame index (0-indexed) within a T_pred window corresponding to each
    protocol horizon (0.5s / 1.0s / 1.5s), given the dataset's fps."""
    result = {}
    for h in (0.5, 1.0, 1.5):
        i = min(int(round(h * fps)) - 1, t_pred - 1)
        result[h] = i
    return result


def trajectory_metrics(
    pred_xy_px: torch.Tensor, target_xy_px: torch.Tensor, target_mask: torch.Tensor, fps: int, t_pred: int
) -> dict[str, float]:
    """pred_xy_px, target_xy_px: [B, T_pred, 2] in pixel units. target_mask: [B, T_pred] bool.
    ADE/FDE + MSE at the frozen 0.5s/1.0s/1.5s protocol horizons."""
    diff = pred_xy_px - target_xy_px
    dist = diff.norm(dim=-1)  # [B, T]
    mask = target_mask.float()

    ade = (dist * mask).sum() / mask.sum().clamp(min=1)
    # windows are built with require_full_future=True, so the last frame is always valid
    fde = dist[:, -1].mean()

    sq = (diff**2).sum(dim=-1)  # [B, T]
    result = {"ade": ade.item(), "fde": fde.item()}
    for h, idx in horizon_indices(fps, t_pred).items():
        result[f"mse_{h}s"] = sq[:, idx].mean().item()
    return result


def intent_metrics(probs, labels) -> dict[str, float]:
    """probs, labels: 1D numpy arrays of equal length, NaN labels already filtered by caller."""
    preds = (probs >= 0.5).astype(int)
    out: dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = float("nan")  # only one class present in this split
    out["f1"] = float(f1_score(labels, preds, zero_division=0))
    out["precision"] = float(precision_score(labels, preds, zero_division=0))
    out["recall"] = float(recall_score(labels, preds, zero_division=0))
    return out
