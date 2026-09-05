import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.eval.metrics import box_rmse_metrics, horizon_indices, intent_metrics, trajectory_metrics


def test_horizon_indices_match_frozen_protocol():
    idx = horizon_indices(fps=30, t_pred=45)
    assert idx[0.5] == 14  # frame 15 (1-indexed) -> index 14
    assert idx[1.0] == 29  # frame 30 -> index 29
    assert idx[1.5] == 44  # frame 45 (last) -> index 44


def test_trajectory_metrics_zero_error_when_pred_equals_target():
    B, T = 4, 45
    target = torch.rand(B, T, 2) * 1000
    mask = torch.ones(B, T, dtype=torch.bool)
    result = trajectory_metrics(target.clone(), target, mask, fps=30, t_pred=T)
    assert abs(result["ade"]) < 1e-5
    assert abs(result["fde"]) < 1e-5
    for h in (0.5, 1.0, 1.5):
        assert abs(result[f"mse_{h}s"]) < 1e-5


def test_trajectory_metrics_ignores_masked_frames():
    B, T = 1, 3
    pred = torch.zeros(B, T, 2)
    target = torch.zeros(B, T, 2)
    target[0, 1] = torch.tensor([1000.0, 1000.0])  # huge error, but masked out
    mask = torch.tensor([[True, False, True]])
    result = trajectory_metrics(pred, target, mask, fps=30, t_pred=T)
    assert result["ade"] < 1e-5  # masked frame must not contaminate ADE


def test_intent_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    probs = np.array([0.05, 0.1, 0.9, 0.95], dtype=np.float64)
    result = intent_metrics(probs, labels)
    assert result["auc"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_box_rmse_zero_when_equal():
    B, T = 2, 45
    boxes = torch.rand(B, T, 4) * 100 + 50
    mask = torch.ones(B, T, dtype=torch.bool)
    result = box_rmse_metrics(boxes.clone(), boxes, mask, horizon=30)
    assert result["arb"] < 1e-5
    assert result["frb"] < 1e-5


def test_orb_regression_loss_masked():
    from lemc.train.losses import orb_regression_loss

    residual = torch.zeros(4, 5, 2)
    target = torch.ones(4, 5, 2)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[0, 0] = True
    residual[0, 0] = 1.0
    loss = orb_regression_loss(residual, target, mask)
    assert loss.item() < 1e-5
