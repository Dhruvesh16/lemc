import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.train.losses import obd_correlation_loss


def test_obd_correlation_loss_perfect_positive_correlation():
    # construct residual whose magnitude is exactly proportional to ego_speed
    speed = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    residual = torch.zeros(5, 3, 2)
    residual[:, :, 0] = speed.unsqueeze(-1)  # magnitude == speed at every transition
    ego_speed = speed.unsqueeze(-1).repeat(1, 4)  # [B, T_obs]
    loss = obd_correlation_loss(residual, ego_speed)
    assert loss.item() < 1e-4  # perfect positive correlation -> loss ~0


def test_obd_correlation_loss_no_correlation_is_near_one():
    torch.manual_seed(0)
    residual = torch.rand(200, 3, 2)  # random, independent of speed
    ego_speed = torch.rand(200, 4) * 50
    loss = obd_correlation_loss(residual, ego_speed)
    assert 0.7 < loss.item() < 1.3  # roughly uncorrelated -> loss near 1


def test_obd_correlation_loss_excludes_nan_speed_windows():
    speed = torch.tensor([1.0, 2.0, 3.0, 4.0, float("nan"), float("nan")])
    residual = torch.zeros(6, 3, 2)
    residual[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 999.0, -999.0]).unsqueeze(-1)
    ego_speed = speed.unsqueeze(-1).repeat(1, 4)
    loss = obd_correlation_loss(residual, ego_speed)
    assert loss.item() < 1e-4  # the two NaN-speed (garbage-magnitude) windows must not pollute the correlation


def test_obd_correlation_loss_too_few_valid_returns_zero():
    ego_speed = torch.tensor([[1.0, 1.0], [float("nan"), float("nan")]])
    residual = torch.rand(2, 1, 2)
    loss = obd_correlation_loss(residual, ego_speed)
    assert loss.item() == 0.0


def test_obd_correlation_loss_none_residual_returns_zero():
    loss = obd_correlation_loss(None, torch.rand(4, 5))
    assert loss.item() == 0.0


def test_obd_correlation_loss_is_differentiable():
    residual = torch.rand(10, 3, 2, requires_grad=True)
    ego_speed = torch.rand(10, 4) * 30
    loss = obd_correlation_loss(residual, ego_speed)
    loss.backward()
    assert residual.grad is not None
