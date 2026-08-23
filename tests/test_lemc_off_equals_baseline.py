import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.models.backbone import GRUTrajectoryModel
from lemc.models.predictor import TrajectoryPredictor
from lemc.train.seeds import seed_everything


def test_lemc_off_is_bit_identical_to_bare_backbone():
    """The 'differs in exactly one place' contract: with use_lemc=False,
    TrajectoryPredictor must be indistinguishable from running
    GRUTrajectoryModel directly on the ego track. If this ever fails, the
    LEMC-on/off ablation is contaminated by something other than LEMC."""
    backbone_cfg = dict(input_dim=4, hidden_dim=32, num_layers=2, t_pred=10, predict_intent=True)

    seed_everything(0)
    predictor = TrajectoryPredictor(use_lemc=False, lemc_cfg={}, backbone_cfg=backbone_cfg)

    seed_everything(0)
    bare_backbone = GRUTrajectoryModel(**backbone_cfg)

    # same init (same seed) -> same weights; sanity-check before comparing outputs
    for p1, p2 in zip(predictor.backbone.parameters(), bare_backbone.parameters()):
        assert torch.equal(p1, p2)

    B, T_obs, N = 4, 15, 6
    torch.manual_seed(42)
    agent_tracks = torch.rand(B, T_obs, N, 4)
    agent_mask = torch.ones(B, T_obs, N, dtype=torch.bool)

    traj_pred_a, intent_a, residual_a = predictor(agent_tracks, agent_mask)
    traj_pred_b, intent_b = bare_backbone(agent_tracks[:, :, 0, :])

    assert residual_a is None
    assert torch.equal(traj_pred_a, traj_pred_b)
    assert torch.equal(intent_a, intent_b)


def test_lemc_on_uses_lemc_module():
    backbone_cfg = dict(input_dim=4, hidden_dim=16, num_layers=1, t_pred=5, predict_intent=False)
    lemc_cfg = dict(hidden_dim=8, temporal_net="gru")
    predictor = TrajectoryPredictor(use_lemc=True, lemc_cfg=lemc_cfg, backbone_cfg=backbone_cfg)
    assert predictor.lemc is not None

    B, T_obs, N = 2, 15, 4
    agent_tracks = torch.rand(B, T_obs, N, 4)
    agent_mask = torch.ones(B, T_obs, N, dtype=torch.bool)
    traj_pred, intent_logit, residual = predictor(agent_tracks, agent_mask)
    assert residual is not None
    assert residual.shape == (B, T_obs - 1, 2)
    assert traj_pred.shape == (B, 5, 2)
