from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import GRUTrajectoryModel
from .lemc import LEMCModule

EGO_IDX = 0


class TrajectoryPredictor(nn.Module):
    """Wires LEMC (optional) in front of the backbone. `use_lemc` is the ONE
    switch that changes model structure -- everything else (dataset, backbone,
    loss, training loop, eval) is an identical code path either way. This is
    what keeps the LEMC-on/off ablation clean; see
    tests/test_lemc_off_equals_baseline.py for the regression test that
    enforces it."""

    def __init__(self, use_lemc: bool, lemc_cfg: dict, backbone_cfg: dict):
        super().__init__()
        self.lemc = LEMCModule(**lemc_cfg) if use_lemc else None
        self.backbone = GRUTrajectoryModel(**backbone_cfg)

    def forward(
        self,
        agent_tracks: torch.Tensor,
        agent_mask: torch.Tensor,
        ego_speed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """agent_tracks: [B, T_obs, N, 4], agent_mask: [B, T_obs, N].
        ego_speed: [B, T_obs] when backbone.use_speed_input (GT OBD ablation).
        Returns (traj_pred, intent_logit, residual) -- residual is None when use_lemc=False."""
        if self.lemc is not None:
            ego_track, residual = self.lemc(agent_tracks, agent_mask)
        else:
            ego_track = agent_tracks[:, :, EGO_IDX, :]
            residual = None

        traj_pred, intent_logit = self.backbone(ego_track, ego_speed=ego_speed)
        return traj_pred, intent_logit, residual
