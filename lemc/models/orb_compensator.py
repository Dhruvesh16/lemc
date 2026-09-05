"""Non-learned LEMC: apply cached ORB partial-affine to the ego track.

compensated_xy[t] = xy[t] - cumsum_i( M_i @ xy[i] - xy[i] )
Uses the same residual interface as LEMCModule so TrajectoryPredictor can
swap it in via use_lemc + lemc_cfg.kind='orb_affine'.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ORBAffineCompensator(nn.Module):
    """Applies precomputed per-transition shifts (already in residual units)
    supplied via the batch — this module just needs forward(agent_tracks, mask)
    for API compatibility. Actual shifts are injected by wrapping the dataset
    residual path: we instead expose forward_with_orb(agent_tracks, orb_shift, orb_mask).
    """

    def forward(self, agent_tracks: torch.Tensor, agent_mask: torch.Tensor):
        raise RuntimeError("ORBAffineCompensator requires forward_with_orb(...)")

    def forward_with_orb(
        self,
        agent_tracks: torch.Tensor,
        orb_shift: torch.Tensor,
        orb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """orb_shift: [B,T-1,2] affine-induced displacement in residual units.
        orb_mask: [B,T-1] — invalid transitions zeroed before cumsum."""
        residual = orb_shift * orb_mask.unsqueeze(-1).float()
        shift = torch.cumsum(residual, dim=1)
        shift = torch.nn.functional.pad(shift, (0, 0, 1, 0))
        ego = agent_tracks[:, :, 0, :]
        compensated_xy = ego[..., :2] - shift
        compensated = torch.cat([compensated_xy, ego[..., 2:]], dim=-1)
        return compensated, residual
