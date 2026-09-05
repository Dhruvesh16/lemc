from __future__ import annotations

import torch
import torch.nn as nn


class GRUTrajectoryModel(nn.Module):
    """Deliberately boring backbone: Linear embed -> GRU -> trajectory + intent heads.
    Consumes a single pedestrian's [B, T_obs, 4] track (+ optional per-frame
    ego speed). No other agents, no ego-motion compensation inside the backbone.
    Any improvement from LEMC must be attributable to LEMC alone."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        t_pred: int = 45,
        predict_intent: bool = True,
        use_speed_input: bool = False,
    ):
        super().__init__()
        self.t_pred = t_pred
        self.predict_intent = predict_intent
        self.use_speed_input = use_speed_input
        in_dim = input_dim + (1 if use_speed_input else 0)

        self.embed = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.traj_head = nn.Linear(hidden_dim, t_pred * 2)
        if predict_intent:
            self.intent_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, track: torch.Tensor, ego_speed: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """track: [B, T_obs, 4]; ego_speed: [B, T_obs] when use_speed_input.
        -> traj_pred [B, T_pred, 2] (cx, cy only), intent_logit [B, 1] or None."""
        if self.use_speed_input:
            if ego_speed is None:
                raise ValueError("use_speed_input=True but ego_speed was not provided")
            speed = torch.nan_to_num(ego_speed, nan=0.0).unsqueeze(-1)  # [B, T, 1]
            # crude scale: PIE OBD is km/h, typically 0-60; keep input O(1)
            x = torch.cat([track, speed / 30.0], dim=-1)
        else:
            x = track
        x = self.embed(x)
        _, h_n = self.gru(x)
        h_final = h_n[-1]  # [B, hidden_dim], last layer's final hidden state

        traj_pred = self.traj_head(h_final).view(-1, self.t_pred, 2)
        intent_logit = self.intent_head(h_final) if self.predict_intent else None
        return traj_pred, intent_logit
