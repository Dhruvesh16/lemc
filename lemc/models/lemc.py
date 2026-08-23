from __future__ import annotations

import torch
import torch.nn as nn

FEAT_DIM = 8  # median_dx, median_dy, trimmed_dx, trimmed_dy, iqr_dx, iqr_dy, median_scale_ratio, valid_count
TRIM_FRAC = 0.1  # fraction trimmed from each end for the trimmed-mean stat
MIN_OTHER_AGENTS = 2  # below this, displacement stats fall back to "no signal" rather than trusting n=1
EGO_IDX = 0


def _masked_trimmed_mean(vals: torch.Tensor, valid: torch.Tensor, trim_frac: float) -> torch.Tensor:
    """vals, valid: [..., N]. NaN sorts to the end in ascending order, so the
    first valid_count entries of a sort are exactly the valid values, ascending."""
    vals_for_sort = torch.where(valid, vals, torch.full_like(vals, float("nan")))
    sorted_vals, _ = torch.sort(vals_for_sort, dim=-1)  # NaN -> end
    valid_count = valid.sum(dim=-1)  # [...]
    k = torch.floor(trim_frac * valid_count)  # [...]
    idx = torch.arange(vals.shape[-1], device=vals.device).view(*([1] * (vals.dim() - 1)), -1)
    keep = (idx >= k.unsqueeze(-1)) & (idx < (valid_count - k).unsqueeze(-1))
    kept_vals = torch.where(keep, torch.nan_to_num(sorted_vals, nan=0.0), torch.zeros_like(sorted_vals))
    return kept_vals.sum(dim=-1) / keep.sum(dim=-1).clamp(min=1)


def compute_frame_stats(agent_tracks: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """
    agent_tracks: [B, T_obs, N, 4] (cx, cy, w, h), ego at index EGO_IDX=0.
    agent_mask: [B, T_obs, N] bool.
    Returns: [B, T_obs-1, FEAT_DIM] robust per-frame-transition statistics.

    Displacement stats (median/trimmed-mean/IQR of dx,dy) are computed over
    OTHER agents ONLY (ego excluded) -- ego must never vote in the estimate of
    "shared" motion used to compensate its own track, or the estimate becomes
    circular: with too few real other agents, ego's own displacement would
    dominate the "shared" statistic and LEMC would partially subtract the
    pedestrian's genuine motion, mistaking it for camera drift. A transition
    with fewer than MIN_OTHER_AGENTS valid other agents falls back to zero
    (no displacement signal) rather than trusting n=1.

    The scale-ratio cue (h[t+1]/h[t]) is the one exception: the density gate
    spec requires it to work "even on a single track", so it's computed over
    ALL agents including ego -- ego's own height ratio is a legitimate,
    intentionally-confounded single-track proxy for forward motion.

    An agent (other or ego) contributes to a transition t only if valid at
    BOTH frame t and t+1 -- appearing/disappearing shouldn't manufacture a
    displacement from/to a position it never had.

    IMPORTANT: the scale-ratio stat divides raw height values, so this function
    must be called on image-dim-normalized (not z-standardized) coordinates --
    see NormStats in lemc/data/dataset.py for why standardized height would
    break the ratio's sign/positivity.
    """
    trans_valid_all = agent_mask[:, :-1, :] & agent_mask[:, 1:, :]  # [B, T-1, N], ego included

    # --- displacement stats: other agents only ---
    other_tracks = agent_tracks[:, :, 1:, :]  # [B, T_obs, N-1, 4] -- N-1 can be 0 if max_agents==1
    other_valid = trans_valid_all[:, :, 1:]  # [B, T-1, N-1]
    B, Tm1 = trans_valid_all.shape[0], trans_valid_all.shape[1]

    if other_tracks.shape[2] == 0:
        # no "other agent" slots exist at all (max_agents==1) -- can't call
        # nanmedian/nanquantile on a zero-size reduction dim, so short-circuit
        zeros = torch.zeros(B, Tm1, device=agent_tracks.device, dtype=agent_tracks.dtype)
        median_dx = median_dy = trimmed_dx = trimmed_dy = iqr = zeros
        other_valid_count = torch.zeros(B, Tm1, device=agent_tracks.device, dtype=torch.long)
    else:
        dx = other_tracks[:, 1:, :, 0] - other_tracks[:, :-1, :, 0]  # [B, T-1, N-1]
        dy = other_tracks[:, 1:, :, 1] - other_tracks[:, :-1, :, 1]

        dx_nan = torch.where(other_valid, dx, torch.full_like(dx, float("nan")))
        dy_nan = torch.where(other_valid, dy, torch.full_like(dy, float("nan")))

        other_valid_count = other_valid.sum(dim=-1)  # [B, T-1]
        enough_others = other_valid_count >= MIN_OTHER_AGENTS

        median_dx = torch.where(enough_others, torch.nan_to_num(torch.nanmedian(dx_nan, dim=-1).values, nan=0.0), 0.0)
        median_dy = torch.where(enough_others, torch.nan_to_num(torch.nanmedian(dy_nan, dim=-1).values, nan=0.0), 0.0)

        trimmed_dx = torch.where(enough_others, _masked_trimmed_mean(dx, other_valid, TRIM_FRAC), 0.0)
        trimmed_dy = torch.where(enough_others, _masked_trimmed_mean(dy, other_valid, TRIM_FRAC), 0.0)

        q75_dx = torch.nan_to_num(torch.nanquantile(dx_nan, 0.75, dim=-1), nan=0.0)
        q25_dx = torch.nan_to_num(torch.nanquantile(dx_nan, 0.25, dim=-1), nan=0.0)
        q75_dy = torch.nan_to_num(torch.nanquantile(dy_nan, 0.75, dim=-1), nan=0.0)
        q25_dy = torch.nan_to_num(torch.nanquantile(dy_nan, 0.25, dim=-1), nan=0.0)
        iqr = torch.where(enough_others, (q75_dx - q25_dx) + (q75_dy - q25_dy), torch.zeros_like(q75_dx))

    # --- scale-ratio cue: ALL agents including ego (density gate verdict: AMBER on
    # PIE -- required from the start, see DECISIONS.md; must work on ego alone) ---
    h = agent_tracks[:, :, :, 3].clamp(min=1e-4)  # epsilon floor: a real normalized bbox height is never ~0
    scale_ratio = h[:, 1:, :] / h[:, :-1, :]  # [B, T-1, N]
    scale_ratio_nan = torch.where(trans_valid_all, scale_ratio, torch.full_like(scale_ratio, float("nan")))
    median_scale_ratio = torch.nan_to_num(torch.nanmedian(scale_ratio_nan, dim=-1).values, nan=1.0)  # 1.0 = no change

    feats = torch.stack(
        [
            median_dx,
            median_dy,
            trimmed_dx,
            trimmed_dy,
            iqr,
            torch.zeros_like(iqr),  # reserved slot (keeps FEAT_DIM stable if iqr split later)
            median_scale_ratio,
            other_valid_count.float(),
        ],
        dim=-1,
    )
    return feats


class LEMCModule(nn.Module):
    """Learnable Ego-Motion Compensation. Estimates the shared (camera-motion)
    component of agent displacement from track geometry alone -- no speed
    sensor -- and subtracts it from the ego pedestrian's track before
    prediction. Trained end-to-end via the downstream prediction loss."""

    def __init__(self, hidden_dim: int = 32, temporal_net: str = "gru"):
        super().__init__()
        self.temporal_net_type = temporal_net
        if temporal_net == "gru":
            self.temporal_net = nn.GRU(FEAT_DIM, hidden_dim, batch_first=True)
        elif temporal_net == "conv1d":
            self.temporal_net = nn.Conv1d(FEAT_DIM, hidden_dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"unknown temporal_net: {temporal_net}")
        self.residual_head = nn.Linear(hidden_dim, 2)

    def forward(self, agent_tracks: torch.Tensor, agent_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        agent_tracks: [B, T_obs, N, 4], agent_mask: [B, T_obs, N] (ego at index 0, always valid).
        Returns: (compensated_ego_track [B, T_obs, 4], residual [B, T_obs-1, 2]).
        w, h of the ego track are left untouched (only cx, cy shift).
        """
        stats = compute_frame_stats(agent_tracks, agent_mask)  # [B, T_obs-1, FEAT_DIM]

        if self.temporal_net_type == "gru":
            h, _ = self.temporal_net(stats)  # [B, T_obs-1, hidden_dim]
        else:
            h = self.temporal_net(stats.transpose(1, 2)).transpose(1, 2)  # [B, T_obs-1, hidden_dim]

        residual = self.residual_head(h)  # [B, T_obs-1, 2], per-transition predicted shift
        shift = torch.cumsum(residual, dim=1)  # [B, T_obs-1, 2], absolute drift relative to frame 0
        shift = torch.nn.functional.pad(shift, (0, 0, 1, 0))  # [B, T_obs, 2], zero shift at t=0

        ego_track = agent_tracks[:, :, 0, :]  # [B, T_obs, 4]
        compensated_xy = ego_track[..., :2] - shift
        compensated_ego_track = torch.cat([compensated_xy, ego_track[..., 2:]], dim=-1)  # w,h untouched

        return compensated_ego_track, residual
