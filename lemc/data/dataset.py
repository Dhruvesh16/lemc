from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .track_store import SceneStore, window_scene


def _agents_in_window(store: SceneStore, start_frame: int, t_obs: int, ego_pid: str, max_agents: int) -> list[str]:
    """Deterministic (sorted-by-pid) ordering of every non-ego agent visible in
    any of the t_obs observed frames, ego always at index 0, truncated to max_agents."""
    seen = set()
    for f in range(start_frame, start_frame + t_obs):
        for pid in store.frame_to_boxes.get(f, {}):
            if pid != ego_pid:
                seen.add(pid)
    others = sorted(seen)
    return [ego_pid] + others[: max_agents - 1]


def build_window_index(
    stores: dict[str, SceneStore],
    split: str,
    t_obs: int,
    t_pred: int,
    max_agents: int,
    overlap: float,
    require_full_future: bool = True,
) -> list[tuple]:
    """
    Returns a list of (scene_id, ego_pid, start_frame, agent_pids) tuples for
    every valid window in the given split. A window is valid iff window_scene
    doesn't drop it (ego present every observed frame) and, if
    require_full_future, the target track is fully present too.
    """
    stride = t_obs if overlap == 0 else max(1, int((1 - overlap) * t_obs))
    windows = []
    dropped = 0
    for scene_id, store in stores.items():
        if store.set_split != split:
            continue
        for ego_pid, frames in store.ped_frames.items():
            if not frames:
                continue
            first, last = frames[0], frames[-1]
            start = first
            while start + t_obs + t_pred <= last + 1:
                agent_pids = _agents_in_window(store, start, t_obs, ego_pid, max_agents)
                result = window_scene(store, ego_pid, start, t_obs, t_pred, agent_pids, max_agents)
                if result is None:
                    dropped += 1
                else:
                    _, _, _, target_mask = result
                    if not require_full_future or target_mask.all():
                        windows.append((scene_id, ego_pid, start, agent_pids))
                    else:
                        dropped += 1
                start += stride
    print(f"[{split}] built {len(windows)} windows, dropped {dropped} (ego gap or incomplete future)")
    return windows


@dataclass
class NormStats:
    """cx, cy ONLY (2 channels). w, h are deliberately left as pure
    image-dim-normalized values, never z-standardized: LEMC's scale-ratio cue
    (h[t+1]/h[t]) needs a channel where the affine shift of standardization
    doesn't corrupt the ratio (standardized h can cross zero / flip sign even
    though true height is always positive). cx,cy standardization is safe
    because it only affects displacement/median-style statistics, where the
    shift term cancels in a difference -- see lemc/models/lemc.py."""

    mean: torch.Tensor  # [2]
    std: torch.Tensor  # [2]


class LEMCWindowDataset(Dataset):
    """
    Each item: agent_tracks [t_obs, max_agents, 4], agent_mask [t_obs, max_agents],
    target_track [t_pred, 4], target_mask [t_pred], ego_speed [t_obs] (OBD speed,
    NaN where unavailable e.g. JAAD), crossing_label (float, NaN if unknown).

    Coordinates are always divided by (width, height, width, height) first
    (deterministic image-dim normalization). If `norm_stats` is provided, a
    further per-channel standardization ((x - mean) / std, fit on train only)
    is applied on top -- pass norm_stats=None to get pure image-dim-normalized
    values (used by `fit_norm_stats` below to compute those stats).

    Masked (agent absent) positions are 0.0 in raw pixel units but are NOT
    guaranteed to stay exactly 0.0 after normalization -- always read
    `agent_mask` alongside `agent_tracks`, never infer presence from value.
    """

    def __init__(
        self,
        stores: dict[str, SceneStore],
        windows: list[tuple],
        t_obs: int,
        t_pred: int,
        max_agents: int,
        norm_stats: NormStats | None = None,
    ):
        self.stores = stores
        self.windows = windows
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.max_agents = max_agents
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        scene_id, ego_pid, start_frame, agent_pids = self.windows[idx]
        store = self.stores[scene_id]
        result = window_scene(store, ego_pid, start_frame, self.t_obs, self.t_pred, agent_pids, self.max_agents)
        assert result is not None, "window index should only contain valid windows"
        agent_tracks, agent_mask, target_track, target_mask = result

        dims = np.array([store.width, store.height, store.width, store.height], dtype=np.float32)
        agent_tracks = agent_tracks / dims
        target_track = target_track / dims

        agent_tracks_t = torch.from_numpy(agent_tracks)
        target_track_t = torch.from_numpy(target_track)
        if self.norm_stats is not None:
            # standardize cx,cy only -- w,h stay pure image-dim-normalized (see NormStats docstring)
            xy = (agent_tracks_t[..., :2] - self.norm_stats.mean) / self.norm_stats.std
            agent_tracks_t = torch.cat([xy, agent_tracks_t[..., 2:]], dim=-1)
            txy = (target_track_t[..., :2] - self.norm_stats.mean) / self.norm_stats.std
            target_track_t = torch.cat([txy, target_track_t[..., 2:]], dim=-1)

        # NOTE: must not write `store.frame_to_ego.get(f, np.nan) or np.nan` --
        # `or` treats a legitimate OBD_speed of 0.0 (car stopped) as falsy and
        # silently rewrites it to NaN, corrupting exactly the low-speed windows
        # the OBD correlation validation (plan Step 6) most needs.
        ego_speed = np.array(
            [
                store.frame_to_ego[f] if f in store.frame_to_ego and store.frame_to_ego[f] is not None else np.nan
                for f in range(start_frame, start_frame + self.t_obs)
            ],
            dtype=np.float32,
        )
        attrs = store.ped_attributes.get(ego_pid, {})
        crossing_label = float(attrs.get("crossing", np.nan))
        if crossing_label < 0:
            # PIE encodes "not applicable / unknown" as -1, not NaN -- must not
            # leak into BCE as a bogus target (confirmed: 468/1842 peds are -1).
            crossing_label = np.nan

        return {
            "agent_tracks": agent_tracks_t,
            "agent_mask": torch.from_numpy(agent_mask),
            "target_track": target_track_t,
            "target_mask": torch.from_numpy(target_mask),
            "ego_speed": torch.from_numpy(ego_speed),
            "crossing_label": torch.tensor(crossing_label, dtype=torch.float32),
            "scene_id": scene_id,
            "ego_pid": ego_pid,
            "start_frame": start_frame,
        }


def denormalize_xy(xy_std: torch.Tensor, norm_stats: NormStats, width: float, height: float) -> torch.Tensor:
    """Inverse of the (image-dim divide -> standardize) pipeline, for cx,cy only.
    xy_std: [..., 2] standardized values -> returns pixel-space [..., 2]."""
    mean = norm_stats.mean.to(xy_std.device)
    std = norm_stats.std.to(xy_std.device)
    xy_dimnorm = xy_std * std + mean
    dims = torch.tensor([width, height], dtype=xy_std.dtype, device=xy_std.device)
    return xy_dimnorm * dims


def fit_norm_stats(train_dataset_no_stats: LEMCWindowDataset) -> NormStats:
    """train_dataset_no_stats must be constructed with norm_stats=None. Fits
    cx,cy stats only -- w,h are never standardized, see NormStats docstring."""
    assert train_dataset_no_stats.norm_stats is None
    vals = []
    for i in range(len(train_dataset_no_stats)):
        item = train_dataset_no_stats[i]
        vals.append(item["agent_tracks"][:, 0, :2].numpy())  # ego (idx 0, always valid), cx,cy only
    vals = np.concatenate(vals, axis=0)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormStats(mean=torch.tensor(mean, dtype=torch.float32), std=torch.tensor(std, dtype=torch.float32))
