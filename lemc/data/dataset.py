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
    tte_min: int | None = None,
    tte_max: int | None = None,
) -> list[tuple]:
    """
    Returns a list of (scene_id, ego_pid, start_frame, agent_pids) tuples for
    every valid window in the given split. A window is valid iff window_scene
    doesn't drop it (ego present every observed frame) and, if
    require_full_future, the target track is fully present too.

    When tte_min/tte_max are set (Zhang / BiPeds field protocol), keep only
    windows whose time-to-event
        tte = crossing_point - (start_frame + t_obs)
    lies in [tte_min, tte_max]. Pedestrians without a usable crossing_point
    are skipped under TTE filtering.
    """
    stride = t_obs if overlap == 0 else max(1, int((1 - overlap) * t_obs))
    use_tte = tte_min is not None and tte_max is not None
    windows = []
    dropped = 0
    dropped_tte = 0
    for scene_id, store in stores.items():
        if store.set_split != split:
            continue
        for ego_pid, frames in store.ped_frames.items():
            if not frames:
                continue
            crossing_point = None
            if use_tte:
                attrs = store.ped_attributes.get(ego_pid, {})
                cp = attrs.get("crossing_point")
                if cp is None or int(cp) < 0:
                    dropped_tte += 1
                    continue
                crossing_point = int(cp)
            first, last = frames[0], frames[-1]
            start = first
            while start + t_obs + t_pred <= last + 1:
                if use_tte:
                    tte = crossing_point - (start + t_obs)
                    if tte < tte_min or tte > tte_max:
                        dropped_tte += 1
                        start += stride
                        continue
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
    tte_msg = f", dropped_tte={dropped_tte}" if use_tte else ""
    print(f"[{split}] built {len(windows)} windows, dropped {dropped} (ego gap or incomplete future){tte_msg}")
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


def protocol_tte_kwargs(cfg: dict) -> dict:
    """Pull TTE bounds from config when present (field protocol)."""
    prot = cfg.get("protocol", {})
    tte_min, tte_max = prot.get("tte_min"), prot.get("tte_max")
    if tte_min is None or tte_max is None:
        return {}
    return {"tte_min": int(tte_min), "tte_max": int(tte_max)}


class LEMCWindowDataset(Dataset):
    """
    Each item: agent_tracks [t_obs, max_agents, 4], agent_mask [t_obs, max_agents],
    target_track [t_pred, 4], target_mask [t_pred], ego_speed [t_obs] (OBD speed,
    NaN where unavailable e.g. JAAD), crossing_label (float, NaN if unknown),
    orb_shift [t_obs-1, 2] + orb_mask [t_obs-1] when an ORB cache is attached.

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
        orb_shifts: dict | None = None,
        orb_affines: dict | None = None,
    ):
        self.stores = stores
        self.windows = windows
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.max_agents = max_agents
        self.norm_stats = norm_stats
        # scene_id -> {frame_t: (dx_px, dy_px)} for transition t -> t+1; NaN if failed
        self.orb_shifts = orb_shifts
        # scene_id -> {frame_t: (2,3) affine np.ndarray} for transition t -> t+1
        self.orb_affines = orb_affines

    def __len__(self) -> int:
        return len(self.windows)

    def _affine_disp(self, scene_id: str, frame: int, cx: float, cy: float):
        """Pixel displacement induced by affine at `frame` for point (cx,cy)."""
        if self.orb_affines is None:
            return None
        M = self.orb_affines.get(scene_id, {}).get(frame)
        if M is None:
            return None
        mapped = M @ np.array([cx, cy, 1.0], dtype=np.float64)
        dx, dy = float(mapped[0] - cx), float(mapped[1] - cy)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return None
        return dx, dy

    def _to_residual_units(self, dx_px: float, dy_px: float, width: float, height: float):
        dx_n, dy_n = dx_px / width, dy_px / height
        if self.norm_stats is not None:
            dx_n = dx_n / float(self.norm_stats.std[0])
            dy_n = dy_n / float(self.norm_stats.std[1])
        return dx_n, dy_n

    def _orb_window(self, scene_id: str, start_frame: int, width: float, height: float, ego_xy_px: np.ndarray):
        """Return (orb_shift [T-1,2] in residual units, orb_mask [T-1])."""
        Tm1 = self.t_obs - 1
        shift = np.zeros((Tm1, 2), dtype=np.float32)
        mask = np.zeros((Tm1,), dtype=bool)

        for i, f in enumerate(range(start_frame, start_frame + Tm1)):
            dx_px = dy_px = None
            disp = self._affine_disp(scene_id, f, float(ego_xy_px[i, 0]), float(ego_xy_px[i, 1]))
            if disp is not None:
                dx_px, dy_px = disp
            elif self.orb_shifts is not None:
                val = self.orb_shifts.get(scene_id, {}).get(f)
                if val is not None:
                    dx_px, dy_px = float(val[0]), float(val[1])

            if dx_px is None or not (np.isfinite(dx_px) and np.isfinite(dy_px)):
                continue
            shift[i] = self._to_residual_units(dx_px, dy_px, width, height)
            mask[i] = True
        return torch.from_numpy(shift), torch.from_numpy(mask)

    def _camera_cumsum(
        self,
        scene_id: str,
        start_frame: int,
        n_steps: int,
        width: float,
        height: float,
        xy_px: np.ndarray,
    ):
        """Absolute camera shift [n_steps, 2] relative to frame start (0 at t=0).

        xy_px: centres at frames start .. start+n_steps-1 (length n_steps).
        Uses hold-last for missing affine transitions.
        """
        out = np.zeros((n_steps, 2), dtype=np.float32)
        cum = np.zeros(2, dtype=np.float32)
        last = np.zeros(2, dtype=np.float32)
        for i in range(n_steps - 1):
            f = start_frame + i
            disp = self._affine_disp(scene_id, f, float(xy_px[i, 0]), float(xy_px[i, 1]))
            if disp is None and self.orb_shifts is not None:
                val = self.orb_shifts.get(scene_id, {}).get(f)
                if val is not None:
                    disp = (float(val[0]), float(val[1]))
            if disp is not None:
                last = np.array(self._to_residual_units(disp[0], disp[1], width, height), dtype=np.float32)
            cum = cum + last  # hold-last if missing
            out[i + 1] = cum
        return torch.from_numpy(out)

    def __getitem__(self, idx: int) -> dict:
        scene_id, ego_pid, start_frame, agent_pids = self.windows[idx]
        store = self.stores[scene_id]
        result = window_scene(store, ego_pid, start_frame, self.t_obs, self.t_pred, agent_pids, self.max_agents)
        assert result is not None, "window index should only contain valid windows"
        agent_tracks, agent_mask, target_track, target_mask = result

        # pixel ego centres for affine target (before normalization) — obs + future
        ego_xy_px = agent_tracks[:, 0, :2].copy()
        target_xy_px = target_track[:, :2].copy()
        full_xy_px = np.concatenate([ego_xy_px, target_xy_px], axis=0)

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

        orb_shift, orb_mask = self._orb_window(
            scene_id, start_frame, store.width, store.height, ego_xy_px
        )
        cam_full = self._camera_cumsum(
            scene_id, start_frame, self.t_obs + self.t_pred, store.width, store.height, full_xy_px
        )
        camera_shift_obs = cam_full[: self.t_obs]
        camera_shift_pred = cam_full[self.t_obs :]

        return {
            "agent_tracks": agent_tracks_t,
            "agent_mask": torch.from_numpy(agent_mask),
            "target_track": target_track_t,
            "target_mask": torch.from_numpy(target_mask),
            "ego_speed": torch.from_numpy(ego_speed),
            "crossing_label": torch.tensor(crossing_label, dtype=torch.float32),
            "orb_shift": orb_shift,
            "orb_mask": orb_mask,
            "camera_shift_obs": camera_shift_obs,
            "camera_shift_pred": camera_shift_pred,
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
