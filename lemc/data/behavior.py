"""Pedestrian action / vehicle-state lookups for sign-gate and stratified eval."""
from __future__ import annotations

import os
import pickle

STANDING = 0  # PIE + JAAD pedestrian action code


def load_ped_action_lookup(dataset: str, data_root: str | None = None) -> dict[tuple[str, str], dict[int, int]]:
    """(scene_id, pid) -> {frame: action_int}."""
    if dataset == "pie":
        from .paths import PIE_DATA_ROOT
        from .pie_loader import load_pie_database

        root = data_root or PIE_DATA_ROOT
        cache = os.path.join(root, "data_cache", "pie_database.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                db = pickle.load(f)
        else:
            db = load_pie_database(root)
        lookup: dict[tuple[str, str], dict[int, int]] = {}
        for set_id, videos in db.items():
            for video_id, scene in videos.items():
                scene_id = f"{set_id}/{video_id}"
                for pid, rec in scene["ped_annotations"].items():
                    if "behavior" not in rec:
                        continue
                    actions = rec["behavior"]["action"]
                    frames = rec["frames"]
                    lookup[(scene_id, pid)] = {int(f): int(a) for f, a in zip(frames, actions)}
        return lookup

    if dataset == "jaad":
        from .jaad_loader import load_jaad_database, resolve_jaad_root

        root = resolve_jaad_root(data_root)
        db = load_jaad_database(root)
        lookup = {}
        for vid, scene in db.items():
            for pid, rec in scene.get("ped_annotations", {}).items():
                beh = rec.get("behavior") or {}
                actions = beh.get("action")
                frames = rec.get("frames")
                if actions is None or frames is None:
                    continue
                lookup[(vid, pid)] = {int(f): int(a) for f, a in zip(frames, actions)}
        return lookup

    raise ValueError(f"unknown dataset {dataset}")


def window_is_standing(
    action_lookup: dict[tuple[str, str], dict[int, int]],
    scene_id: str,
    ego_pid: str,
    start_frame: int,
    t_obs: int,
) -> bool:
    frame_actions = action_lookup.get((scene_id, ego_pid))
    if frame_actions is None:
        return False
    for f in range(start_frame, start_frame + t_obs):
        if frame_actions.get(f, -1) != STANDING:
            return False
    return True
