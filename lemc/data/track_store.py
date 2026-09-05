from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np

from .paths import JAAD_DATA_ROOT, PIE_DATA_ROOT
from .pie_loader import load_pie_database, load_pie_splits
from .jaad_loader import load_jaad_database, load_jaad_splits, resolve_jaad_root

EGO_IDX = 0  # convention: agent_pids[0] is always the ego pedestrian


@dataclass
class SceneStore:
    scene_id: str
    width: int
    height: int
    set_split: str  # 'train' | 'val' | 'test'
    frame_to_boxes: dict = field(default_factory=dict)  # frame -> {pid: (cx, cy, w, h)}
    frame_to_ego: dict = field(default_factory=dict)  # frame -> OBD_speed (PIE only)
    frame_to_vehicle_action: dict = field(default_factory=dict)  # frame -> int (JAAD only)
    ped_attributes: dict = field(default_factory=dict)  # pid -> attrs dict
    ped_frames: dict = field(default_factory=dict)  # pid -> sorted list of frames present


def _corners_to_center(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (cx, cy, w, h)


def _split_for_set(set_id: str, splits: dict) -> str | None:
    for split, sids in splits.items():
        if set_id in sids:
            return split
    return None


def build_pie_track_store(data_root: str = PIE_DATA_ROOT) -> dict[str, SceneStore]:
    db = load_pie_database(data_root)
    splits = load_pie_splits(data_root)

    stores: dict[str, SceneStore] = {}
    for set_id, videos in db.items():
        split = _split_for_set(set_id, splits)
        for video_id, scene in videos.items():
            scene_id = f"{set_id}/{video_id}"
            store = SceneStore(
                scene_id=scene_id,
                width=scene["width"],
                height=scene["height"],
                set_split=split,
            )

            for pid, rec in scene["ped_annotations"].items():
                frames = rec["frames"]
                boxes = rec["bbox"]
                store.ped_frames[pid] = list(frames)
                store.ped_attributes[pid] = rec.get("attributes", {})
                for f, box in zip(frames, boxes):
                    store.frame_to_boxes.setdefault(f, {})[pid] = _corners_to_center(box)

            for f, rec in scene.get("vehicle_annotations", {}).items():
                store.frame_to_ego[f] = rec.get("OBD_speed")

            stores[scene_id] = store

    return stores



def build_jaad_track_store(data_root: str | None = None) -> dict[str, SceneStore]:
    """Build SceneStore dict from JAAD 2.0 annotations (no OBD; vehicle action ints).

    scene_id is the video id (e.g. video_0001). Only pedestrians (ids containing
    no trailing group marker 'p') that appear in ped_annotations are ego candidates;
    bystander 'ped' tracks without behavior are still available as other agents.
    """
    root = resolve_jaad_root(data_root)
    db = load_jaad_database(root)
    splits = load_jaad_splits(root)
    vid_to_split = {}
    for split, vids in splits.items():
        for vid in vids:
            vid_to_split[vid] = split

    stores: dict[str, SceneStore] = {}
    for vid, scene in db.items():
        split = vid_to_split.get(vid)
        if split is None:
            continue  # video not in default split files
        store = SceneStore(
            scene_id=vid,
            width=int(scene.get("width", 1920)),
            height=int(scene.get("height", 1080)),
            set_split=split,
        )
        for pid, rec in scene.get("ped_annotations", {}).items():
            # Skip 'people' group boxes (id ends with 'p') — not single-agent targets
            if pid.endswith("p"):
                continue
            frames = list(rec["frames"])
            boxes = rec["bbox"]
            store.ped_frames[pid] = frames
            attrs = dict(rec.get("attributes") or {})
            store.ped_attributes[pid] = attrs
            for f, box in zip(frames, boxes):
                store.frame_to_boxes.setdefault(int(f), {})[pid] = _corners_to_center(box)

        veh = scene.get("vehicle_annotations") or {}
        for f, action in veh.items():
            store.frame_to_vehicle_action[int(f)] = int(action)

        stores[vid] = store
    return stores


def save_track_store(stores: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_track_store(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_detections_into_stores(stores: dict[str, SceneStore], detections: dict[str, dict[int, dict]]) -> None:
    """Mutates stores in place: adds detector-tracked pseudo-agents (cars,
    traffic lights, etc. from scripts/06_detect_objects.py) into
    frame_to_boxes alongside the real pedestrian boxes. Pseudo-agent ids are
    prefixed (f"{scene}_det_{track_id}") so they can never collide with a
    real PIE pedestrian id.

    Deliberately does NOT touch ped_frames: detected objects must never
    become an "ego" prediction target in build_window_index (which iterates
    ped_frames.items() for that), only additional "other agents" for LEMC's
    shared-motion statistic (which scans frame_to_boxes directly)."""
    for scene_id, dets_by_frame in detections.items():
        if scene_id not in stores:
            continue
        store = stores[scene_id]
        for frame, boxes in dets_by_frame.items():
            store.frame_to_boxes.setdefault(frame, {}).update(boxes)


def window_scene(
    store: SceneStore,
    ego_pid: str,
    start_frame: int,
    t_obs: int,
    t_pred: int,
    agent_pids: list[str],
    max_agents: int,
):
    """
    Returns (agent_tracks, agent_mask, target_track, target_mask) or None if the
    ego pedestrian is missing a box at any observed frame in [start_frame, start_frame+t_obs)
    (such windows are dropped, never interpolated).

    agent_tracks: [t_obs, max_agents, 4] float32 (cx, cy, w, h), 0.0-filled where masked.
    agent_mask:   [t_obs, max_agents] bool. False entries must never be read as real
                  boxes -- a real box can legitimately sit near (0,0,...) at a frame edge.
    target_track: [t_pred, 4] float32, the ego pedestrian's future track.
    target_mask:  [t_pred] bool -- future can also be partially missing near track edges.

    agent_pids[0] must be ego_pid (EGO_IDX convention).
    """
    assert agent_pids[EGO_IDX] == ego_pid

    obs_frames = range(start_frame, start_frame + t_obs)
    pred_frames = range(start_frame + t_obs, start_frame + t_obs + t_pred)

    agent_tracks = np.zeros((t_obs, max_agents, 4), dtype=np.float32)
    agent_mask = np.zeros((t_obs, max_agents), dtype=bool)

    for t, f in enumerate(obs_frames):
        boxes_at_f = store.frame_to_boxes.get(f, {})
        if ego_pid not in boxes_at_f:
            return None  # ego gap -> drop window, do not interpolate
        for n, pid in enumerate(agent_pids[:max_agents]):
            if pid in boxes_at_f:
                agent_tracks[t, n] = boxes_at_f[pid]
                agent_mask[t, n] = True

    target_track = np.zeros((t_pred, 4), dtype=np.float32)
    target_mask = np.zeros((t_pred,), dtype=bool)
    for t, f in enumerate(pred_frames):
        boxes_at_f = store.frame_to_boxes.get(f, {})
        if ego_pid in boxes_at_f:
            target_track[t] = boxes_at_f[ego_pid]
            target_mask[t] = True

    return agent_tracks, agent_mask, target_track, target_mask
