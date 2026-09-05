"""Step 11 — Stratified ADE/FDE by ego state (accelerating / constant / decelerating).

Labels each test window from OBD speed finite differences over the observation
window. Reports metrics for a GT-speed model vs ORB-LEMC (rows 2 and 4).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy, protocol_tte_kwargs
from lemc.data.paths import track_store_path
from lemc.data.track_store import load_track_store
from lemc.eval.metrics import intent_metrics, trajectory_metrics
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config

PIE_WIDTH, PIE_HEIGHT = 1920.0, 1080.0


def ego_state_labels(ego_speed: torch.Tensor, accel_thresh: float = 0.15) -> np.ndarray:
    """ego_speed: [B, T]. Returns list of 'accelerating'|'constant'|'decelerating'."""
    # finite diff of nan-filled speeds; require enough finite samples
    labels = []
    for i in range(ego_speed.shape[0]):
        s = ego_speed[i].numpy()
        valid = np.isfinite(s)
        if valid.sum() < 3:
            labels.append("unknown")
            continue
        # linear slope over valid frames
        t = np.arange(len(s), dtype=np.float64)[valid]
        y = s[valid]
        slope = np.polyfit(t, y, 1)[0]  # speed units per frame
        if slope > accel_thresh:
            labels.append("accelerating")
        elif slope < -accel_thresh:
            labels.append("decelerating")
        else:
            labels.append("constant")
    return np.array(labels)


def eval_subset(model, ds, indices, cfg, norm_stats, use_speed: bool):
    if len(indices) == 0:
        return None
    loader = DataLoader(Subset(ds, indices), batch_size=64, shuffle=False)
    t_pred = cfg["protocol"]["t_pred"]
    fps = cfg["protocol"]["fps"]
    preds, tgts, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            ego_speed = batch["ego_speed"] if use_speed else None
            traj_pred, _, _ = model(batch["agent_tracks"], batch["agent_mask"], ego_speed=ego_speed)
            preds.append(denormalize_xy(traj_pred, norm_stats, PIE_WIDTH, PIE_HEIGHT))
            tgts.append(denormalize_xy(batch["target_track"][:, :, :2], norm_stats, PIE_WIDTH, PIE_HEIGHT))
            masks.append(batch["target_mask"])
    return trajectory_metrics(torch.cat(preds), torch.cat(tgts), torch.cat(masks), fps, t_pred)


def load_model(cfg, ckpt_dir):
    use_lemc = cfg.get("use_lemc", False)
    use_speed = cfg.get("use_speed_input", False) or cfg["backbone"].get("use_speed_input", False)
    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=cfg["protocol"]["t_pred"],
        predict_intent=cfg["backbone"].get("predict_intent", True),
        use_speed_input=use_speed,
    )
    model = TrajectoryPredictor(use_lemc=use_lemc, lemc_cfg=cfg.get("lemc", {}), backbone_cfg=backbone_cfg)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu"))
    model.eval()
    with open(os.path.join(ckpt_dir, "norm_stats.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    return model, norm_stats, use_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-config", required=True)
    parser.add_argument("--speed-ckpt", required=True)
    parser.add_argument("--orb-config", required=True)
    parser.add_argument("--orb-ckpt", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg_s = load_config(args.speed_config)
    cfg_o = load_config(args.orb_config)
    # use ORB config's store/protocol for shared windows
    cfg = cfg_o
    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]
    tte_kw = protocol_tte_kwargs(cfg)
    overlap = cfg["protocol"].get("overlap_eval", 0.5)

    stores = load_track_store(track_store_path(cfg))
    windows = build_window_index(stores, args.split, t_obs, t_pred, max_agents, overlap=overlap, **tte_kw)
    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)

    # label every window (raw speeds, no norm needed)
    speeds = torch.stack([ds[i]["ego_speed"] for i in range(len(ds))])
    labels = ego_state_labels(speeds)
    print("=== Ego-state counts ===")
    for name in ("accelerating", "constant", "decelerating", "unknown"):
        print(f"  {name}: {(labels == name).sum()}")

    model_s, stats_s, use_s = load_model(cfg_s, args.speed_ckpt)
    model_o, stats_o, use_o = load_model(cfg_o, args.orb_ckpt)
    ds_s = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=stats_s)
    ds_o = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=stats_o)

    print("=== Stratified ADE/FDE ===")
    for name in ("accelerating", "constant", "decelerating"):
        idx = np.where(labels == name)[0].tolist()
        rs = eval_subset(model_s, ds_s, idx, cfg_s, stats_s, use_s)
        ro = eval_subset(model_o, ds_o, idx, cfg_o, stats_o, use_o)
        if rs is None or ro is None:
            continue
        print(
            f"  {name} (n={len(idx)}): "
            f"speed ADE={rs['ade']:.2f} FDE={rs['fde']:.2f} | "
            f"ORB-LEMC ADE={ro['ade']:.2f} FDE={ro['fde']:.2f}"
        )


if __name__ == "__main__":
    main()
