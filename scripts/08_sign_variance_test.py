"""Step 1 — sign and variance test for LEMC compensation.

Removing camera motion should flatten near-stationary pedestrians:
    var(compensated centres) < var(raw centres)

If compensated variance is systematically higher, the residual sign is wrong
(LEMC is adding the shift instead of subtracting it).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy, protocol_tte_kwargs
from lemc.data.paths import PIE_DATA_ROOT, track_store_path
from lemc.data.track_store import load_track_store
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config

# PIE behavior: action 0 = standing, 1 = walking (pie_data.py _map_text_to_scalar)
STANDING = 0
PIE_WIDTH, PIE_HEIGHT = 1920.0, 1080.0


def _load_action_lookup(data_root: str = PIE_DATA_ROOT) -> dict[tuple[str, str], dict[int, int]]:
    """(scene_id, pid) -> {frame: action_int}. Prefer the cached PIE database."""
    cache = os.path.join(data_root, "data_cache", "pie_database.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            db = pickle.load(f)
    else:
        from lemc.data.pie_loader import load_pie_database

        db = load_pie_database(data_root)

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


def _window_is_standing(
    action_lookup: dict[tuple[str, str], dict[int, int]],
    scene_id: str,
    ego_pid: str,
    start_frame: int,
    t_obs: int,
) -> bool:
    """True iff every observed frame has action=standing for this pedestrian."""
    frame_actions = action_lookup.get((scene_id, ego_pid))
    if frame_actions is None:
        return False
    for f in range(start_frame, start_frame + t_obs):
        if frame_actions.get(f, -1) != STANDING:
            return False
    return True


def _xy_variance(xy: np.ndarray) -> float:
    """Trace of 2D position covariance: var(cx) + var(cy)."""
    return float(np.var(xy[:, 0]) + np.var(xy[:, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--speed-eps", type=float, default=1e-3, help="OBD speed > this counts as moving")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("use_lemc"):
        raise SystemExit("Sign test requires use_lemc=True (need a residual to compensate with).")

    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]
    tte_kw = protocol_tte_kwargs(cfg)
    overlap_eval = cfg["protocol"].get("overlap_eval", 0.0)

    stores = load_track_store(track_store_path(cfg))
    windows = build_window_index(
        stores, args.split, t_obs, t_pred, max_agents, overlap=overlap_eval, **tte_kw
    )

    with open(os.path.join(args.checkpoint_dir, "norm_stats.pkl"), "rb") as f:
        norm_stats = pickle.load(f)

    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=norm_stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=cfg["backbone"].get("predict_intent", True),
    )
    model = TrajectoryPredictor(use_lemc=True, lemc_cfg=cfg.get("lemc", {}), backbone_cfg=backbone_cfg)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best.pt"), map_location="cpu"))
    model.eval()

    print("Loading PIE behavior action labels for standing filter...")
    action_lookup = _load_action_lookup()

    var_raw_all, var_comp_all = [], []
    standing_moving_idx = []  # indices into the all-window lists
    standing_moving_meta = []

    with torch.no_grad():
        global_idx = 0
        for batch in loader:
            agent_tracks = batch["agent_tracks"]
            agent_mask = batch["agent_mask"]
            compensated, _ = model.lemc(agent_tracks, agent_mask)

            raw_xy = denormalize_xy(agent_tracks[:, :, 0, :2], norm_stats, PIE_WIDTH, PIE_HEIGHT)
            comp_xy = denormalize_xy(compensated[..., :2], norm_stats, PIE_WIDTH, PIE_HEIGHT)
            mean_speed = batch["ego_speed"].nanmean(dim=1).numpy()

            B = agent_tracks.shape[0]
            for i in range(B):
                vr = _xy_variance(raw_xy[i].numpy())
                vc = _xy_variance(comp_xy[i].numpy())
                var_raw_all.append(vr)
                var_comp_all.append(vc)

                scene_id = batch["scene_id"][i]
                ego_pid = batch["ego_pid"][i]
                start_frame = int(batch["start_frame"][i])
                speed = float(mean_speed[i])
                is_standing = _window_is_standing(action_lookup, scene_id, ego_pid, start_frame, t_obs)
                moving = (not np.isnan(speed)) and speed > args.speed_eps
                if is_standing and moving:
                    standing_moving_idx.append(global_idx + i)
                    standing_moving_meta.append((scene_id, ego_pid, start_frame, speed, vr, vc))

            global_idx += B

    var_raw = np.asarray(var_raw_all)
    var_comp = np.asarray(var_comp_all)
    improved = var_comp < var_raw
    frac_all = float(improved.mean())

    print(f"=== Sign / variance test ({args.split}, n={len(var_raw)}) ===")
    print(f"  fraction var_comp < var_raw: {frac_all:.4f}  ({improved.sum()}/{len(var_raw)})")
    print(f"  median var_raw={np.median(var_raw):.2f}  var_comp={np.median(var_comp):.2f}")
    print(f"  median ratio var_comp/var_raw={np.median(var_comp / np.maximum(var_raw, 1e-12)):.4f}")

    if standing_moving_idx:
        sm_raw = var_raw[standing_moving_idx]
        sm_comp = var_comp[standing_moving_idx]
        sm_improved = sm_comp < sm_raw
        frac_sm = float(sm_improved.mean())
        print(
            f"=== Standing + OBD speed > {args.speed_eps} "
            f"(n={len(standing_moving_idx)}) ==="
        )
        print(f"  fraction var_comp < var_raw: {frac_sm:.4f}  ({sm_improved.sum()}/{len(sm_raw)})")
        print(f"  median var_raw={np.median(sm_raw):.2f}  var_comp={np.median(sm_comp):.2f}")
        print(
            f"  median ratio var_comp/var_raw="
            f"{np.median(sm_comp / np.maximum(sm_raw, 1e-12)):.4f}"
        )
        gate = "PASS" if frac_sm > 0.5 else "FAIL — likely sign error"
        print(f"  GATE (standing+moving must mostly flatten): {gate}")
    else:
        frac_sm = float("nan")
        print("=== Standing + moving: no windows found ===")

    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = os.path.basename(args.checkpoint_dir.rstrip("/"))
    out_csv = os.path.join(args.out_dir, f"sign_variance_{run_tag}_{args.split}.csv")
    np.savetxt(
        out_csv,
        np.stack([var_raw, var_comp, improved.astype(np.float64)], axis=1),
        header="var_raw,var_comp,comp_lower",
        delimiter=",",
        comments="",
    )
    print(f"  per-window vars saved to {out_csv}")

    summary_path = os.path.join(args.out_dir, f"sign_variance_{run_tag}_{args.split}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_all={len(var_raw)}\n")
        f.write(f"frac_comp_lower_all={frac_all:.6f}\n")
        f.write(f"n_standing_moving={len(standing_moving_idx)}\n")
        f.write(f"frac_comp_lower_standing_moving={frac_sm:.6f}\n")
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
    main()
