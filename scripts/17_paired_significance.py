"""Paired per-window ADE: baseline vs fixed ORB (Wilcoxon signed-rank)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy, protocol_tte_kwargs
from lemc.data.paths import TRACK_STORE_DIR, track_store_path
from lemc.data.track_store import load_track_store
from lemc.eval.metrics import per_window_ade
from lemc.models.backbone import GRUTrajectoryModel
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config
from lemc.utils.device import get_device, move_batch

IMG_W, IMG_H = 1920.0, 1080.0


def _window_ades(model, ds, device, use_speed=False, fixed_orb=False):
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    ades = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if fixed_orb:
                ego_raw = batch["agent_tracks"][:, :, 0, :]
                ego_comp = torch.cat(
                    [ego_raw[..., :2] - batch["camera_shift_obs"], ego_raw[..., 2:]], dim=-1
                )
                traj, _ = model.backbone(ego_comp)
                traj = traj + batch["camera_shift_pred"]
            else:
                ego_speed = batch["ego_speed"] if use_speed else None
                traj, _, _ = model(batch["agent_tracks"], batch["agent_mask"], ego_speed=ego_speed)
            pred = denormalize_xy(traj, ds.norm_stats, IMG_W, IMG_H)
            tgt = denormalize_xy(batch["target_track"][:, :, :2], ds.norm_stats, IMG_W, IMG_H)
            ades.append(per_window_ade(pred, tgt, batch["target_mask"]))
    return torch.cat(ades).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/pie_nolemc_augmented.yaml")
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--orb-config", default="configs/pie_fixed_orb_affine.yaml")
    parser.add_argument("--orb-ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    cfg_b = load_config(args.baseline_config)
    cfg_o = load_config(args.orb_config)
    t_obs, t_pred = cfg_b["protocol"]["t_obs"], cfg_b["protocol"]["t_pred"]
    max_agents = cfg_b["max_agents"]
    tte_kw = protocol_tte_kwargs(cfg_b)
    overlap = cfg_b["protocol"].get("overlap_eval", 0.5)

    stores = load_track_store(track_store_path(cfg_b))
    windows = build_window_index(stores, "test", t_obs, t_pred, max_agents, overlap=overlap, **tte_kw)

    with open(os.path.join(args.baseline_ckpt, "norm_stats.pkl"), "rb") as f:
        ns_b = pickle.load(f)
    with open(os.path.join(args.orb_ckpt, "norm_stats.pkl"), "rb") as f:
        ns_o = pickle.load(f)

    ds_b = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=ns_b)
    aff_path = os.path.join(TRACK_STORE_DIR, cfg_o.get("orb_affines_file", "pie_orb_affines.pkl"))
    with open(aff_path, "rb") as f:
        orb_affines = pickle.load(f)
    ds_o = LEMCWindowDataset(
        stores, windows, t_obs, t_pred, max_agents, norm_stats=ns_o, orb_affines=orb_affines
    )

    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg_b["backbone"]["hidden_dim"],
        num_layers=cfg_b["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=True,
    )
    model_b = TrajectoryPredictor(use_lemc=False, lemc_cfg={}, backbone_cfg=backbone_cfg)
    model_b.load_state_dict(torch.load(os.path.join(args.baseline_ckpt, "best.pt"), map_location=device))
    model_b.to(device).eval()

    orb_backbone = GRUTrajectoryModel(**backbone_cfg)
    state = torch.load(os.path.join(args.orb_ckpt, "best.pt"), map_location=device)
    orb_backbone.load_state_dict(
        {k.replace("backbone.", "", 1): v for k, v in state.items() if k.startswith("backbone.")}
    )

    class OrbWrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.backbone = bb

    orb_model = OrbWrapper(orb_backbone).to(device).eval()

    ade_b = _window_ades(model_b, ds_b, device)
    ade_o = _window_ades(orb_model, ds_o, device, fixed_orb=True)
    stat, p = wilcoxon(ade_b, ade_o)
    diff = ade_o - ade_b
    print(f"n={len(ade_b)} baseline ADE mean={ade_b.mean():.2f} fixed ORB mean={ade_o.mean():.2f}")
    print(f"paired diff (ORB-baseline) mean={diff.mean():.2f} median={np.median(diff):.2f}")
    print(f"Wilcoxon stat={stat:.1f} p={p:.4g}")


if __name__ == "__main__":
    main()
