import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy
from lemc.data.paths import track_store_path
from lemc.data.track_store import load_track_store
from lemc.eval.metrics import intent_metrics, trajectory_metrics
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config

# PIE images are all 1920x1080; JAAD may differ once its dims are confirmed (plan Step 9).
PIE_WIDTH, PIE_HEIGHT = 1920.0, 1080.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_obs, t_pred, fps = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"], cfg["protocol"]["fps"]
    max_agents = cfg["max_agents"]
    use_lemc = cfg.get("use_lemc", False)

    stores = load_track_store(track_store_path(cfg))
    windows = build_window_index(stores, args.split, t_obs, t_pred, max_agents, overlap=0.0)

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
    lemc_cfg = cfg.get("lemc", {})
    model = TrajectoryPredictor(use_lemc=use_lemc, lemc_cfg=lemc_cfg, backbone_cfg=backbone_cfg)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best.pt"), map_location="cpu"))
    model.eval()

    width, height = (PIE_WIDTH, PIE_HEIGHT) if cfg["dataset"] == "pie" else (None, None)
    if width is None:
        raise NotImplementedError("Confirm JAAD image dims before evaluating on JAAD (plan Step 9).")

    all_pred_px, all_target_px, all_target_mask = [], [], []
    all_intent_probs, all_intent_labels = [], []
    all_residual_mag = []

    with torch.no_grad():
        for batch in loader:
            traj_pred, intent_logit, residual = model(batch["agent_tracks"], batch["agent_mask"])

            all_pred_px.append(denormalize_xy(traj_pred, norm_stats, width, height))
            all_target_px.append(denormalize_xy(batch["target_track"][:, :, :2], norm_stats, width, height))
            all_target_mask.append(batch["target_mask"])

            if intent_logit is not None:
                probs = torch.sigmoid(intent_logit).squeeze(-1)
                labels = batch["crossing_label"]
                valid = ~torch.isnan(labels)
                all_intent_probs.append(probs[valid])
                all_intent_labels.append(labels[valid])

            if residual is not None:
                all_residual_mag.append(residual.norm(dim=-1).flatten())

    pred_px = torch.cat(all_pred_px)
    target_px = torch.cat(all_target_px)
    target_mask = torch.cat(all_target_mask)

    traj_results = trajectory_metrics(pred_px, target_px, target_mask, fps, t_pred)
    print(f"=== Trajectory metrics ({cfg['dataset']}, {args.split}, n={len(ds)}) ===")
    for k, v in traj_results.items():
        print(f"  {k}: {v:.4f}")

    if all_intent_probs:
        probs = torch.cat(all_intent_probs).numpy()
        labels = torch.cat(all_intent_labels).numpy()
        if len(probs) > 0:
            intent_results = intent_metrics(probs, labels)
            print(f"=== Intent metrics ({cfg['dataset']}, {args.split}, n={len(probs)}) ===")
            for k, v in intent_results.items():
                print(f"  {k}: {v:.4f}")
        else:
            print("No windows with a known crossing label in this split -- skipping intent metrics.")

    if all_residual_mag:
        mag = torch.cat(all_residual_mag)
        print(f"=== LEMC residual magnitude ({args.split}) === mean={mag.mean():.5f} std={mag.std():.5f}")


if __name__ == "__main__":
    main()
