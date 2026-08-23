import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy
from lemc.data.paths import track_store_path
from lemc.data.track_store import load_track_store
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["dataset"] != "pie":
        raise SystemExit(
            "OBD-speed validation is PIE-only (JAAD has no speed sensor by definition -- "
            "that's the whole reason LEMC exists). Refusing to run on "
            f"'{cfg['dataset']}'."
        )
    if not cfg.get("use_lemc"):
        raise SystemExit("This checkpoint was trained with use_lemc=False -- nothing to validate (no residual).")

    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]

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
    model = TrajectoryPredictor(use_lemc=True, lemc_cfg=cfg.get("lemc", {}), backbone_cfg=backbone_cfg)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best.pt"), map_location="cpu"))
    model.eval()

    residual_mag_per_window = []
    obd_speed_per_window = []
    global_idx_per_window = []  # ds index, so we can revisit specific windows for the qualitative figure

    with torch.no_grad():
        global_idx = 0
        for batch in loader:
            _, _, residual = model(batch["agent_tracks"], batch["agent_mask"])
            mean_mag = residual.norm(dim=-1).mean(dim=1)  # [B], mean residual magnitude over the window
            mean_speed = batch["ego_speed"].nanmean(dim=1)  # [B], nan when the whole window has no OBD reading

            batch_size = mean_speed.shape[0]
            idx_in_batch = np.arange(global_idx, global_idx + batch_size)
            global_idx += batch_size

            valid = ~torch.isnan(mean_speed)
            residual_mag_per_window.append(mean_mag[valid].numpy())
            obd_speed_per_window.append(mean_speed[valid].numpy())
            global_idx_per_window.append(idx_in_batch[valid.numpy()])

    residual_mag = np.concatenate(residual_mag_per_window)
    obd_speed = np.concatenate(obd_speed_per_window)
    ds_indices = np.concatenate(global_idx_per_window)

    # qualitative figure: pick a real spread across the speed distribution (10th/40th/60th/90th
    # percentile), not just the first few windows encountered in iteration order.
    example_records = []
    for q in (10, 40, 60, 90):
        target_speed = np.percentile(obd_speed, q)
        pick = np.argmin(np.abs(obd_speed - target_speed))
        ds_idx = int(ds_indices[pick])
        item = ds[ds_idx]
        with torch.no_grad():
            agent_tracks = item["agent_tracks"].unsqueeze(0)
            agent_mask = item["agent_mask"].unsqueeze(0)
            compensated, _ = model.lemc(agent_tracks, agent_mask)
        raw_ego = agent_tracks[0, :, 0, :2]
        example_records.append(
            {
                "raw": denormalize_xy(raw_ego, norm_stats, 1920.0, 1080.0).numpy(),
                "compensated": denormalize_xy(compensated[0, :, :2], norm_stats, 1920.0, 1080.0).numpy(),
                "speed": float(obd_speed[pick]),
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)

    r, p = pearsonr(residual_mag, obd_speed)
    r2 = r**2
    print(f"=== LEMC residual vs OBD speed correlation ({args.split}, n={len(residual_mag)}) ===")
    print(f"  Pearson r = {r:.4f}, R^2 = {r2:.4f}, p = {p:.2e}")

    run_tag = os.path.basename(args.checkpoint_dir.rstrip("/"))  # e.g. pie_lemc_seed0 or pie_lemc_augmented_seed0
    csv_path = os.path.join(args.out_dir, f"lemc_obd_correlation_{run_tag}_{args.split}.csv")
    np.savetxt(
        csv_path,
        np.stack([residual_mag, obd_speed], axis=1),
        header="residual_magnitude,obd_speed",
        delimiter=",",
        comments="",
    )
    print(f"  raw data saved to {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(obd_speed, residual_mag, s=8, alpha=0.3)
    coeffs = np.polyfit(obd_speed, residual_mag, 1)
    xs = np.linspace(obd_speed.min(), obd_speed.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="red", linewidth=2)
    ax.set_xlabel("mean OBD speed over window")
    ax.set_ylabel("mean LEMC residual magnitude over window")
    ax.set_title(f"LEMC residual vs ego speed (R²={r2:.3f}, {args.split})")
    fig.tight_layout()
    scatter_path = os.path.join(args.out_dir, f"lemc_obd_scatter_{run_tag}_{args.split}.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  scatter plot saved to {scatter_path}")

    # qualitative figure: raw vs compensated ego track, spanning low -> high ego speed
    fig, axes = plt.subplots(1, len(example_records), figsize=(4 * len(example_records), 4))
    if len(example_records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, example_records):
        ax.plot(rec["raw"][:, 0], rec["raw"][:, 1], "o-", label="raw", alpha=0.7)
        ax.plot(rec["compensated"][:, 0], rec["compensated"][:, 1], "s-", label="LEMC-compensated", alpha=0.7)
        ax.invert_yaxis()  # image coordinates: y grows downward
        ax.set_title(f"mean speed={rec['speed']:.2f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    qual_path = os.path.join(args.out_dir, f"lemc_qualitative_{run_tag}_{args.split}.png")
    fig.savefig(qual_path, dpi=150)
    plt.close(fig)
    print(f"  qualitative before/after figure saved to {qual_path}")


if __name__ == "__main__":
    main()
