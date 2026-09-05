import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, fit_norm_stats, protocol_tte_kwargs
from lemc.data.paths import TRACK_STORE_DIR, track_store_path
from lemc.data.track_store import load_track_store
from lemc.models.predictor import TrajectoryPredictor
from lemc.train.loop import CollapseMonitor, reference_displacement_scale, run_epoch
from lemc.train.seeds import seed_everything
from lemc.utils.config import load_config, run_name as build_run_name
from lemc.utils.device import get_device
from lemc.utils.logging import CSVLogger


def _load_orb_affines(cfg: dict):
    fname = cfg.get("orb_affines_file") or cfg.get("orb_shifts_file")
    if not fname:
        return None, None
    # prefer affine cache when present
    affine_name = cfg.get("orb_affines_file", "pie_orb_affines.pkl")
    affine_path = affine_name if os.path.isabs(affine_name) else os.path.join(TRACK_STORE_DIR, affine_name)
    if os.path.exists(affine_path):
        with open(affine_path, "rb") as f:
            return None, pickle.load(f)
    path = fname if os.path.isabs(fname) else os.path.join(TRACK_STORE_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ORB cache missing ({affine_path} and {path}) — run scripts/09b_extract_orb_affine.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["protocol"]["seeds"][0]
    seed_everything(seed)

    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]
    use_lemc = cfg.get("use_lemc", False)
    tte_kw = protocol_tte_kwargs(cfg)
    overlap_eval = cfg["protocol"].get("overlap_eval", 0.0)

    stores = load_track_store(track_store_path(cfg))
    orb_shifts, orb_affines = (None, None)
    if cfg["loss"].get("orb_weight", 0.0):
        orb_shifts, orb_affines = _load_orb_affines(cfg)

    train_windows = build_window_index(
        stores,
        "train",
        t_obs,
        t_pred,
        max_agents,
        overlap=cfg["train"].get("overlap_train", 0.5),
        **tte_kw,
    )
    val_windows = build_window_index(
        stores, "val", t_obs, t_pred, max_agents, overlap=overlap_eval, **tte_kw
    )

    ds_train_raw = LEMCWindowDataset(stores, train_windows, t_obs, t_pred, max_agents, norm_stats=None)
    norm_stats = fit_norm_stats(ds_train_raw)

    ds_train = LEMCWindowDataset(
        stores,
        train_windows,
        t_obs,
        t_pred,
        max_agents,
        norm_stats=norm_stats,
        orb_shifts=orb_shifts,
        orb_affines=orb_affines,
    )
    ds_val = LEMCWindowDataset(
        stores,
        val_windows,
        t_obs,
        t_pred,
        max_agents,
        norm_stats=norm_stats,
        orb_shifts=orb_shifts,
        orb_affines=orb_affines,
    )

    train_loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False)

    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=cfg["backbone"].get("predict_intent", True),
        use_speed_input=cfg.get("use_speed_input", False) or cfg["backbone"].get("use_speed_input", False),
    )
    lemc_cfg = cfg.get("lemc", {})
    device = get_device()
    model = TrajectoryPredictor(use_lemc=use_lemc, lemc_cfg=lemc_cfg, backbone_cfg=backbone_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    print(f"device: {device}")

    run_name = build_run_name(cfg, seed)
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    collapse_monitor = None
    if use_lemc:
        ref_scale = reference_displacement_scale(train_loader)
        collapse_monitor = CollapseMonitor(ref_scale, threshold_frac=0.05, patience=5)
        print(f"LEMC collapse-detection reference displacement scale: {ref_scale:.6f}")

    csv_path = os.path.join("results", f"{run_name}_epochs.csv")
    logger = CSVLogger(csv_path)

    best_val = float("inf")
    patience = cfg["train"].get("early_stop_patience", 8)
    bad_epochs = 0

    for epoch in range(cfg["train"]["epochs"]):
        t0 = time.time()
        train_loss, train_residual = run_epoch(model, train_loader, optimizer, cfg, train=True, device=device)
        val_loss, val_residual = run_epoch(model, val_loader, optimizer, cfg, train=False, device=device)
        dt = time.time() - t0

        log_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time_s": dt}
        msg = f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} ({dt:.1f}s)"

        if train_residual is not None:
            log_row.update(
                {
                    "train_residual_mean": train_residual["mean"],
                    "train_residual_std": train_residual["std"],
                    "val_residual_mean": val_residual["mean"],
                    "val_residual_std": val_residual["std"],
                }
            )
            msg += (
                f" | residual_mag train={train_residual['mean']:.5f}(+-{train_residual['std']:.5f})"
                f" val={val_residual['mean']:.5f}(+-{val_residual['std']:.5f})"
            )
        print(msg)
        logger.log(log_row)

        if collapse_monitor is not None:
            collapse_monitor.check(train_residual["mean"])

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            with open(os.path.join(ckpt_dir, "norm_stats.pkl"), "wb") as f:
                pickle.dump(norm_stats, f)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    logger.close()
    print(f"done. best val_loss={best_val:.5f}, checkpoint saved to {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
