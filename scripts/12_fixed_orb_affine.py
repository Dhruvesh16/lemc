"""Train / eval with fixed ORB-affine compensation (no learned LEMC residual)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lemc.data.behavior import load_ped_action_lookup, window_is_standing
from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy, fit_norm_stats, protocol_tte_kwargs
from lemc.data.paths import TRACK_STORE_DIR, track_store_path
from lemc.data.track_store import load_track_store
from lemc.eval.metrics import box_rmse_metrics, intent_metrics, trajectory_metrics
from lemc.models.backbone import GRUTrajectoryModel
from lemc.models.orb_compensator import ORBAffineCompensator
from lemc.train.losses import intent_loss, trajectory_loss
from lemc.train.seeds import seed_everything
from lemc.utils.config import load_config, run_name as build_run_name
from lemc.utils.device import get_device, move_batch
from lemc.utils.logging import CSVLogger

IMG_W, IMG_H = 1920.0, 1080.0
VEH_STOPPED = 0  # JAAD vehicle action code


class FixedORBPredictor(torch.nn.Module):
    def __init__(self, backbone_cfg: dict):
        super().__init__()
        self.compensator = ORBAffineCompensator()
        self.backbone = GRUTrajectoryModel(**backbone_cfg)

    def forward(self, agent_tracks, agent_mask, orb_shift, orb_mask, ego_speed=None):
        ego, residual = self.compensator.forward_with_orb(agent_tracks, orb_shift, orb_mask)
        traj, intent = self.backbone(ego, ego_speed=ego_speed)
        return traj, intent, residual


def run_epoch(model, loader, optimizer, cfg, train: bool, device: torch.device):
    model.train(mode=train)
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            ego_raw = batch["agent_tracks"][:, :, 0, :]
            ego_comp = torch.cat(
                [ego_raw[..., :2] - batch["camera_shift_obs"], ego_raw[..., 2:]], dim=-1
            )
            tgt_comp_xy = batch["target_track"][:, :, :2] - batch["camera_shift_pred"]
            traj, intent = model.backbone(ego_comp)
            loss = cfg["loss"]["traj_weight"] * trajectory_loss(traj, tgt_comp_xy, batch["target_mask"])
            if intent is not None:
                loss = loss + cfg["loss"]["intent_weight"] * intent_loss(intent, batch["crossing_label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def _resolve_run_name(cfg: dict, seed: int, checkpoint_dir: str | None) -> tuple[str, str]:
    if checkpoint_dir:
        return os.path.basename(checkpoint_dir.rstrip("/")), checkpoint_dir
    if cfg.get("dataset") == "pie" and cfg.get("fixed_orb_legacy_run"):
        run = f"{cfg['fixed_orb_legacy_run']}_seed{seed}"
    else:
        run = build_run_name(cfg, seed)
    return run, os.path.join("checkpoints", run)


def _vehicle_moving(stores, scene_id: str, start: int, t_obs: int, dataset: str) -> bool:
    store = stores[scene_id]
    if dataset == "pie":
        for f in range(start, start + t_obs):
            sp = store.frame_to_ego.get(f)
            if sp is not None and float(sp) > 1e-3:
                return True
        return False
    for f in range(start, start + t_obs):
        act = store.frame_to_vehicle_action.get(f)
        if act is not None and int(act) != VEH_STOPPED:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pie_lemc_on_augmented_orb_affine.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    dataset = cfg.get("dataset", "pie")
    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]
    tte_kw = protocol_tte_kwargs(cfg)
    overlap_eval = cfg["protocol"].get("overlap_eval", 0.5)
    device = get_device()
    print(f"device: {device}")

    stores = load_track_store(track_store_path(cfg))
    aff_name = cfg.get("orb_affines_file", "pie_orb_affines.pkl")
    aff_path = aff_name if os.path.isabs(aff_name) else os.path.join(TRACK_STORE_DIR, aff_name)
    with open(aff_path, "rb") as f:
        orb_affines = pickle.load(f)

    run, ckpt_dir = _resolve_run_name(cfg, args.seed, args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=True,
    )
    model = FixedORBPredictor(backbone_cfg).to(device)

    if not args.eval_only:
        train_w = build_window_index(
            stores, "train", t_obs, t_pred, max_agents, overlap=cfg["train"].get("overlap_train", 0.5), **tte_kw
        )
        val_w = build_window_index(stores, "val", t_obs, t_pred, max_agents, overlap=overlap_eval, **tte_kw)
        ns = fit_norm_stats(LEMCWindowDataset(stores, train_w, t_obs, t_pred, max_agents))
        ds_tr = LEMCWindowDataset(
            stores, train_w, t_obs, t_pred, max_agents, norm_stats=ns, orb_affines=orb_affines
        )
        ds_va = LEMCWindowDataset(
            stores, val_w, t_obs, t_pred, max_agents, norm_stats=ns, orb_affines=orb_affines
        )
        tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True)
        va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
        logger = CSVLogger(os.path.join("results", f"{run}_epochs.csv"))
        best, bad, patience = float("inf"), 0, cfg["train"].get("early_stop_patience", 20)
        for epoch in range(cfg["train"]["epochs"]):
            t0 = time.time()
            tl = run_epoch(model, tr, opt, cfg, True, device)
            vl = run_epoch(model, va, opt, cfg, False, device)
            print(f"epoch {epoch}: train={tl:.5f} val={vl:.5f} ({time.time()-t0:.1f}s)")
            logger.log({"epoch": epoch, "train_loss": tl, "val_loss": vl})
            if vl < best:
                best, bad = vl, 0
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
                with open(os.path.join(ckpt_dir, "norm_stats.pkl"), "wb") as f:
                    pickle.dump(ns, f)
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stop at {epoch}")
                    break
        logger.close()
        print(f"best val={best:.5f} -> {ckpt_dir}")

    with open(os.path.join(ckpt_dir, "norm_stats.pkl"), "rb") as f:
        ns = pickle.load(f)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device))
    model.eval()
    test_w = build_window_index(stores, "test", t_obs, t_pred, max_agents, overlap=overlap_eval, **tte_kw)
    ds = LEMCWindowDataset(stores, test_w, t_obs, t_pred, max_agents, norm_stats=ns, orb_affines=orb_affines)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    action_lookup = load_ped_action_lookup(dataset)
    dim = torch.tensor([IMG_W, IMG_H], dtype=torch.float32, device=device)

    preds_ego, tgts_ego, masks = [], [], []
    pboxes, tboxes = [], []
    iprobs, ilabels = [], []
    var_raw, var_comp = [], []
    standing_ok = standing_n = 0

    def vxy(xy):
        return float(np.var(xy[:, 0]) + np.var(xy[:, 1]))

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            ego_raw = batch["agent_tracks"][:, :, 0, :]
            ego_comp = torch.cat(
                [ego_raw[..., :2] - batch["camera_shift_obs"], ego_raw[..., 2:]], dim=-1
            )
            traj_comp, intent = model.backbone(ego_comp)
            traj_ego = traj_comp + batch["camera_shift_pred"]
            tgt_ego = batch["target_track"][:, :, :2]

            preds_ego.append(denormalize_xy(traj_ego, ns, IMG_W, IMG_H))
            tgts_ego.append(denormalize_xy(tgt_ego, ns, IMG_W, IMG_H))
            masks.append(batch["target_mask"])

            last_wh = batch["agent_tracks"][:, -1, 0, 2:] * dim
            wh_tgt = batch["target_track"][:, :, 2:] * dim
            pboxes.append(
                torch.cat(
                    [denormalize_xy(traj_ego, ns, IMG_W, IMG_H), last_wh.unsqueeze(1).expand(-1, t_pred, -1)],
                    -1,
                )
            )
            tboxes.append(torch.cat([denormalize_xy(tgt_ego, ns, IMG_W, IMG_H), wh_tgt], -1))

            if intent is not None:
                probs = torch.sigmoid(intent).squeeze(-1)
                lab = batch["crossing_label"]
                valid = ~torch.isnan(lab)
                iprobs.append(probs[valid])
                ilabels.append(lab[valid])

            raw_xy = denormalize_xy(ego_raw[..., :2], ns, IMG_W, IMG_H)
            comp_xy = denormalize_xy(ego_comp[..., :2], ns, IMG_W, IMG_H)
            for i in range(raw_xy.shape[0]):
                vr, vc = vxy(raw_xy[i].cpu().numpy()), vxy(comp_xy[i].cpu().numpy())
                var_raw.append(vr)
                var_comp.append(vc)
                sid = batch["scene_id"][i]
                pid = batch["ego_pid"][i]
                start = int(batch["start_frame"][i])
                standing = window_is_standing(action_lookup, sid, pid, start, t_obs)
                moving = _vehicle_moving(stores, sid, start, t_obs, dataset)
                if standing and moving:
                    standing_n += 1
                    standing_ok += int(vc < vr)

    mask_cat = torch.cat(masks)
    ego_m = trajectory_metrics(torch.cat(preds_ego), torch.cat(tgts_ego), mask_cat, 30, t_pred)
    ego_m.update(box_rmse_metrics(torch.cat(pboxes), torch.cat(tboxes), mask_cat, 30))
    print(f"=== Fixed ORB-affine ({run}, {dataset} test, n={len(ds)}) ===")
    for k, v in ego_m.items():
        print(f"  {k}: {v:.4f}")
    if iprobs:
        intent_m = intent_metrics(torch.cat(iprobs).cpu().numpy(), torch.cat(ilabels).cpu().numpy())
        print("=== Intent ===")
        for k, v in intent_m.items():
            print(f"  {k}: {v:.4f}")
    vr, vc = np.array(var_raw), np.array(var_comp)
    print(
        f"=== Sign gate === flatten all={(vc<vr).mean():.4f} "
        f"standing+moving={standing_ok/max(standing_n,1):.4f} (n={standing_n})"
    )

    out = os.path.join("results", f"eval_{run}.txt")
    with open(out, "w") as f:
        for k, v in ego_m.items():
            f.write(f"{k}: {v:.4f}\n")
        if iprobs:
            for k, v in intent_m.items():
                f.write(f"{k}: {v:.4f}\n")
        f.write(f"frac_flatten_all: {(vc<vr).mean():.4f}\n")
        f.write(f"frac_flatten_standing_moving: {standing_ok/max(standing_n,1):.4f}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
