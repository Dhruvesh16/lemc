from __future__ import annotations

import torch

from lemc.utils.device import move_batch

from .losses import (
    intent_loss,
    obd_correlation_loss,
    orb_regression_loss,
    residual_l2_penalty,
    trajectory_loss,
)


def run_epoch(model, loader, optimizer, cfg: dict, train: bool, device: torch.device | None = None) -> tuple[float, dict | None]:
    """Returns (avg_loss, residual_stats). residual_stats is None when the
    model has no LEMC module (use_lemc=False), else {'mean': ..., 'std': ...}
    over every per-transition residual magnitude seen this epoch -- the
    collapse-detection signal (plan Step 5)."""
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    residual_mags = []
    use_speed_input = cfg.get("use_speed_input", False)

    dev = device or next(model.parameters()).device

    for batch in loader:
        batch = move_batch(batch, dev)
        agent_tracks = batch["agent_tracks"]
        agent_mask = batch["agent_mask"]
        target_xy = batch["target_track"][:, :, :2]
        ego_speed = batch["ego_speed"] if use_speed_input else None

        with torch.set_grad_enabled(train):
            traj_pred, intent_logit, residual = model(agent_tracks, agent_mask, ego_speed=ego_speed)
            tl = trajectory_loss(traj_pred, target_xy, batch["target_mask"])
            il = intent_loss(intent_logit, batch["crossing_label"]) if intent_logit is not None else torch.zeros(())
            rl = residual_l2_penalty(residual)
            ol = (
                obd_correlation_loss(residual, batch["ego_speed"])
                if cfg["loss"].get("obd_weight", 0.0)
                else torch.zeros(())
            )
            orb_l = (
                orb_regression_loss(residual, batch["orb_shift"], batch["orb_mask"])
                if cfg["loss"].get("orb_weight", 0.0)
                else torch.zeros(())
            )
            loss = (
                cfg["loss"]["traj_weight"] * tl
                + cfg["loss"]["intent_weight"] * il
                + cfg["loss"].get("residual_l2_penalty", 0.0) * rl
                + cfg["loss"].get("obd_weight", 0.0) * ol
                + cfg["loss"].get("orb_weight", 0.0) * orb_l
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if residual is not None:
            residual_mags.append(residual.detach().norm(dim=-1).flatten())

    avg_loss = total_loss / max(n_batches, 1)
    residual_stats = None
    if residual_mags:
        all_mag = torch.cat(residual_mags)
        residual_stats = {"mean": all_mag.mean().item(), "std": all_mag.std().item()}
    return avg_loss, residual_stats


def reference_displacement_scale(loader) -> float:
    """Median |Δ(cx,cy)| of the ego track over a few batches, in whatever
    units the loader's agent_tracks are in (standardized cx,cy -- same units
    LEMC's residual is expressed in, since it's subtracted directly from
    them). Used as the scale-relative denominator for the collapse threshold:
    an absolute pixel constant would be meaningless once cx,cy are standardized."""
    mags = []
    for i, batch in enumerate(loader):
        ego_xy = batch["agent_tracks"][:, :, 0, :2]  # [B, T_obs, 2]
        d = ego_xy[:, 1:, :] - ego_xy[:, :-1, :]
        mags.append(d.norm(dim=-1).flatten())
        if i >= 5:  # a handful of batches is enough for a stable reference
            break
    return torch.cat(mags).median().item()


class CollapseMonitor:
    """Fires a loud warning if LEMC's mean residual magnitude stays below a
    scale-relative threshold for too many consecutive epochs -- the module
    silently switching itself off. Must run from the very first LEMC training
    run, not bolted on after the fact (plan Step 5)."""

    def __init__(self, ref_scale: float, threshold_frac: float = 0.05, patience: int = 5):
        self.threshold_frac = threshold_frac
        self.threshold = threshold_frac * ref_scale
        self.patience = patience
        self.consecutive_low = 0

    def check(self, mean_residual_mag: float) -> bool:
        """Returns True if this epoch triggered a (loud, printed) collapse warning."""
        if mean_residual_mag < self.threshold:
            self.consecutive_low += 1
        else:
            self.consecutive_low = 0

        if self.consecutive_low >= self.patience:
            print(
                f"  !!! COLLAPSE WARNING: mean residual magnitude ({mean_residual_mag:.6f}) has stayed "
                f"below {self.threshold:.6f} ({self.threshold_frac * 100:.0f}% of reference displacement "
                f"scale) for {self.consecutive_low} consecutive epochs. LEMC may have switched itself "
                f"off -- inspect before trusting any result. !!!"
            )
            return True
        return False
