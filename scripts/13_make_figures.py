"""Generate paper figures (Steps 13) into results/figures/."""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lemc.data.dataset import LEMCWindowDataset, build_window_index, protocol_tte_kwargs
from lemc.data.paths import PIE_DATA_ROOT, TRACK_STORE_DIR, track_store_path
from lemc.data.track_store import load_track_store
from lemc.utils.config import load_config

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "baseline": "#2c3e50",
    "speed": "#c0392b",
    "fixed_orb": "#1a7a4c",
    "raw": "#7f8c8d",
    "comp": "#1a7a4c",
}


def fig5_stratified():
    strata = ["Accelerating\n(n=251)", "Constant\n(n=1364)", "Decelerating\n(n=158)"]
    baseline = [58.5, 64.5, 65.0]
    speed = [88.6, 62.2, 82.0]
    fixed = [52.2, 59.4, 67.5]

    x = np.arange(len(strata))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w, baseline, w, label="Baseline", color=COLORS["baseline"])
    ax.bar(x, speed, w, label="GT OBD speed", color=COLORS["speed"])
    ax.bar(x + w, fixed, w, label="Fixed ORB affine", color=COLORS["fixed_orb"])
    ax.set_xticks(x)
    ax.set_xticklabels(strata)
    ax.set_ylabel("ADE (px)")
    ax.set_title("Trajectory error by ego-vehicle state")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    fig.savefig(os.path.join(OUT, "fig5_stratified_ade.png"))
    fig.savefig(os.path.join(OUT, "fig5_stratified_ade.pdf"))
    plt.close(fig)
    print("wrote fig5_stratified_ade")


def fig_main_grid():
    methods = [
        "Baseline",
        "GT speed",
        "LEMC+OBD",
        "LEMC+ORB\n(learned)",
        "LEMC\nno aux",
        "Fixed ORB\naffine",
    ]
    ade = [60.81, 66.96, 85.77, 81.73, 81.68, 60.49]
    err = [3.31, 1.28, 11.19, 4.61, 5.13, 1.83]
    colors = [
        COLORS["baseline"],
        COLORS["speed"],
        "#8e44ad",
        "#8e44ad",
        "#8e44ad",
        COLORS["fixed_orb"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(methods))
    ax.bar(x, ade, yerr=err, color=colors, capsize=3, ecolor="#333")
    ax.axhline(60.81, color=COLORS["baseline"], ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("ADE (px)")
    ax.set_title("Main grid — PIE test (mean ± std, 3 seeds)")
    fig.savefig(os.path.join(OUT, "fig_main_grid_ade.png"))
    fig.savefig(os.path.join(OUT, "fig_main_grid_ade.pdf"))
    plt.close(fig)
    print("wrote fig_main_grid_ade")


def fig_sign_gate():
    labels = ["Attempt-3\nOBD aux", "Fixed ORB\naffine"]
    vals = [0.75, 93.12]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(labels, vals, color=["#c0392b", COLORS["fixed_orb"]], width=0.55)
    ax.axhline(50, color="#999", ls=":", lw=1)
    ax.set_ylabel("% windows with var_comp < var_raw")
    ax.set_title("Sign gate — standing + moving")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_sign_gate.png"))
    fig.savefig(os.path.join(OUT, "fig_sign_gate.pdf"))
    plt.close(fig)
    print("wrote fig_sign_gate")


def _pick_teaser_window(stores, windows, action, aff, t_obs):
    best = None
    best_score = -1.0
    for scene_id, ego_pid, start, _ in windows:
        fa = action.get((scene_id, ego_pid), {})
        if not all(fa.get(f, -1) == 0 for f in range(start, start + t_obs)):
            continue
        store = stores[scene_id]
        speeds = [store.frame_to_ego.get(f) for f in range(start, start + t_obs)]
        speeds = [float(s) for s in speeds if s is not None]
        if not speeds or np.mean(speeds) < 5.0:
            continue
        xy = []
        for f in range(start, start + t_obs):
            if ego_pid not in store.frame_to_boxes.get(f, {}):
                break
            xy.append(store.frame_to_boxes[f][ego_pid][:2])
        if len(xy) < t_obs:
            continue
        xy = np.array(xy, dtype=np.float64)
        var = float(np.var(xy[:, 0]) + np.var(xy[:, 1]))
        if not all(f in aff.get(scene_id, {}) for f in range(start, start + t_obs - 1)):
            continue
        score = var * np.mean(speeds)
        if score > best_score:
            best_score = score
            best = (scene_id, ego_pid, start, xy, float(np.mean(speeds)))
    return best


def _compensate_xy(scene_id, start, xy, aff):
    comp = xy.copy()
    cum = np.zeros(2)
    for i in range(len(xy) - 1):
        M = aff[scene_id].get(start + i)
        if M is not None:
            pt = np.array([xy[i, 0], xy[i, 1], 1.0])
            mapped = M @ pt
            cum = cum + (mapped[:2] - xy[i])
        comp[i + 1] = xy[i + 1] - cum
    return comp


def fig1_teaser_and_fig3_qual():
    cfg = load_config("configs/pie_lemc_on_augmented_orb_affine.yaml")
    stores = load_track_store(track_store_path(cfg))
    t_obs, t_pred, max_a = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"], cfg["max_agents"]
    windows = build_window_index(
        stores, "test", t_obs, t_pred, max_a, overlap=0.5, **protocol_tte_kwargs(cfg)
    )
    with open(os.path.join(TRACK_STORE_DIR, "pie_orb_affines.pkl"), "rb") as f:
        aff = pickle.load(f)
    with open(os.path.join(PIE_DATA_ROOT, "data_cache", "pie_database.pkl"), "rb") as f:
        db = pickle.load(f)
    action = {}
    for sid, vids in db.items():
        for vid, scene in vids.items():
            for pid, rec in scene["ped_annotations"].items():
                if "behavior" in rec:
                    action[(f"{sid}/{vid}", pid)] = {
                        int(fr): int(a) for fr, a in zip(rec["frames"], rec["behavior"]["action"])
                    }

    pick = _pick_teaser_window(stores, windows, action, aff, t_obs)
    if pick is None:
        print("teaser: no window found")
        return
    scene_id, ego_pid, start, raw, speed = pick
    comp = _compensate_xy(scene_id, start, raw, aff)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(raw[:, 0], raw[:, 1], "o-", color=COLORS["raw"], label="Raw ego-view track", ms=5)
    ax.plot(comp[:, 0], comp[:, 1], "s-", color=COLORS["comp"], label="ORB-affine compensated", ms=5)
    ax.scatter(raw[0, 0], raw[0, 1], c="black", s=60, zorder=5, label="t = 0")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("image x (px)")
    ax.set_ylabel("image y (px)")
    ax.set_title(f"Standing pedestrian · mean OBD {speed:.1f} km/h\n{scene_id} · {ego_pid}")
    ax.legend(frameon=False, loc="best")
    fig.savefig(os.path.join(OUT, "fig1_teaser.png"))
    fig.savefig(os.path.join(OUT, "fig1_teaser.pdf"))
    plt.close(fig)
    print(f"wrote fig1_teaser ({scene_id})")

    ds = LEMCWindowDataset(
        stores, windows, t_obs, t_pred, max_a, norm_stats=None, orb_affines=aff
    )
    mean_speeds = []
    for i in range(len(ds)):
        s = ds[i]["ego_speed"].numpy()
        mean_speeds.append(np.nanmean(s) if np.isfinite(s).any() else np.nan)
    mean_speeds = np.array(mean_speeds)
    valid = np.isfinite(mean_speeds)
    idxs = np.where(valid)[0]
    speeds_v = mean_speeds[valid]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, q in zip(axes, (10, 40, 60, 90)):
        target = np.percentile(speeds_v, q)
        pick_i = idxs[np.argmin(np.abs(speeds_v - target))]
        scene_id, ego_pid, start, _ = windows[int(pick_i)]
        store = stores[scene_id]
        raw = np.array(
            [store.frame_to_boxes[f][ego_pid][:2] for f in range(start, start + t_obs)],
            dtype=np.float64,
        )
        comp = _compensate_xy(scene_id, start, raw, aff)
        ax.plot(raw[:, 0], raw[:, 1], "o-", color=COLORS["raw"], label="raw", ms=3, alpha=0.85)
        ax.plot(
            comp[:, 0], comp[:, 1], "s-", color=COLORS["comp"], label="compensated", ms=3, alpha=0.85
        )
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"speed≈{mean_speeds[pick_i]:.1f}")
        if q == 10:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Raw vs ORB-affine compensated ego tracks", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_qualitative.png"))
    fig.savefig(os.path.join(OUT, "fig3_qualitative.pdf"))
    plt.close(fig)
    print("wrote fig3_qualitative")


def fig4_orb_obd_scatter():
    cfg = load_config("configs/pie_lemc_on_augmented_orb_affine.yaml")
    stores = load_track_store(track_store_path(cfg))
    with open(os.path.join(TRACK_STORE_DIR, "pie_orb_affines.pkl"), "rb") as f:
        aff = pickle.load(f)
    mags, speeds = [], []
    for sid, store in stores.items():
        for f, M in aff.get(sid, {}).items():
            sp = store.frame_to_ego.get(f)
            if sp is None:
                continue
            mags.append(float(np.hypot(M[0, 2], M[1, 2])))
            speeds.append(float(sp))
    mags, speeds = np.array(mags), np.array(speeds)
    rng = np.random.default_rng(0)
    if len(mags) > 8000:
        sel = rng.choice(len(mags), 8000, replace=False)
        mags_p, speeds_p = mags[sel], speeds[sel]
    else:
        mags_p, speeds_p = mags, speeds
    r = np.corrcoef(mags, speeds)[0, 1]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(speeds_p, mags_p, s=4, alpha=0.25, c=COLORS["fixed_orb"], rasterized=True)
    coeffs = np.polyfit(speeds, mags, 1)
    xs = np.linspace(speeds.min(), speeds.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="#222", lw=2)
    ax.set_xlabel("OBD speed (km/h)")
    ax.set_ylabel("|ORB translation| (px)")
    ax.set_title(f"ORB affine |t| vs OBD (R²={r*r:.3f}, n={len(mags)})")
    fig.savefig(os.path.join(OUT, "fig4_orb_obd_scatter.png"))
    fig.savefig(os.path.join(OUT, "fig4_orb_obd_scatter.pdf"))
    plt.close(fig)
    print("wrote fig4_orb_obd_scatter")


def fig2_architecture():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, edgecolor="#222", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.2, 1.0, 1.8, 1.2, "Multi-agent\nbboxes", "#ecf0f1")
    box(2.3, 1.0, 2.0, 1.2, "ORB affine\n(offline)", "#d5f5e3")
    box(4.6, 1.0, 2.0, 1.2, "Compensate\nego track", "#abebc6")
    box(6.9, 1.0, 1.6, 1.2, "GRU\nbackbone", "#d6eaf8")
    box(8.7, 1.55, 1.1, 0.7, "traj", "#fadbd8")
    box(8.7, 0.7, 1.1, 0.7, "intent", "#fadbd8")
    for x0, x1 in ((2.0, 2.3), (4.3, 4.6), (6.6, 6.9), (8.5, 8.7)):
        ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(
        5,
        2.6,
        "Fixed ORB-affine compensation (sensor-free at inference)",
        ha="center",
        fontsize=12,
    )
    fig.savefig(os.path.join(OUT, "fig2_architecture.png"))
    fig.savefig(os.path.join(OUT, "fig2_architecture.pdf"))
    plt.close(fig)
    print("wrote fig2_architecture")


if __name__ == "__main__":
    fig5_stratified()
    fig_main_grid()
    fig_sign_gate()
    fig4_orb_obd_scatter()
    fig2_architecture()
    fig1_teaser_and_fig3_qual()
    print(f"all figures in {OUT}/")
