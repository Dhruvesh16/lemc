"""Paper figure pack: trajectory overlays on frames, intent ROC, detection frames, metrics.

Output: results/paper_figures/{metrics,trajectory,intent,frames,orb}/
Run:  .venv/bin/python scripts/19_make_paper_figures.py
"""
from __future__ import annotations

import os
import pickle
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from lemc.data.dataset import LEMCWindowDataset, build_window_index, denormalize_xy, protocol_tte_kwargs
from lemc.data.paths import TRACK_STORE_DIR, dataset_image_root, frame_image_path, track_store_path
from lemc.data.track_store import load_track_store
from lemc.models.backbone import GRUTrajectoryModel
from lemc.models.predictor import TrajectoryPredictor
from lemc.utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "paper_figures")
IMG_W, IMG_H = 1920.0, 1080.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

DIRS = ["metrics", "trajectory", "intent", "frames", "orb"]
for d in DIRS:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

C = {
    "obs": "#2c3e50", "gt": "#27ae60", "pred_base": "#e74c3c",
    "pred_orb": "#1a7a4c", "bbox": "#3498db", "ego": "#e67e22", "speed": "#c0392b",
}


def save(fig, subdir: str, name: str):
    path = os.path.join(OUT, subdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {subdir}/{name}")


def parse_eval(path: str) -> dict[str, float]:
    out = {}
    for line in open(path):
        if ":" not in line:
            continue
        k, v = line.strip().split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out


def load_orb_affines(cfg: dict) -> dict | None:
    fn = cfg.get("orb_affines_file")
    if not fn:
        return None
    path = os.path.join(TRACK_STORE_DIR, fn)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_backbone(ckpt_dir: str, cfg: dict) -> GRUTrajectoryModel:
    t_pred = cfg["protocol"]["t_pred"]
    bb = GRUTrajectoryModel(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=cfg["backbone"].get("predict_intent", True),
        use_speed_input=cfg.get("use_speed_input", False) or cfg["backbone"].get("use_speed_input", False),
    )
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    bb_state = {k.replace("backbone.", "", 1): v for k, v in state.items() if k.startswith("backbone.")}
    bb.load_state_dict(bb_state)
    bb.eval()
    return bb


def load_baseline_predictor(ckpt_dir: str, cfg: dict) -> TrajectoryPredictor:
    t_pred = cfg["protocol"]["t_pred"]
    backbone_cfg = dict(
        input_dim=4,
        hidden_dim=cfg["backbone"]["hidden_dim"],
        num_layers=cfg["backbone"]["num_layers"],
        t_pred=t_pred,
        predict_intent=cfg["backbone"].get("predict_intent", True),
        use_speed_input=cfg.get("use_speed_input", False),
    )
    model = TrajectoryPredictor(use_lemc=False, lemc_cfg={}, backbone_cfg=backbone_cfg)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def collect_preds(model, ds, fixed_orb: bool, norm_stats):
    """Return per-window trajectory preds + intent for baseline or fixed-ORB backbone."""
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds, tgts, masks = [], [], []
    iprobs, ilabels = [], []
    meta = []

    for batch in loader:
        if fixed_orb:
            ego_raw = batch["agent_tracks"][:, :, 0, :]
            ego_comp = torch.cat(
                [ego_raw[..., :2] - batch["camera_shift_obs"], ego_raw[..., 2:]], dim=-1
            )
            traj_comp, intent = model(ego_comp)
            traj_ego = traj_comp + batch["camera_shift_pred"]
        else:
            traj_ego, intent, _ = model(batch["agent_tracks"], batch["agent_mask"])

        pred_px = denormalize_xy(traj_ego, norm_stats, IMG_W, IMG_H)
        tgt_px = denormalize_xy(batch["target_track"][:, :, :2], norm_stats, IMG_W, IMG_H)
        preds.append(pred_px)
        tgts.append(tgt_px)
        masks.append(batch["target_mask"])

        if intent is not None:
            probs = torch.sigmoid(intent).squeeze(-1)
            lab = batch["crossing_label"]
            valid = ~torch.isnan(lab)
            iprobs.append(probs[valid])
            ilabels.append(lab[valid])

        for i in range(len(batch["scene_id"])):
            meta.append({
                "scene_id": batch["scene_id"][i],
                "ego_pid": batch["ego_pid"][i],
                "start": int(batch["start_frame"][i]),
            })

    return {
        "pred": torch.cat(preds),
        "tgt": torch.cat(tgts),
        "mask": torch.cat(masks),
        "iprobs": torch.cat(iprobs).numpy() if iprobs else None,
        "ilabels": torch.cat(ilabels).numpy() if ilabels else None,
        "meta": meta,
    }


def per_window_ade(pred, tgt, mask):
    diff = (pred - tgt).norm(dim=-1)
    return (diff * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)


def draw_trajectory_on_frame(
    img_bgr,
    obs_xy: np.ndarray,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    title: str,
    out_path: str,
):
    """obs_xy: [T_obs,2], gt/pred: [T_pred,2] in pixel coords."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.plot(obs_xy[:, 0], obs_xy[:, 1], "o-", color=C["obs"], lw=2, ms=4, label="observed")
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "s-", color=C["gt"], lw=2, ms=3, label="GT future")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "^-", color=C["pred_orb"], lw=2, ms=3, label="predicted")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_compare_panel(
    img_bgr,
    obs_xy, gt_xy, pred_base, pred_orb,
    title: str, out_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pred, label, color in [
        (axes[0], pred_base, "Baseline GRU", C["pred_base"]),
        (axes[1], pred_orb, "Fixed ORB affine", C["pred_orb"]),
    ]:
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.plot(obs_xy[:, 0], obs_xy[:, 1], "o-", color=C["obs"], lw=1.5, ms=3, label="observed")
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "s-", color=C["gt"], lw=1.5, ms=3, label="GT")
        ax.plot(pred[:, 0], pred[:, 1], "^-", color=color, lw=1.5, ms=3, label="pred")
        ax.set_title(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_detection_frame(img_bgr, store, frame: int, ego_pid: str, agent_pids: list, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    boxes = store.frame_to_boxes.get(frame, {})
    for pid, box in boxes.items():
        if pid not in agent_pids:
            continue
        cx, cy, w, h = box
        x1, y1 = cx - w / 2, cy - h / 2
        color = C["ego"] if pid == ego_pid else C["bbox"]
        lw = 2.5 if pid == ego_pid else 1.2
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, pid if pid == ego_pid else "", color=color, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 1. Intent bar charts from eval files ─────────────────────────────────────

def fig_intent_bars():
    specs = [
        ("PIE", [
            ("pie_nolemc_augmented_base", "Baseline"),
            ("pie_nolemc_augmented_speed", "GT speed"),
            ("pie_fixed_orb_affine_v2", "Fixed ORB"),
        ]),
        ("JAAD", [
            ("jaad_nolemc_jaad_base", "Baseline"),
            ("jaad_nolemc_fixed_orb_v2", "Fixed ORB"),
        ]),
    ]
    for dataset, methods in specs:
        aucs, f1s, labels = [], [], []
        for tag, label in methods:
            path = os.path.join(ROOT, "results", f"eval_{tag}_seed0.txt")
            if not os.path.exists(path):
                continue
            m = parse_eval(path)
            aucs.append(m.get("auc", np.nan))
            f1s.append(m.get("f1", np.nan))
            labels.append(label)
        if not labels:
            continue
        colors = [C["obs"], C["speed"], C["pred_orb"]][: len(labels)]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, vals, ylab in [(axes[0], aucs, "AUC"), (axes[1], f1s, "F1")]:
            ax.bar(x, vals, color=colors, width=0.55)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylab)
            ax.set_ylim(0, 1.05)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        fig.suptitle(f"{dataset} intent (seed 0)", fontsize=11)
        fig.tight_layout()
        save(fig, "intent", f"{dataset.lower()}_intent_auc_f1.png")


# ── 2. ROC curves (run inference seed 0) ─────────────────────────────────────

def fig_roc_curves():
    jobs = [
        ("pie", "configs/pie_nolemc_augmented.yaml", "checkpoints/pie_nolemc_augmented_base_seed0", False, "Baseline"),
        ("pie", "configs/pie_fixed_orb_affine.yaml", "checkpoints/pie_fixed_orb_affine_v2_seed0", True, "Fixed ORB"),
        ("jaad", "configs/jaad_nolemc.yaml", "checkpoints/jaad_nolemc_jaad_base_seed0", False, "Baseline"),
        ("jaad", "configs/jaad_fixed_orb_affine.yaml", "checkpoints/jaad_nolemc_fixed_orb_v2_seed0", True, "Fixed ORB"),
    ]
    by_ds: dict[str, list] = {}
    for dataset, cfg_path, ckpt, fixed_orb, label in jobs:
        cfg = load_config(cfg_path)
        stores = load_track_store(track_store_path(cfg))
        t_obs = cfg["protocol"]["t_obs"]
        t_pred = cfg["protocol"]["t_pred"]
        max_a = cfg["max_agents"]
        overlap = cfg["protocol"].get("overlap_eval", 0.5)
        windows = build_window_index(stores, "test", t_obs, t_pred, max_a, overlap=overlap, **protocol_tte_kwargs(cfg))
        with open(os.path.join(ckpt, "norm_stats.pkl"), "rb") as f:
            ns = pickle.load(f)
        orb = load_orb_affines(cfg)
        ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_a, norm_stats=ns, orb_affines=orb)
        if fixed_orb:
            model = load_backbone(ckpt, cfg)
        else:
            model = load_baseline_predictor(ckpt, cfg)
        out = collect_preds(model, ds, fixed_orb, ns)
        if out["iprobs"] is None:
            continue
        by_ds.setdefault(dataset, []).append((label, out["iprobs"], out["ilabels"]))

    for dataset, curves in by_ds.items():
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for label, probs, labels in curves:
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{dataset.upper()} intent ROC (test)")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        save(fig, "intent", f"{dataset}_roc_curves.png")


# ── 3. Trajectory overlays on real frames ────────────────────────────────────

def fig_trajectory_frames():
    jobs = [
        ("pie", "configs/pie_nolemc_augmented.yaml", "checkpoints/pie_nolemc_augmented_base_seed0",
         "configs/pie_fixed_orb_affine.yaml", "checkpoints/pie_fixed_orb_affine_v2_seed0"),
        ("jaad", "configs/jaad_nolemc.yaml", "checkpoints/jaad_nolemc_jaad_base_seed0",
         "configs/jaad_fixed_orb_affine.yaml", "checkpoints/jaad_nolemc_fixed_orb_v2_seed0"),
    ]
    for dataset, cfg_b, ckpt_b, cfg_o, ckpt_o in jobs:
        cfg = load_config(cfg_o)
        stores = load_track_store(track_store_path(cfg))
        t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
        max_a = cfg["max_agents"]
        overlap = cfg["protocol"].get("overlap_eval", 0.5)
        windows = build_window_index(stores, "test", t_obs, t_pred, max_a, overlap=overlap, **protocol_tte_kwargs(cfg))
        orb = load_orb_affines(cfg)
        image_root = dataset_image_root(cfg)

        with open(os.path.join(ckpt_b, "norm_stats.pkl"), "rb") as f:
            ns_b = pickle.load(f)
        with open(os.path.join(ckpt_o, "norm_stats.pkl"), "rb") as f:
            ns_o = pickle.load(f)

        ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_a, norm_stats=ns_o, orb_affines=orb)
        model_b = load_baseline_predictor(ckpt_b, load_config(cfg_b))
        model_o = load_backbone(ckpt_o, cfg)
        out_b = collect_preds(model_b, ds, False, ns_b)
        out_o = collect_preds(model_o, ds, True, ns_o)

        ade_b = per_window_ade(out_b["pred"], out_b["tgt"], out_b["mask"])
        ade_o = per_window_ade(out_o["pred"], out_o["tgt"], out_o["mask"])
        gain = ade_b - ade_o  # positive = ORB better

        # pick top-2 windows where ORB helps most and image exists
        order = torch.argsort(gain, descending=True)
        picked = 0
        for wi in order.tolist():
            if picked >= 2:
                break
            meta = out_o["meta"][wi]
            sid, pid, start = meta["scene_id"], meta["ego_pid"], meta["start"]
            frame = start + t_obs - 1
            img_path = frame_image_path(image_root, sid, frame, dataset)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            item = ds[wi]
            obs_xy = denormalize_xy(item["agent_tracks"][:, 0, :2].unsqueeze(0), ns_o, IMG_W, IMG_H)[0].numpy()
            gt_xy = out_o["tgt"][wi].numpy()
            pb = out_b["pred"][wi].numpy()
            po = out_o["pred"][wi].numpy()
            ab, ao = ade_b[wi].item(), ade_o[wi].item()

            title = f"{sid} · {pid} · ADE base={ab:.0f}px orb={ao:.0f}px"
            out_path = os.path.join(OUT, "trajectory", f"{dataset}_compare_{picked}.png")
            draw_compare_panel(img, obs_xy, gt_xy, pb, po, title, out_path)
            print(f"  trajectory/{dataset}_compare_{picked}.png")
            picked += 1


# ── 4. Multi-agent detection frames ──────────────────────────────────────────

def fig_detection_frames():
    for dataset, cfg_path in [
        ("pie", "configs/pie_lemc_on_augmented_orb_affine.yaml"),
        ("jaad", "configs/jaad_fixed_orb_affine.yaml"),
    ]:
        cfg = load_config(cfg_path)
        stores = load_track_store(track_store_path(cfg))
        image_root = dataset_image_root(cfg)
        t_obs = cfg["protocol"]["t_obs"]
        max_a = cfg["max_agents"]
        windows = build_window_index(
            stores, "test", t_obs, cfg["protocol"]["t_pred"], max_a,
            overlap=0.5, **protocol_tte_kwargs(cfg),
        )
        # pick window with most agents visible
        best = None
        best_n = 0
        for scene_id, ego_pid, start, agent_pids in windows[:500]:
            n = sum(
                1 for f in range(start, start + t_obs)
                for pid in agent_pids if pid in stores[scene_id].frame_to_boxes.get(f, {})
            )
            if n > best_n:
                frame = start + t_obs // 2
                path = frame_image_path(image_root, scene_id, frame, dataset)
                if os.path.exists(path):
                    best_n = n
                    best = (scene_id, ego_pid, start, agent_pids, frame, path)
        if best is None:
            print(f"  skip detection frame for {dataset} (no images)")
            continue
        sid, ego_pid, start, agent_pids, frame, path = best
        img = cv2.imread(path)
        title = f"{dataset.upper()} multi-agent scene · {sid} · {len(agent_pids)} agents"
        out_path = os.path.join(OUT, "frames", f"{dataset}_multi_agent_detection.png")
        draw_detection_frame(img, stores[sid], frame, ego_pid, agent_pids, title, out_path)
        print(f"  frames/{dataset}_multi_agent_detection.png")


# ── 5. Metrics + orb copies ───────────────────────────────────────────────────

def copy_and_metrics():
    src_all = os.path.join(ROOT, "results", "all_figures")
    src_figs = os.path.join(ROOT, "results", "figures")
    copies = {
        "metrics": [
            "pie_grid_ade_fde.png", "jaad_grid_metrics.png", "cross_dataset_ade.png",
            "pie_stratified_ade.png", "paper_fig_main_grid_ade.png", "paper_fig5_stratified_ade.png",
        ],
        "orb": [
            "pie_sign_gate.png", "jaad_sign_gate.png", "paper_fig1_teaser.png",
            "paper_fig4_orb_obd_scatter.png", "paper_fig2_architecture.png", "paper_fig_sign_gate.png",
        ],
        "trajectory": [
            "paper_fig3_qualitative.png", "jaad_training_curves.png",
            "lemc_qualitative_pie_lemc_augmented_orb_affine_seed0_test.png",
            "lemc_qualitative_pie_lemc_augmented_auxobd_seed0_test.png",
        ],
        "intent": [
            "lemc_obd_scatter_pie_lemc_augmented_auxobd_seed0_test.png",
        ],
        "frames": [
            "density_pie_track_store_augmented_train.png",
            "density_jaad_track_store_train.png",
        ],
    }
    for subdir, names in copies.items():
        for name in names:
            for src_root in (src_all, src_figs, os.path.join(ROOT, "results")):
                src = os.path.join(src_root, name.replace("paper_", ""))
                alt = os.path.join(src_root, name)
                for p in (alt, src):
                    if os.path.exists(p):
                        shutil.copy2(p, os.path.join(OUT, subdir, name if "paper_" in name or p == alt else os.path.basename(p)))
                        break

    for f in os.listdir(src_figs):
        if f.endswith(".png"):
            sub = "orb" if any(k in f for k in ("orb", "sign", "teaser", "architecture")) else "trajectory"
            shutil.copy2(os.path.join(src_figs, f), os.path.join(OUT, sub, f"paper_{f}"))


def build_gallery_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    pdf_path = os.path.join(OUT, "paper_figures_gallery.pdf")
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("LEMC Paper Figures Gallery", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    sections = [
        ("metrics", "Metrics"),
        ("trajectory", "Trajectory (predicted vs GT on frames)"),
        ("intent", "Intent (AUC / F1 / ROC)"),
        ("frames", "Frames (multi-agent detection + density)"),
        ("orb", "ORB validation + sign gate"),
    ]
    for subdir, title in sections:
        folder = os.path.join(OUT, subdir)
        pngs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
        if not pngs:
            continue
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        for name in pngs:
            path = os.path.join(folder, name)
            img = Image(path)
            ratio = img.imageHeight / float(img.imageWidth)
            w = 6.5 * inch
            img.drawWidth = w
            img.drawHeight = w * ratio
            story.append(Paragraph(name, styles["Normal"]))
            story.append(img)
            story.append(Spacer(1, 0.15 * inch))
        story.append(PageBreak())

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    doc.build(story)
    print(f"  wrote paper_figures_gallery.pdf")


def write_index():
    lines = ["# Paper figures index\n", f"Generated into `{OUT}/`\n"]
    for sub in DIRS:
        lines.append(f"\n## {sub}/\n")
        folder = os.path.join(OUT, sub)
        for f in sorted(os.listdir(folder)):
            if f.endswith(".png"):
                lines.append(f"- `{f}`\n")
    path = os.path.join(OUT, "INDEX.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote INDEX.md")


if __name__ == "__main__":
    print(f"Paper figures → {OUT}/")
    print("Intent bars...")
    fig_intent_bars()
    print("ROC curves...")
    fig_roc_curves()
    print("Trajectory frame overlays...")
    fig_trajectory_frames()
    print("Detection frames...")
    fig_detection_frames()
    print("Copy metrics/orb...")
    copy_and_metrics()
    build_gallery_pdf()
    write_index()
    n = sum(len([f for f in os.listdir(os.path.join(OUT, d)) if f.endswith(".png")]) for d in DIRS)
    print(f"\nDone — {n} PNGs in results/paper_figures/")
