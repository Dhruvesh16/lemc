"""Generate all result figures — existing paper figs + new JAAD figures — into results/all_figures/."""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "all_figures")
FIGS = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

C = {
    "baseline": "#2c3e50",
    "fixed_orb": "#1a7a4c",
    "speed": "#c0392b",
    "lemc": "#8e44ad",
    "raw": "#7f8c8d",
    "jaad_base": "#2980b9",
    "jaad_orb": "#16a085",
}


# ── helper ──────────────────────────────────────────────────────────────────

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


def load_epoch_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── 1. PIE main grid ────────────────────────────────────────────────────────

def fig_pie_grid():
    methods = ["Baseline", "GT speed", "LEMC+OBD", "LEMC+ORB\n(learned)", "LEMC\nno aux", "Fixed ORB\naffine"]
    ade  = [60.81, 66.96, 85.77, 81.73, 81.68, 60.49]
    err  = [3.31,  1.28, 11.19,  4.61,  5.13,  1.83]
    fde  = [121.75, 136.89, 164.25, 163.54, 159.91, 130.28]
    ferr = [9.87,   3.02,  11.33,   7.55,   2.39,   1.31]
    colors = [C["baseline"], C["speed"], C["lemc"], C["lemc"], C["lemc"], C["fixed_orb"]]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, vals, errs, ylabel, title in [
        (axes[0], ade,  err,  "ADE (px)", "ADE — PIE test (mean±std, 3 seeds)"),
        (axes[1], fde, ferr,  "FDE (px)", "FDE — PIE test (mean±std, 3 seeds)"),
    ]:
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3, ecolor="#333", width=0.6)
        ax.axhline(vals[0], color=C["baseline"], ls="--", lw=1, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    save(fig, "pie_grid_ade_fde.png")


# ── 2. PIE sign gate ────────────────────────────────────────────────────────

def fig_pie_sign_gate():
    labels = ["Attempt-3\nOBD aux", "Fixed ORB\naffine"]
    vals = [0.75, 93.12]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(labels, vals, color=[C["lemc"], C["fixed_orb"]], width=0.55)
    ax.axhline(50, color="#999", ls=":", lw=1)
    ax.set_ylabel("% windows with var_comp < var_raw")
    ax.set_title("Sign gate — PIE standing + moving")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha="center", fontsize=11)
    save(fig, "pie_sign_gate.png")


# ── 3. JAAD grid — ADE / FDE bar chart ──────────────────────────────────────

def fig_jaad_grid():
    methods = ["Baseline\nGRU", "Fixed ORB\naffine"]
    ade  = [83.10, 62.68]; aerr = [2.00, 2.34]
    fde  = [156.25, 120.11]; ferr = [4.31, 2.82]
    arb  = [44.76, 34.80]; rberr = [2.15, 2.25]
    frb  = [71.23, 54.17]; fberr = [2.08, 0.91]
    colors = [C["jaad_base"], C["jaad_orb"]]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    specs = [
        (ade, aerr, "ADE (px)", "ADE"),
        (fde, ferr, "FDE (px)", "FDE"),
        (arb, rberr, "ARB @ 30f (px)", "ARB"),
        (frb, fberr, "FRB @ 30f (px)", "FRB"),
    ]
    for ax, (vals, errs, ylabel, title) in zip(axes, specs):
        bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=4, ecolor="#333", width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel); ax.set_title(f"JAAD {title} (3 seeds)")
        ax.set_ylim(0, max(vals) * 1.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02,
                    f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, "jaad_grid_metrics.png")


# ── 4. JAAD sign gate ────────────────────────────────────────────────────────

def fig_jaad_sign_gate():
    labels = ["Fixed ORB\nall windows", "Fixed ORB\nstanding+moving"]
    vals = [71.86, 96.94]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(labels, vals, color=[C["jaad_orb"], C["jaad_orb"]], width=0.5)
    ax.axhline(50, color="#999", ls=":", lw=1)
    ax.set_ylabel("% windows with var_comp < var_raw")
    ax.set_title("Sign gate — JAAD fixed ORB")
    ax.set_ylim(0, 110)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    save(fig, "jaad_sign_gate.png")


# ── 5. Cross-dataset ADE comparison ──────────────────────────────────────────

def fig_cross_dataset():
    datasets = ["PIE\n(n=1773)", "JAAD\n(n=295)"]
    base_ade = [60.81, 83.10]; base_err = [3.31, 2.00]
    orb_ade  = [60.49, 62.68]; orb_err  = [1.83, 2.34]
    x = np.arange(len(datasets)); w = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - w/2, base_ade, w, yerr=base_err, label="Baseline GRU",
                color=C["baseline"], capsize=4, ecolor="#333")
    b2 = ax.bar(x + w/2, orb_ade,  w, yerr=orb_err,  label="Fixed ORB affine",
                color=C["fixed_orb"], capsize=4, ecolor="#333")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("ADE (px)")
    ax.set_title("Cross-dataset: Baseline vs Fixed ORB (3 seeds)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "cross_dataset_ade.png")


# ── 6. JAAD training curves (loss over epochs) ───────────────────────────────

def fig_jaad_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tag, label, color) in zip(
        axes,
        [("jaad_nolemc_jaad_base", "Baseline GRU", C["jaad_base"]),
         ("jaad_nolemc_fixed_orb_v2", "Fixed ORB affine", C["jaad_orb"])],
    ):
        for seed in range(3):
            csv = os.path.join(RES, f"{tag}_seed{seed}_epochs.csv")
            df = load_epoch_csv(csv)
            if df is None:
                continue
            ax.plot(df["epoch"], df["train_loss"], color=color, alpha=0.4, lw=1.2,
                    label=f"train s{seed}" if seed == 0 else "_")
            ax.plot(df["epoch"], df["val_loss"], color=color, alpha=0.9, lw=1.5, ls="--",
                    label=f"val s{seed}" if seed == 0 else "_")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"JAAD training — {label}")
        ax.legend(["train (seeds)", "val (seeds)"], frameon=False)
    fig.tight_layout()
    save(fig, "jaad_training_curves.png")


# ── 7. PIE stratified ADE (already in figures/, regenerate here too) ─────────

def fig_pie_stratified():
    strata = ["Accelerating\n(n=251)", "Constant\n(n=1364)", "Decelerating\n(n=158)"]
    baseline = [58.5, 64.5, 65.0]
    speed    = [88.6, 62.2, 82.0]
    fixed    = [52.2, 59.4, 67.5]
    x = np.arange(len(strata)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - w, baseline, w, label="Baseline", color=C["baseline"])
    ax.bar(x,     speed,    w, label="GT OBD speed", color=C["speed"])
    ax.bar(x + w, fixed,    w, label="Fixed ORB affine", color=C["fixed_orb"])
    ax.set_xticks(x); ax.set_xticklabels(strata)
    ax.set_ylabel("ADE (px)"); ax.set_title("PIE — ADE by ego-vehicle state")
    ax.legend(frameon=False); ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "pie_stratified_ade.png")


# ── 8. Copy all existing figures ─────────────────────────────────────────────

def copy_existing():
    copied = 0
    # paper figures
    for f in os.listdir(FIGS):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(FIGS, f), os.path.join(OUT, f"paper_{f}"))
            copied += 1
    # other result PNGs (density, qualitative, scatter)
    for f in os.listdir(RES):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(RES, f), os.path.join(OUT, f))
            copied += 1
    print(f"  copied {copied} existing images")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating figures → {OUT}/")
    fig_pie_grid()
    fig_pie_sign_gate()
    fig_jaad_grid()
    fig_jaad_sign_gate()
    fig_cross_dataset()
    fig_jaad_training_curves()
    fig_pie_stratified()
    copy_existing()
    all_pngs = [f for f in os.listdir(OUT) if f.endswith(".png")]
    print(f"\nDone — {len(all_pngs)} PNGs in {OUT}/")
    for f in sorted(all_pngs):
        sz = os.path.getsize(os.path.join(OUT, f)) // 1024
        print(f"  {f:55s} {sz:4d} KB")
