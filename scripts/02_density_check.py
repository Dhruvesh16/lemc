import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lemc.data.density_check import density_verdict, per_frame_agent_counts, summarize
from lemc.data.paths import TRACK_STORE_DIR
from lemc.data.track_store import load_track_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pie", "jaad"], required=True)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(TRACK_STORE_DIR), "results"))
    parser.add_argument(
        "--track-store-file",
        default=None,
        help="override the track store filename (e.g. pie_track_store_augmented.pkl)",
    )
    args = parser.parse_args()

    store_filename = args.track_store_file or f"{args.dataset}_track_store.pkl"
    store_path = os.path.join(TRACK_STORE_DIR, store_filename)
    stores = load_track_store(store_path)

    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = os.path.splitext(store_filename)[0]  # e.g. pie_track_store or pie_track_store_augmented
    print(f"=== Density check: {run_tag} ===")
    for split in ["train", "val", "test", None]:
        counts = per_frame_agent_counts(stores, split=split)
        if counts.size == 0:
            continue
        stats = summarize(counts)
        label = split if split else "all splits combined"
        print(f"\n-- {label} --")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if split == "train" or (split is None):
            verdict_split = "train" if split == "train" else "all"
            print(f"  VERDICT ({verdict_split}): {density_verdict(stats['median'])}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(counts, bins=range(0, int(counts.max()) + 2), align="left")
        ax.set_xlabel("agents per frame")
        ax.set_ylabel("frame count")
        ax.set_title(f"{run_tag} agent density ({label})")
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, f"density_{run_tag}_{label.replace(' ', '_')}.png")
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"  histogram saved to {fig_path}")

    train_counts = per_frame_agent_counts(stores, split="train")
    train_median = summarize(train_counts)["median"] if train_counts.size else 0.0
    print(f"\n=== GATE DECISION (train split, official protocol): {density_verdict(train_median)} ===")


if __name__ == "__main__":
    main()
