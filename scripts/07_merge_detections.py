import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.data.paths import TRACK_STORE_DIR
from lemc.data.track_store import load_track_store, merge_detections_into_stores, save_track_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pie"], default="pie")
    args = parser.parse_args()

    store_path = os.path.join(TRACK_STORE_DIR, f"{args.dataset}_track_store.pkl")
    detections_path = os.path.join(TRACK_STORE_DIR, f"{args.dataset}_detections.pkl")
    out_path = os.path.join(TRACK_STORE_DIR, f"{args.dataset}_track_store_augmented.pkl")

    stores = load_track_store(store_path)
    detections = load_track_store(detections_path)  # same pickle load, different content shape

    n_scenes_with_dets = sum(1 for k in detections if k in stores)
    n_missing = sum(1 for k in detections if k not in stores)
    print(f"{n_scenes_with_dets} scenes matched, {n_missing} detection scenes had no matching track-store scene")

    merge_detections_into_stores(stores, detections)
    save_track_store(stores, out_path)

    n_frames_with_dets = sum(
        1
        for store in stores.values()
        for frame_boxes in store.frame_to_boxes.values()
        if any("_det_" in pid for pid in frame_boxes)
    )
    print(f"merged. {n_frames_with_dets} frames now have at least one detected pseudo-agent")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
