import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.data.paths import PIE_DATA_ROOT, TRACK_STORE_DIR
from lemc.data.track_store import build_pie_track_store, save_track_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pie", "jaad"], required=True)
    args = parser.parse_args()

    if args.dataset == "pie":
        stores = build_pie_track_store(PIE_DATA_ROOT)
        out_path = os.path.join(TRACK_STORE_DIR, "pie_track_store.pkl")
    else:
        raise NotImplementedError(
            "JAAD data is not on disk yet -- build_jaad_track_store() will mirror "
            "build_pie_track_store() once JAAD is downloaded and verified (plan Step 9)."
        )

    save_track_store(stores, out_path)

    n_scenes = len(stores)
    n_frames = sum(len(s.frame_to_boxes) for s in stores.values())
    n_peds = sum(len(s.ped_frames) for s in stores.values())
    by_split = {}
    for s in stores.values():
        by_split[s.set_split] = by_split.get(s.set_split, 0) + 1

    print(f"Built track store: {n_scenes} scenes, {n_frames} annotated frames, {n_peds} pedestrian tracks")
    print(f"Scenes by split: {by_split}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
