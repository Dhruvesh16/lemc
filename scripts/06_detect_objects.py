import argparse
import glob
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.data.paths import PIE_DATA_ROOT, TRACK_STORE_DIR

_worker_model = None
_worker_device = None


def _init_worker(device):
    global _worker_model, _worker_device
    import torch

    if device == "cpu":
        torch.set_num_threads(1)  # avoid N processes each oversubscribing all cores with intra-op threads
    from ultralytics import YOLO

    _worker_model = YOLO("yolov8n.pt")
    _worker_device = device


def _process_video(task):
    scene_id, frame_dir = task
    from lemc.data.detector import detect_and_track_video

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frame_paths:
        return scene_id, {}
    frame_numbers = [int(os.path.basename(p).split(".")[0]) for p in frame_paths]
    dets_by_index = detect_and_track_video(_worker_model, frame_paths, scene_id.replace("/", "_"), device=_worker_device)
    return scene_id, {frame_numbers[i]: boxes for i, boxes in dets_by_index.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pie"], default="pie")
    parser.add_argument("--workers", type=int, default=8, help="ignored when --device is a GPU (single process used)")
    parser.add_argument("--device", default="cpu", help="'cpu', or a GPU index like '0' for model.track(device=...)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N videos (for testing)")
    args = parser.parse_args()

    images_root = os.path.join(PIE_DATA_ROOT, "images")
    tasks = []
    for set_id in sorted(os.listdir(images_root)):
        set_dir = os.path.join(images_root, set_id)
        if not os.path.isdir(set_dir):
            continue
        for video_id in sorted(os.listdir(set_dir)):
            video_dir = os.path.join(set_dir, video_id)
            if os.path.isdir(video_dir):
                tasks.append((f"{set_id}/{video_id}", video_dir))

    if args.limit:
        tasks = tasks[: args.limit]

    use_gpu = args.device != "cpu"
    out_path = os.path.join(TRACK_STORE_DIR, f"{args.dataset}_detections.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.time()
    all_detections = {}

    if use_gpu:
        # single process: a GPU is one accelerator, not N cores -- multiprocessing
        # here would just make processes fight over the same device.
        print(f"{len(tasks)} videos to process on device={args.device} (single process)")
        _init_worker(args.device)
        for i, task in enumerate(tasks):
            scene_id, dets = _process_video(task)
            all_detections[scene_id] = dets
            n_with_dets = sum(1 for v in dets.values() if v)
            print(f"[{i + 1}/{len(tasks)}] {scene_id}: {n_with_dets} frames with detections ({time.time() - t0:.0f}s elapsed)")
            if (i + 1) % 5 == 0:
                with open(out_path, "wb") as f:
                    pickle.dump(all_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"{len(tasks)} videos to process with {args.workers} CPU workers")
        with mp.Pool(args.workers, initializer=_init_worker, initargs=("cpu",)) as pool:
            for i, (scene_id, dets) in enumerate(pool.imap_unordered(_process_video, tasks)):
                all_detections[scene_id] = dets
                n_with_dets = sum(1 for v in dets.values() if v)
                print(f"[{i + 1}/{len(tasks)}] {scene_id}: {n_with_dets} frames with detections ({time.time() - t0:.0f}s elapsed)")
                if (i + 1) % 5 == 0:
                    with open(out_path, "wb") as f:
                        pickle.dump(all_detections, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_path, "wb") as f:
        pickle.dump(all_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done in {time.time() - t0:.0f}s, saved to {out_path}")


if __name__ == "__main__":
    main()
