"""Step 7 — Offline ORB ego-motion extractor for PIE.

For each consecutive frame pair, estimate a partial affine (dx, dy) via ORB
keypoints with pedestrian boxes masked out, then cache to disk.

Sanity checks printed at the end:
  - correlation of |ORB shift| vs OBD speed (expect positive)
  - near-zero shift when the car is stopped
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from scipy.stats import pearsonr

from lemc.data.paths import PIE_DATA_ROOT, TRACK_STORE_DIR, track_store_path
from lemc.data.track_store import load_track_store
from lemc.utils.config import load_config

# OpenCV/BLAS can oversubscribe with process workers — pin each worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(1)


def _dilate_box(cx, cy, w, h, frac=0.10):
    w2, h2 = w * (1 + frac), h * (1 + frac)
    return int(cx - w2 / 2), int(cy - h2 / 2), int(cx + w2 / 2), int(cy + h2 / 2)


def _pedestrian_mask(shape_hw, boxes_at_f, dilate_frac=0.10):
    """Binary mask: True where pixels are *background* (usable for ego-motion)."""
    H, W = shape_hw
    mask = np.ones((H, W), dtype=np.uint8) * 255
    for box in boxes_at_f.values():
        # only mask real pedestrians (det_* are background structure we want)
        # caller passes ped-only boxes
        cx, cy, w, h = box
        x1, y1, x2, y2 = _dilate_box(cx, cy, w, h, dilate_frac)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask[y1:y2, x1:x2] = 0
    return mask


def _orb_shift(img_t, img_t1, mask_t, mask_t1, nfeatures=800, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(img_t, mask_t)
    k2, d2 = orb.detectAndCompute(img_t1, mask_t1)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 10:
        return None
    pts_t = np.float32([k1[m.queryIdx].pt for m in good])
    pts_t1 = np.float32([k2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(pts_t, pts_t1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    dx, dy = float(M[0, 2]), float(M[1, 2])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    return dx, dy


def _ped_only_boxes(store, frame: int) -> dict:
    """Exclude detector pseudo-agents (ids containing '_det_') from the mask —
    static scene structure (cars, signs) *should* vote in ORB."""
    boxes = store.frame_to_boxes.get(frame, {})
    return {pid: box for pid, box in boxes.items() if "_det_" not in str(pid)}


def _process_scene(args):
    scene_id, frames_sorted, image_root, ped_boxes_by_frame, dilate_frac = args
    # frames_sorted: list of int frames that exist as images
    set_id, video_id = scene_id.split("/", 1)
    video_dir = os.path.join(image_root, set_id, video_id)
    if not os.path.isdir(video_dir):
        return scene_id, {}, "missing_dir"

    shifts = {}
    n_fail = 0
    prev_gray = None
    prev_mask = None
    prev_f = None

    for f in frames_sorted:
        path = os.path.join(video_dir, f"{f:05d}.jpg")
        if not os.path.exists(path):
            prev_gray = prev_mask = prev_f = None
            continue
        try:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            gray = None
        if gray is None:
            prev_gray = prev_mask = prev_f = None
            continue
        mask = _pedestrian_mask(gray.shape, ped_boxes_by_frame.get(f, {}), dilate_frac)

        if prev_gray is not None and prev_f is not None and f == prev_f + 1:
            result = _orb_shift(prev_gray, gray, prev_mask, mask)
            if result is None:
                n_fail += 1
            else:
                shifts[prev_f] = result  # transition prev_f -> f

        prev_gray, prev_mask, prev_f = gray, mask, f

    return scene_id, shifts, f"ok fail={n_fail} ok={len(shifts)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pie_lemc_on_augmented.yaml")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--dilate-frac", type=float, default=0.10)
    parser.add_argument("--max-scenes", type=int, default=None, help="debug: limit scenes")
    parser.add_argument(
        "--window-only",
        action="store_true",
        default=True,
        help="Only extract transitions that appear in train/val/test windows (default).",
    )
    parser.add_argument("--all-frames", action="store_true", help="Extract every consecutive pair (slow).")
    args = parser.parse_args()
    if args.all_frames:
        args.window_only = False

    # unbuffered progress under ProcessPoolExecutor
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    cfg = load_config(args.config)
    stores = load_track_store(track_store_path(cfg))
    image_root = args.image_root or os.path.join(PIE_DATA_ROOT, "images")
    out_path = args.out or os.path.join(TRACK_STORE_DIR, "pie_orb_shifts.pkl")

    needed_frames: dict[str, set[int]] = {sid: set() for sid in stores}
    if args.window_only:
        from lemc.data.dataset import build_window_index, protocol_tte_kwargs

        t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
        max_agents = cfg["max_agents"]
        tte_kw = protocol_tte_kwargs(cfg)
        for split, ov in (
            ("train", cfg["train"].get("overlap_train", 0.5)),
            ("val", cfg["protocol"].get("overlap_eval", 0.5)),
            ("test", cfg["protocol"].get("overlap_eval", 0.5)),
        ):
            windows = build_window_index(
                stores, split, t_obs, t_pred, max_agents, overlap=ov, **tte_kw
            )
            for scene_id, _, start, _ in windows:
                # need frames start .. start+t_obs-1 inclusive for consecutive pairs
                for f in range(start, start + t_obs):
                    needed_frames[scene_id].add(f)
        n_needed = sum(len(v) for v in needed_frames.values())
        print(f"window-only mode: {n_needed} unique frames across splits")

    scene_ids = sorted(stores.keys())
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]

    jobs = []
    for sid in scene_ids:
        store = stores[sid]
        if args.window_only:
            frames = sorted(needed_frames.get(sid, set()))
        else:
            frames = sorted(store.frame_to_boxes.keys())
        if len(frames) < 2:
            continue
        # only ped boxes for masking on the frames we need
        ped_boxes = {f: _ped_only_boxes(store, f) for f in frames}
        jobs.append((sid, frames, image_root, ped_boxes, args.dilate_frac))

    print(f"ORB extract: {len(jobs)} scenes, workers={args.workers}, out={out_path}", flush=True)
    t0 = time.time()
    all_shifts: dict[str, dict[int, tuple]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_scene, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            sid, shifts, status = fut.result()
            all_shifts[sid] = shifts
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done}/{len(jobs)}] last={sid} {status} ({elapsed:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_shifts, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_pairs = sum(len(v) for v in all_shifts.values())
    print(f"saved {n_pairs} frame-pair shifts -> {out_path} ({time.time() - t0:.0f}s)", flush=True)

    # --- sanity checks ---
    mags, speeds = [], []
    stopped_mags = []
    for sid, store in stores.items():
        scene_orb = all_shifts.get(sid, {})
        for f, (dx, dy) in scene_orb.items():
            mag = float(np.hypot(dx, dy))
            sp = store.frame_to_ego.get(f)
            if sp is None:
                continue
            sp = float(sp)
            mags.append(mag)
            speeds.append(sp)
            if sp < 0.5:  # essentially stopped
                stopped_mags.append(mag)

    if len(mags) >= 10:
        r, p = pearsonr(mags, speeds)
        print(f"=== Sanity: |ORB| vs OBD speed (n={len(mags)}) ===")
        print(f"  Pearson r={r:.4f}, R^2={r*r:.4f}, p={p:.2e}")
    if stopped_mags:
        print(
            f"=== Sanity: car stopped (speed<0.5, n={len(stopped_mags)}) ==="
            f" mean|ORB|={np.mean(stopped_mags):.3f}px  median={np.median(stopped_mags):.3f}px"
        )


if __name__ == "__main__":
    main()
