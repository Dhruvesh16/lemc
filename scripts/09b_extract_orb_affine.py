"""Step 7b — Offline ORB *full affine* ego-motion for PIE (window frames only).

Stores 2x3 partial-affine matrices per frame pair. Supervision target for a
pedestrian at (cx,cy) is the affine-induced displacement
    M @ [cx,cy,1] - [cx,cy]
not the global translation (tx,ty) — depth/position-dependent parallax makes
translation-only compensation *increase* track variance (empirically ~20%
flatten vs ~89% with full affine on standing+moving windows).
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

from lemc.data.dataset import build_window_index, protocol_tte_kwargs
from lemc.data.paths import PIE_DATA_ROOT, TRACK_STORE_DIR, dataset_image_root, frame_image_path, track_store_path
from lemc.data.track_store import load_track_store
from lemc.utils.config import load_config

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(1)


def _dilate_box(cx, cy, w, h, frac=0.10):
    w2, h2 = w * (1 + frac), h * (1 + frac)
    return int(cx - w2 / 2), int(cy - h2 / 2), int(cx + w2 / 2), int(cy + h2 / 2)


def _pedestrian_mask(shape_hw, boxes_at_f, dilate_frac=0.10):
    H, W = shape_hw
    mask = np.ones((H, W), dtype=np.uint8) * 255
    for box in boxes_at_f.values():
        cx, cy, w, h = box
        x1, y1, x2, y2 = _dilate_box(cx, cy, w, h, dilate_frac)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask[y1:y2, x1:x2] = 0
    return mask


def _orb_affine(img_t, img_t1, mask_t, mask_t1, nfeatures=800, ratio=0.75):
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
    M, _ = cv2.estimateAffinePartial2D(pts_t, pts_t1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or not np.isfinite(M).all():
        return None
    return M.astype(np.float32)  # (2, 3)


def _ped_only_boxes(store, frame: int) -> dict:
    boxes = store.frame_to_boxes.get(frame, {})
    return {pid: box for pid, box in boxes.items() if "_det_" not in str(pid)}


def _process_scene(args):
    scene_id, frames_sorted, image_root, ped_boxes_by_frame, dilate_frac, dataset = args
    if dataset == "jaad":
        if not os.path.isdir(os.path.join(image_root, scene_id)):
            return scene_id, {}, "missing_dir"
    else:
        set_id, video_id = scene_id.split("/", 1)
        if not os.path.isdir(os.path.join(image_root, set_id, video_id)):
            return scene_id, {}, "missing_dir"

    affines = {}
    n_fail = 0
    prev_gray = prev_mask = prev_f = None

    for f in frames_sorted:
        path = frame_image_path(image_root, scene_id, f, dataset)
        if not os.path.exists(path):
            prev_gray = prev_mask = prev_f = None
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            prev_gray = prev_mask = prev_f = None
            continue
        mask = _pedestrian_mask(gray.shape, ped_boxes_by_frame.get(f, {}), dilate_frac)

        if prev_gray is not None and prev_f is not None and f == prev_f + 1:
            M = _orb_affine(prev_gray, gray, prev_mask, mask)
            if M is None:
                n_fail += 1
            else:
                affines[prev_f] = M  # transition prev_f -> f

        prev_gray, prev_mask, prev_f = gray, mask, f

    return scene_id, affines, f"ok fail={n_fail} ok={len(affines)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pie_lemc_on_augmented_orb.yaml")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--dilate-frac", type=float, default=0.10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    cfg = load_config(args.config)
    stores = load_track_store(track_store_path(cfg))
    dataset = cfg.get("dataset", "pie")
    image_root = dataset_image_root(cfg)
    default_out = "jaad_orb_affines.pkl" if dataset == "jaad" else "pie_orb_affines.pkl"
    out_path = args.out or os.path.join(TRACK_STORE_DIR, cfg.get("orb_affines_file", default_out))

    needed: dict[str, set[int]] = {sid: set() for sid in stores}
    t_obs, t_pred = cfg["protocol"]["t_obs"], cfg["protocol"]["t_pred"]
    max_agents = cfg["max_agents"]
    tte_kw = protocol_tte_kwargs(cfg)
    for split, ov in (
        ("train", cfg["train"].get("overlap_train", 0.5)),
        ("val", cfg["protocol"].get("overlap_eval", 0.5)),
        ("test", cfg["protocol"].get("overlap_eval", 0.5)),
    ):
        for scene_id, _, start, _ in build_window_index(
            stores, split, t_obs, t_pred, max_agents, overlap=ov, **tte_kw
        ):
            # obs + pred frames so targets can be compensated too
            for f in range(start, start + t_obs + t_pred):
                needed[scene_id].add(f)

    jobs = []
    for sid in sorted(stores.keys()):
        frames = sorted(needed.get(sid, set()))
        if len(frames) < 2:
            continue
        ped_boxes = {f: _ped_only_boxes(stores[sid], f) for f in frames}
        jobs.append((sid, frames, image_root, ped_boxes, args.dilate_frac, dataset))

    print(f"ORB affine extract: {len(jobs)} scenes, workers={args.workers}", flush=True)
    t0 = time.time()
    all_aff: dict[str, dict[int, np.ndarray]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_scene, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            sid, affines, status = fut.result()
            all_aff[sid] = affines
            done += 1
            print(f"  [{done}/{len(jobs)}] {sid} {status} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_aff, f, protocol=pickle.HIGHEST_PROTOCOL)
    n = sum(len(v) for v in all_aff.values())
    print(f"saved {n} affines -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

    # sanity: |translation| vs OBD (PIE only) + stopped near-zero
    if dataset != "pie":
        print(f"skip OBD sanity (dataset={dataset})")
        return
    mags, speeds, stopped = [], [], []
    for sid, store in stores.items():
        for f, M in all_aff.get(sid, {}).items():
            mag = float(np.hypot(M[0, 2], M[1, 2]))
            sp = store.frame_to_ego.get(f)
            if sp is None:
                continue
            sp = float(sp)
            mags.append(mag)
            speeds.append(sp)
            if sp < 0.5:
                stopped.append(mag)
    if len(mags) >= 10:
        r, p = pearsonr(mags, speeds)
        print(f"=== |tx,ty| vs OBD (n={len(mags)}) r={r:.4f} R2={r*r:.4f} ===")
    if stopped:
        print(f"=== stopped mean|t|={np.mean(stopped):.3f} median={np.median(stopped):.3f} (n={len(stopped)}) ===")


if __name__ == "__main__":
    main()
