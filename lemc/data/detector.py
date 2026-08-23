from __future__ import annotations

# COCO class ids for objects useful as extra "shared motion" voters: bicycle,
# car, motorcycle, bus, truck, traffic light, fire hydrant, stop sign, parking
# meter. Deliberately excludes 'person' (0) -- pedestrians are already fully
# covered by PIE's own ped_annotations; adding YOLO-detected persons risks
# double-counting the same individuals under a different id scheme.
DETECT_CLASSES = [1, 2, 3, 5, 7, 9, 10, 11, 12]

TRACKER_CONFIG = "bytetrack.yaml"


def detect_and_track_video(
    model, frame_paths: list[str], video_key: str, device: str | int | None = None
) -> dict[int, dict[str, tuple]]:
    """Runs YOLO+ByteTrack over an ordered sequence of frame image paths
    (one video's extracted frames, in frame order). Returns
    {frame_index: {pseudo_pid: (cx, cy, w, h)}} where frame_index is the
    0-based position in frame_paths (caller maps this back to the real PIE
    frame number) and pseudo_pid is f"{video_key}_det_{track_id}" -- prefixed
    so it can never collide with a real PIE pedestrian id.

    device: passed straight to model.track (e.g. 0 for first GPU, 'cpu' for
    CPU, None to let ultralytics auto-select)."""
    result: dict[int, dict[str, tuple]] = {}
    for i, path in enumerate(frame_paths):
        try:
            preds = model.track(
                path, persist=True, verbose=False, classes=DETECT_CLASSES, tracker=TRACKER_CONFIG, device=device
            )
        except Exception as e:
            print(f"  WARNING: detection failed on {path}: {e} -- skipping frame")
            continue
        if not preds:
            print(f"  WARNING: no prediction returned for {path} (likely unreadable image) -- skipping frame")
            continue
        boxes = preds[0].boxes
        if boxes.id is None:
            continue
        frame_dets = {}
        for xyxy, tid in zip(boxes.xyxy.tolist(), boxes.id.tolist()):
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            frame_dets[f"{video_key}_det_{int(tid)}"] = (cx, cy, w, h)
        if frame_dets:
            result[i] = frame_dets
    return result
