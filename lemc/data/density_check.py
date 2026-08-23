from __future__ import annotations

import numpy as np

from .track_store import SceneStore


def per_frame_agent_counts(stores: dict[str, SceneStore], split: str | None = None) -> np.ndarray:
    counts = []
    for store in stores.values():
        if split is not None and store.set_split != split:
            continue
        for boxes in store.frame_to_boxes.values():
            counts.append(len(boxes))
    return np.array(counts, dtype=np.int32)


def density_verdict(median_count: float) -> str:
    if median_count >= 4:
        return "green: proceed with plain multi-agent LEMC"
    elif median_count >= 2:
        return "amber: add scale-change cue from the start"
    else:
        return "red: multi-agent LEMC cannot work as designed -- needs a detector-based fallback"


def summarize(counts: np.ndarray) -> dict:
    return {
        "n_frames": int(counts.size),
        "median": float(np.median(counts)),
        "mean": float(np.mean(counts)),
        "p25": float(np.percentile(counts, 25)),
        "p75": float(np.percentile(counts, 75)),
        "p95": float(np.percentile(counts, 95)),
        "max": int(counts.max()) if counts.size else 0,
    }
