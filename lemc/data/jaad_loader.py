"""JAAD annotation loader via the upstream jaad_data.JAAD interface.

Expects the JAAD 2.0 layout (annotations/, annotations_vehicle/,
annotations_attributes/, split_ids/). Images/clips are optional for building
the track store; ORB extraction requires images/ (or JAAD_clips/ extracted).
"""
from __future__ import annotations

import os
import sys

from .paths import JAAD_DATA_ROOT, _REPO_ROOT

# Prefer the in-repo JAAD 2.0 clone; fall back to LEMC_JAAD_ROOT layout.
_INREPO_JAAD = os.path.join(_REPO_ROOT, "data_external", "JAAD")


def resolve_jaad_root(data_root: str | None = None) -> str:
    if data_root:
        return data_root
    env = os.environ.get("LEMC_JAAD_ROOT")
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(os.path.join(_INREPO_JAAD, "annotations")):
        return _INREPO_JAAD
    return JAAD_DATA_ROOT


def _ensure_jaad_on_path(data_root: str) -> None:
    # Upstream jaad_data.py lives inside the JAAD root.
    if data_root not in sys.path:
        sys.path.insert(0, data_root)


def load_jaad_database(data_root: str | None = None) -> dict:
    root = resolve_jaad_root(data_root)
    assert os.path.isdir(os.path.join(root, "annotations")), (
        f"JAAD annotations not found under {root}. Clone "
        "https://github.com/ykotseruba/JAAD (branch JAAD_2.0) or set LEMC_JAAD_ROOT."
    )
    _ensure_jaad_on_path(root)
    from jaad_data import JAAD

    return JAAD(data_path=root).generate_database()


def load_jaad_splits(data_root: str | None = None, subset: str = "default") -> dict:
    """Return {train|val|test: [video_id, ...]} from split_ids/<subset>/."""
    root = resolve_jaad_root(data_root)
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(root, "split_ids", subset, f"{split}.txt")
        if not os.path.isfile(path):
            # Some checkouts use .txt names without extension variants
            path = os.path.join(root, "split_ids", subset, split)
        with open(path, "rt") as f:
            splits[split] = [line.strip() for line in f if line.strip()]
    return splits
