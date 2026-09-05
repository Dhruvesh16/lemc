"""Smoke test: JAAD track store builds and yields TTE windows."""
from __future__ import annotations

import os

import pytest

from lemc.data.dataset import build_window_index
from lemc.data.paths import JAAD_DATA_ROOT, TRACK_STORE_DIR
from lemc.data.track_store import build_jaad_track_store, load_track_store, save_track_store


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(JAAD_DATA_ROOT, "annotations")),
    reason="JAAD annotations not on disk",
)
def test_build_jaad_track_store():
    stores = build_jaad_track_store(JAAD_DATA_ROOT)
    assert len(stores) >= 300
    store = next(iter(stores.values()))
    assert store.width == 1920 and store.height == 1080
    assert store.set_split in ("train", "val", "test")
    assert len(store.ped_frames) >= 1
    assert len(store.frame_to_boxes) >= 1


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(TRACK_STORE_DIR, "jaad_track_store.pkl")),
    reason="jaad_track_store.pkl not built",
)
def test_jaad_window_index_tte():
    stores = load_track_store(os.path.join(TRACK_STORE_DIR, "jaad_track_store.pkl"))
    test_w = build_window_index(
        stores, "test", t_obs=16, t_pred=45, max_agents=32, overlap=0.5, tte_min=30, tte_max=60
    )
    assert len(test_w) > 100
