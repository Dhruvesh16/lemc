import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.data.track_store import SceneStore, merge_detections_into_stores, window_scene


def make_synthetic_scene() -> SceneStore:
    store = SceneStore(scene_id="synthetic/video_0000", width=1920, height=1080, set_split="train")
    # 3 frames, 2 agents. 'ego' present every frame. 'B' present at frames 0 and 2
    # but missing at frame 1 -- the gap the mask must capture.
    store.frame_to_boxes = {
        0: {"ego": (100.0, 100.0, 20.0, 40.0), "B": (200.0, 100.0, 20.0, 40.0)},
        1: {"ego": (105.0, 100.0, 20.0, 40.0)},  # B absent here
        2: {"ego": (110.0, 100.0, 20.0, 40.0), "B": (400.0, 100.0, 20.0, 40.0)},
    }
    store.ped_frames = {"ego": [0, 1, 2], "B": [0, 2]}
    return store


def test_mask_marks_absent_agent_false():
    store = make_synthetic_scene()
    result = window_scene(
        store, ego_pid="ego", start_frame=0, t_obs=3, t_pred=0, agent_pids=["ego", "B"], max_agents=2
    )
    assert result is not None
    agent_tracks, agent_mask, target_track, target_mask = result

    assert agent_mask[0, 1] == True  # noqa: E712 -- B present at frame 0
    assert agent_mask[1, 1] == False  # noqa: E712 -- B absent at frame 1
    assert agent_mask[2, 1] == True  # noqa: E712 -- B present at frame 2
    assert agent_mask[:, 0].all()  # ego present every observed frame (query criterion)


def test_ego_gap_drops_window():
    store = make_synthetic_scene()
    del store.frame_to_boxes[1]["ego"]  # simulate an ego gap at frame 1
    result = window_scene(
        store, ego_pid="ego", start_frame=0, t_obs=3, t_pred=0, agent_pids=["ego", "B"], max_agents=2
    )
    assert result is None


def test_ignoring_mask_produces_visibly_wrong_displacement():
    """
    Proves the mask isn't decorative: a displacement computed WITHOUT respecting
    it (naively diffing the zero-filled track) hallucinates ~200px jumps for
    agent B at the frame where B is actually absent, purely from touching the
    zero placeholder. A mask-respecting computation must find zero valid
    transitions for B in this window instead.
    """
    store = make_synthetic_scene()
    result = window_scene(
        store, ego_pid="ego", start_frame=0, t_obs=3, t_pred=0, agent_pids=["ego", "B"], max_agents=2
    )
    agent_tracks, agent_mask, _, _ = result

    b_cx = agent_tracks[:, 1, 0]  # [200 (real), 0 (placeholder), 400 (real)]
    naive_dx = np.diff(b_cx)
    assert abs(naive_dx[0]) > 100 and abs(naive_dx[1]) > 100  # the bug this test guards against

    valid = agent_mask[:, 1]  # [True, False, True]
    both_valid = valid[:-1] & valid[1:]
    assert not both_valid.any()  # mask-respecting code finds no valid B transition here


def test_merge_detections_adds_pseudo_agents_without_becoming_ego_candidates():
    store = make_synthetic_scene()
    stores = {"synthetic/video_0000": store}
    detections = {
        "synthetic/video_0000": {
            0: {"synthetic_video_0000_det_1": (500.0, 500.0, 30.0, 60.0)},
            2: {"synthetic_video_0000_det_1": (505.0, 500.0, 30.0, 60.0)},
        }
    }
    merge_detections_into_stores(stores, detections)

    assert "synthetic_video_0000_det_1" in store.frame_to_boxes[0]
    assert "synthetic_video_0000_det_1" not in store.frame_to_boxes.get(1, {})  # untouched frame
    assert "synthetic_video_0000_det_1" in store.frame_to_boxes[2]
    assert "synthetic_video_0000_det_1" not in store.ped_frames  # never an ego candidate

    # existing real pedestrian data at frame 0 must survive the merge (update, not overwrite)
    assert "ego" in store.frame_to_boxes[0]
    assert "B" in store.frame_to_boxes[0]


def test_padding_beyond_max_agents_stays_masked_false():
    store = make_synthetic_scene()
    result = window_scene(
        store, ego_pid="ego", start_frame=0, t_obs=3, t_pred=0, agent_pids=["ego", "B"], max_agents=5
    )
    agent_tracks, agent_mask, _, _ = result
    assert agent_mask.shape == (3, 5)
    assert not agent_mask[:, 2:].any()  # padding slots never marked valid
    assert not agent_tracks[:, 2:].any()  # and stay zero
