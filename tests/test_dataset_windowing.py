import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.data.dataset import LEMCWindowDataset, build_window_index, fit_norm_stats
from lemc.data.track_store import SceneStore


def make_scene(split="train") -> SceneStore:
    store = SceneStore(scene_id=f"synthetic/{split}", width=1000.0, height=500.0, set_split=split)
    # ego present frames 0..5, agent B present frames 0..5 too (dense enough for a t_obs=2 window)
    for f in range(6):
        store.frame_to_boxes[f] = {
            "ego": (100.0 + f, 100.0, 20.0, 40.0),
            "B": (300.0 + f, 100.0, 20.0, 40.0),
        }
    store.ped_frames = {"ego": list(range(6)), "B": list(range(6))}
    store.ped_attributes = {"ego": {"crossing": 1}, "B": {"crossing": 0}}
    return store


def test_build_window_index_and_dataset_shapes():
    stores = {"scene1": make_scene("train")}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    assert len(windows) > 0

    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    item = ds[0]
    assert item["agent_tracks"].shape == (t_obs, max_agents, 4)
    assert item["agent_mask"].shape == (t_obs, max_agents)
    assert item["target_track"].shape == (t_pred, 4)
    assert item["target_mask"].shape == (t_pred,)
    assert item["agent_mask"][:, 0].all()  # ego always valid


def test_image_dim_normalization_applied():
    stores = {"scene1": make_scene("train")}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    item = ds[0]
    ego_cx = item["agent_tracks"][0, 0, 0].item()
    assert 0.0 < ego_cx < 1.0  # raw cx ~100/1000 = 0.1, well within (0,1)


def test_fit_norm_stats_then_standardizes():
    stores = {"scene1": make_scene("train")}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    ds_raw = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    stats = fit_norm_stats(ds_raw)
    assert stats.mean.shape == (2,)  # cx, cy only -- w, h are never standardized
    assert stats.std.shape == (2,)

    ds_std = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=stats)
    all_ego = torch.stack([ds_std[i]["agent_tracks"][:, 0, :] for i in range(len(ds_std))])
    # standardized cx,cy should now be roughly centered near 0
    assert all_ego[..., :2].mean().abs().item() < 1.0


def test_width_height_never_standardized():
    # h[t+1]/h[t] (LEMC's scale-ratio cue) must stay a true ratio -- z-scoring
    # w,h would let it cross zero / flip sign even though real height > 0 always.
    stores = {"scene1": make_scene("train")}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    ds_raw = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    stats = fit_norm_stats(ds_raw)
    ds_std = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=stats)

    raw_wh = ds_raw[0]["agent_tracks"][:, 0, 2:]
    std_wh = ds_std[0]["agent_tracks"][:, 0, 2:]
    assert torch.allclose(raw_wh, std_wh)  # untouched by norm_stats


def test_ego_gap_excluded_from_window_index():
    # build_window_index treats every pedestrian as the target ego in turn, so
    # gapping pedestrian "ego"'s box at frame 1 must drop only windows where
    # "ego" (not "B") is the target -- "B" has no gap and stays eligible.
    stores = {"scene1": make_scene("train")}
    del stores["scene1"].frame_to_boxes[1]["ego"]  # gap at frame 1, pedestrian "ego" only
    t_obs, t_pred, max_agents = 3, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)

    ego_windows = [w for w in windows if w[1] == "ego"]
    b_windows = [w for w in windows if w[1] == "B"]

    for scene_id, ego_pid, start_frame, agent_pids in ego_windows:
        assert not (start_frame <= 1 < start_frame + t_obs)  # pedestrian "ego" must skip the gap window
    assert len(b_windows) > 0  # pedestrian "B" (no gap) still produces its window


def test_pie_negative_one_crossing_label_becomes_nan():
    # PIE encodes "crossing intent not applicable" as -1, not NaN (confirmed:
    # 468/1842 pedestrians in the real annotation cache). A bare -1 target fed
    # into BCE silently produces garbage (even negative) loss -- must be
    # converted to NaN so it's excluded the same way as a genuinely missing label.
    stores = {"scene1": make_scene("train")}
    stores["scene1"].ped_attributes["ego"] = {"crossing": -1}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    ego_windows = [i for i in range(len(ds)) if ds.windows[i][1] == "ego"]
    assert len(ego_windows) > 0
    for i in ego_windows:
        assert torch.isnan(ds[i]["crossing_label"])


def test_zero_obd_speed_is_not_treated_as_missing():
    # OBD_speed == 0.0 (car genuinely stopped) must survive as 0.0, not be
    # silently rewritten to NaN -- a `x or np.nan` pattern would do exactly
    # that, since 0.0 is falsy in Python, corrupting the lowest-speed windows
    # the OBD correlation validation (plan Step 6) most needs.
    store = make_scene("train")
    store.frame_to_ego = {f: 0.0 for f in range(6)}  # car stopped at every frame
    stores = {"scene1": store}
    t_obs, t_pred, max_agents = 2, 1, 4
    windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    ds = LEMCWindowDataset(stores, windows, t_obs, t_pred, max_agents, norm_stats=None)
    item = ds[0]
    assert not torch.isnan(item["ego_speed"]).any()
    assert torch.all(item["ego_speed"] == 0.0)


def test_split_filtering():
    stores = {"train_scene": make_scene("train"), "val_scene": make_scene("val")}
    t_obs, t_pred, max_agents = 2, 1, 4
    train_windows = build_window_index(stores, "train", t_obs, t_pred, max_agents, overlap=0.0)
    val_windows = build_window_index(stores, "val", t_obs, t_pred, max_agents, overlap=0.0)
    assert all(w[0] == "train_scene" for w in train_windows)
    assert all(w[0] == "val_scene" for w in val_windows)
