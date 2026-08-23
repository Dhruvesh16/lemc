import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemc.models.lemc import FEAT_DIM, LEMCModule, compute_frame_stats


def make_batch(B=4, T=15, N=6, all_valid=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    agent_tracks = torch.rand(B, T, N, 4, generator=g)
    agent_tracks[..., 2:] = agent_tracks[..., 2:].clamp(min=0.01)  # w,h > 0, like real boxes
    if all_valid:
        agent_mask = torch.ones(B, T, N, dtype=torch.bool)
    else:
        agent_mask = torch.rand(B, T, N, generator=g) > 0.3
        agent_mask[:, :, 0] = True  # ego always valid
    return agent_tracks, agent_mask


def test_compute_frame_stats_shape():
    agent_tracks, agent_mask = make_batch(B=3, T=10, N=5)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert stats.shape == (3, 9, FEAT_DIM)
    assert torch.isfinite(stats).all()


def test_compute_frame_stats_all_masked_transition_no_nan():
    # zero density at every transition -> must fall back cleanly, no NaN leaking out
    agent_tracks, _ = make_batch(B=2, T=5, N=3)
    agent_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    agent_mask[:, :, 0] = True  # ego always valid, but treat it as excluded from "other agents" here
    # simulate zero *other-agent* density by masking everyone but index 0, and
    # verifying stats aggregated over N still produce no NaNs (ego alone contributes
    # one valid transition per frame, not zero -- so also test genuinely all-agents-absent)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert torch.isfinite(stats).all()

    agent_mask_empty = torch.zeros(2, 5, 3, dtype=torch.bool)  # nobody valid anywhere
    stats_empty = compute_frame_stats(agent_tracks, agent_mask_empty)
    assert torch.isfinite(stats_empty).all()
    assert (stats_empty[..., 7] == 0).all()  # valid_count feature must read 0
    assert (stats_empty[..., 6] == 1.0).all()  # median_scale_ratio fallback == 1.0 (no change)


def test_compute_frame_stats_no_other_agent_slots_at_all():
    # max_agents=1 -> zero "other agent" slots exist in the tensor at all (not
    # just zero valid ones) -- nanmedian on an empty reduction dim raises
    # IndexError unless explicitly short-circuited.
    agent_tracks, agent_mask = make_batch(B=2, T=5, N=1)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert torch.isfinite(stats).all()
    assert (stats[..., :6] == 0).all()  # no displacement signal possible with zero other agents
    assert (stats[..., 7] == 0).all()  # other_valid_count == 0


def test_single_other_agent_does_not_influence_displacement_stats():
    # MIN_OTHER_AGENTS=2: exactly one other agent present must NOT be trusted
    # (falls back to zero), even though that one agent has real displacement.
    B, T, N = 1, 3, 2  # ego + exactly 1 other agent
    agent_tracks = torch.zeros(B, T, N, 4)
    agent_tracks[..., 2:] = 0.1  # w,h
    agent_tracks[0, 0, 1] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    agent_tracks[0, 1, 1] = torch.tensor([0.9, 0.9, 0.1, 0.1])  # huge real displacement, other agent
    agent_tracks[0, 2, 1] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    agent_mask = torch.ones(B, T, N, dtype=torch.bool)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert torch.allclose(stats[0, 0, :6], torch.zeros(6))  # displacement stats gated off, n=1 other agent


def test_ego_excluded_from_displacement_but_included_in_scale_ratio():
    # ego alone (no other agents) moving a lot must NOT show up in displacement
    # stats (would be circular), but DOES count toward the scale-ratio median
    # (the doc's "works even on a single track" cue).
    B, T, N = 1, 3, 1  # ego only
    agent_tracks = torch.zeros(B, T, N, 4)
    agent_tracks[0, 0, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    agent_tracks[0, 1, 0] = torch.tensor([0.9, 0.9, 0.1, 0.2])  # ego moves a lot AND grows
    agent_tracks[0, 2, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    agent_mask = torch.ones(B, T, N, dtype=torch.bool)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert torch.allclose(stats[0, :, :6], torch.zeros(2, 6))  # ego's own displacement never leaks in
    assert torch.isclose(stats[0, 0, 6], torch.tensor(2.0), atol=1e-4)  # but ego's own height ratio does


def test_scale_ratio_matches_manual_computation():
    B, T, N = 1, 3, 1
    agent_tracks = torch.zeros(B, T, N, 4)
    agent_tracks[0, 0, 0] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    agent_tracks[0, 1, 0] = torch.tensor([0.5, 0.5, 0.1, 0.2])  # height doubled
    agent_tracks[0, 2, 0] = torch.tensor([0.5, 0.5, 0.1, 0.1])  # height halved back
    agent_mask = torch.ones(B, T, N, dtype=torch.bool)
    stats = compute_frame_stats(agent_tracks, agent_mask)
    assert torch.isclose(stats[0, 0, 6], torch.tensor(2.0), atol=1e-4)
    assert torch.isclose(stats[0, 1, 6], torch.tensor(0.5), atol=1e-4)


def test_lemc_forward_shapes_and_wh_untouched():
    agent_tracks, agent_mask = make_batch(B=4, T=15, N=8)
    lemc = LEMCModule(hidden_dim=16, temporal_net="gru")
    compensated, residual = lemc(agent_tracks, agent_mask)
    assert compensated.shape == (4, 15, 4)
    assert residual.shape == (4, 14, 2)
    assert torch.equal(compensated[..., 2:], agent_tracks[:, :, 0, 2:])  # w,h bit-for-bit untouched


def test_lemc_forward_conv1d_variant():
    agent_tracks, agent_mask = make_batch(B=2, T=15, N=4)
    lemc = LEMCModule(hidden_dim=16, temporal_net="conv1d")
    compensated, residual = lemc(agent_tracks, agent_mask)
    assert compensated.shape == (2, 15, 4)
    assert residual.shape == (2, 14, 2)


def test_lemc_zero_shift_at_t0():
    agent_tracks, agent_mask = make_batch(B=2, T=15, N=4)
    lemc = LEMCModule(hidden_dim=16)
    compensated, _ = lemc(agent_tracks, agent_mask)
    # shift is defined as zero at t=0 by construction (cumsum padded), so
    # compensated position at t=0 must equal the raw ego position at t=0
    assert torch.allclose(compensated[:, 0, :2], agent_tracks[:, 0, 0, :2])


def test_lemc_gradients_flow_to_module_params():
    agent_tracks, agent_mask = make_batch(B=2, T=15, N=4)
    lemc = LEMCModule(hidden_dim=16)
    compensated, _ = lemc(agent_tracks, agent_mask)
    compensated.sum().backward()
    assert lemc.residual_head.weight.grad is not None
    assert lemc.residual_head.weight.grad.abs().sum() > 0
