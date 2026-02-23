"""Tests for smoothing logic and temporal edge cases."""

from pose_smooth.config import SmoothConfig
from pose_smooth.filters.ema import smooth_frames


def test_teleport_gating_rejects_large_low_confidence_jump():
    frames = [
        {"frame_idx": 0, "keypoints": [[0.0, 0.0, 0.95], [5.0, 5.0, 0.95]]},
        {"frame_idx": 1, "keypoints": [[100.0, 0.0, 0.5], [6.0, 5.0, 0.95]]},
        {"frame_idx": 2, "keypoints": [[2.0, 0.0, 0.95], [7.0, 5.0, 0.95]]},
    ]

    cfg = SmoothConfig(alpha=0.6, min_score=0.3, max_jump_px=20.0, score_decay=0.9)
    out = smooth_frames(frames, cfg)

    assert out[1]["meta"]["teleport_rejected"][0] is True
    assert out[1]["meta"]["used_observation"][0] is False
    assert out[1]["meta"]["missing"][0] is True
    assert out[1]["keypoints"][0][0] == 0.0


def test_missing_joint_holds_position_and_decays_score():
    frames = [
        {"frame_idx": 0, "keypoints": [[10.0, 10.0, 1.0]]},
        {"frame_idx": 1, "keypoints": [[12.0, 12.0, 0.1]]},
    ]

    cfg = SmoothConfig(alpha=0.5, min_score=0.3, max_jump_px=100.0, score_decay=0.8)
    out = smooth_frames(frames, cfg)

    assert out[1]["keypoints"][0][0] == 10.0
    assert out[1]["keypoints"][0][1] == 10.0
    assert abs(out[1]["keypoints"][0][2] - 0.8) < 1e-8
