"""Tests for jitter and missingness metrics."""

from pose_smooth.metrics.jitter import compute_metrics, jitter, missing_rate


def test_jitter_ignores_low_confidence_and_computes_mean_step():
    frames = [
        {"frame_idx": 0, "keypoints": [[0.0, 0.0, 0.9], [10.0, 0.0, 0.2]]},
        {"frame_idx": 1, "keypoints": [[3.0, 4.0, 0.9], [25.0, 0.0, 0.2]]},
    ]
    assert abs(jitter(frames, min_score=0.3) - 5.0) < 1e-8


def test_compute_metrics_before_after_and_missing_rate():
    raw = [
        {"frame_idx": 0, "keypoints": [[0.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
        {"frame_idx": 1, "keypoints": [[10.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
        {"frame_idx": 2, "keypoints": [[20.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
    ]
    smooth = [
        {"frame_idx": 0, "keypoints": [[0.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
        {"frame_idx": 1, "keypoints": [[6.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
        {"frame_idx": 2, "keypoints": [[12.0, 0.0, 0.9], [0.0, 0.0, 0.1]]},
    ]

    m = compute_metrics(raw, smooth, min_score=0.3)
    assert m["jitter_after"] is not None
    assert m["jitter_after"] < m["jitter_before"]
    assert abs(m["missing_rate"] - missing_rate(raw, min_score=0.3)) < 1e-10
