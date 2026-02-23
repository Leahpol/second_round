"""Tests for JSONL I/O and sanitization behavior."""

import json

from pose_smooth.io.jsonl import read_jsonl, write_jsonl


def test_read_sanitizes_nan_and_uses_last_known(tmp_path):
    p = tmp_path / "input.jsonl"
    lines = [
        {"frame_idx": 0, "keypoints": [[10.0, 20.0, 0.9], [1.0, 2.0, 0.8]]},
        {"frame_idx": 1, "keypoints": [[float("nan"), float("nan"), 0.7], [3.0, 4.0, 0.9]]},
    ]
    with p.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, allow_nan=True))
            f.write("\n")

    frames = read_jsonl(p, num_keypoints=2)
    kp0 = frames[1]["keypoints"][0]
    assert kp0[0] == 10.0
    assert kp0[1] == 20.0
    assert kp0[2] == 0.0


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "roundtrip.jsonl"
    src = [{"frame_idx": 0, "keypoints": [[1.0, 2.0, 0.9], [3.0, 4.0, 0.1]]}]
    write_jsonl(p, src)
    dst = read_jsonl(p, num_keypoints=2)
    assert dst[0]["frame_idx"] == 0
    assert dst[0]["keypoints"] == src[0]["keypoints"]
