"""JSONL input/output and frame sanitization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pose_smooth.config import DEFAULT_NUM_KEYPOINTS
from pose_smooth.types import FrameRecord


def read_jsonl(path: str | Path, num_keypoints: int = DEFAULT_NUM_KEYPOINTS) -> List[FrameRecord]:
    """
    Read frame records from JSONL and sanitize invalid values.

    Inputs:
    - path: file path to input JSONL.
    - num_keypoints: expected joint count per frame (default 17).

    Returns:
    - List[FrameRecord]: parsed frames with cleaned keypoints.

    Goal:
    - Parse each line into a frame dict.
    - Validate required fields (`frame_idx`, `keypoints`).
    - Enforce each keypoint has length 3: [x, y, score].
    - Handle NaN/inf for x/y by replacing coordinates with last-known (or 0,0 if none)
      and force score to 0 for that keypoint so later logic treats it as missing.

    Hints:
    - Keep per-joint `last_valid_xy` state while reading.
    - Clamp score into [0, 1] and treat non-finite score as 0.
    - Raise `ValueError` with line number context when schema is broken.
    """
    raise NotImplementedError("Implement read_jsonl")


def write_jsonl(path: str | Path, frames: Iterable[FrameRecord]) -> None:
    """
    Write frame records to JSONL, one frame per line.

    Inputs:
    - path: output file path.
    - frames: iterable of FrameRecord-like dicts.

    Returns:
    - None

    Goal:
    - Serialize each frame as compact JSON and append newline.

    Hints:
    - Use UTF-8.
    - Use `json.dumps(..., separators=(",", ":"))` for compact output.
    - You may allow NaN on write if your read path sanitizes it.
    """
    raise NotImplementedError("Implement write_jsonl")
