"""JSONL input/output and frame sanitization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pose_smooth.config import DEFAULT_NUM_KEYPOINTS
from pose_smooth.types import FrameRecord
import json
import math


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

    path = Path(path)
    frames = []
    last_valid_xy = [0.0, 0.0]
    # open the file and read the lines
    with path.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                frame_data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            # Basic schema checks per line
            if "frame_idx" not in frame_data or "keypoints" not in frame_data:
                raise ValueError(f"Missing required fields on line {line_num}") 
            if len(frame_data["keypoints"]) != num_keypoints:
                raise ValueError(f"Expected {num_keypoints} keypoints on line {line_num}")
            for keypoint in frame_data["keypoints"]:
                if len(keypoint) != 3:
                    raise ValueError(f"Expected 3 values for keypoint on line {line_num}")
                # sanitize the keypoint coordinates and score
                x, y, score = keypoint
                if not math.isfinite(x):
                    keypoint[0] = last_valid_xy[0]
                    keypoint[2] = 0
                if not math.isfinite(y):
                    keypoint[1] = last_valid_xy[1]
                    keypoint[2] = 0
                if score < 0 or not math.isfinite(score):
                    keypoint[2] = 0
                if score > 1:
                    keypoint[2] = 1
                last_valid_xy = [keypoint[0], keypoint[1]]
            # cast the line into a FrameRecord after all checks
            frame_data["keypoints"] = frame_data["keypoints"]
            frames.append(frame_data)
    return frames


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

    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        for frame in frames:
            line = json.dumps(frame, separators=(",", ":"))
            file.write(line + "\n")