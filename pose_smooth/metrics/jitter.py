"""Jitter and missingness metrics for pose sequences."""

from __future__ import annotations

from typing import Dict, List, Optional

from pose_smooth.types import FrameRecord


def jitter(frames: List[FrameRecord], min_score: float) -> float:
    """
    Compute temporal jitter as mean frame-to-frame displacement.

    Inputs:
    - frames: sequence of keypoint frames.
    - min_score: confidence threshold for validity.

    Returns:
    - float: jitter value.

    Goal:
    - For each consecutive frame pair:
      - compute Euclidean displacement per joint
      - include only joints where both frame scores >= min_score
    - average per-frame valid joint means, then average across frames.

    Hints:
    - Return 0.0 for fewer than 2 frames or no valid pairs.
    - Numpy vectorization makes this compact and less error-prone.
    """
    raise NotImplementedError("Implement jitter")


def missing_rate(frames: List[FrameRecord], min_score: float) -> float:
    """
    Compute fraction of joints considered missing.

    Inputs:
    - frames: sequence of frames.
    - min_score: threshold; score < min_score is missing.

    Returns:
    - float in [0, 1]: missing fraction across all frame-joint entries.

    Goal:
    - Count missing joints globally and divide by total joints.

    Hints:
    - Return 0.0 for empty input.
    """
    raise NotImplementedError("Implement missing_rate")


def compute_metrics(
    raw_frames: List[FrameRecord],
    smoothed_frames: Optional[List[FrameRecord]],
    min_score: float,
) -> Dict[str, Optional[float]]:
    """
    Compute before/after jitter and raw missingness summary.

    Inputs:
    - raw_frames: unsmoothed frames.
    - smoothed_frames: optional smoothed frames.
    - min_score: threshold for validity/missingness.

    Returns:
    - dict with keys:
      - `jitter_before` (float)
      - `jitter_after` (float or None)
      - `missing_rate` (float)

    Goal:
    - Always compute `jitter_before` and `missing_rate` from raw.
    - Compute `jitter_after` only when smoothed input is provided.
    """
    raise NotImplementedError("Implement compute_metrics")
