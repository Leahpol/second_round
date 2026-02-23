"""Score-weighted EMA smoothing with missing-joint handling."""

from __future__ import annotations

from typing import List

from pose_smooth.config import SmoothConfig
from pose_smooth.types import FrameRecord


class PoseEMASmoother:
    """Temporal pose smoother with missing data and teleport gating."""

    def __init__(self, num_keypoints: int, config: SmoothConfig):
        """
        Build stateful per-joint smoother.

        Inputs:
        - num_keypoints: number of joints V.
        - config: validated smoothing hyperparameters.

        Returns:
        - None

        Goal:
        - Store config.
        - Allocate per-joint state for:
          - smoothed x/y
          - smoothed score
          - seen/not-seen flags

        Hints:
        - Numpy arrays are easiest here: shape (V,2) for xy and (V,) for score.
        - Call `config.validate()` once in constructor.
        """
        raise NotImplementedError("Implement PoseEMASmoother.__init__")

    def process_frame(self, frame: FrameRecord) -> FrameRecord:
        """
        Process one frame and return a smoothed frame with meta flags.

        Inputs:
        - frame: dict with `frame_idx`, optional `timestamp_s`, and `keypoints` (V,3).

        Returns:
        - FrameRecord-like dict containing:
          - same `frame_idx` and optional `timestamp_s`
          - smoothed `keypoints` [[x,y,score], ...]
          - `meta` with boolean arrays:
            - `used_observation`
            - `teleport_rejected`
            - `missing`

        Goal:
        - For each joint:
          - If score < min_score => missing.
          - If seen before and jump > max_jump_px and score not very high => reject as teleport (missing).
          - If missing and seen => hold xy, decay score by score_decay.
          - If valid and seen => EMA update for xy and score.
          - If valid and unseen => initialize state from observation.

        Hints:
        - Jump is Euclidean distance between observation xy and current smoothed xy.
        - Use `teleport_trust_score` to allow trusted large jumps.
        - Keep meta flags aligned to joint index.
        """
        raise NotImplementedError("Implement PoseEMASmoother.process_frame")


def smooth_frames(frames: List[FrameRecord], config: SmoothConfig) -> List[FrameRecord]:
    """
    Smooth a full sequence of frames.

    Inputs:
    - frames: list of FrameRecord.
    - config: SmoothConfig.

    Returns:
    - List[FrameRecord]: same length as input, each frame processed in order.

    Goal:
    - Construct one `PoseEMASmoother` and feed frames sequentially.

    Hints:
    - Handle empty input by returning `[]`.
    - Infer `num_keypoints` from first frame.
    """
    raise NotImplementedError("Implement smooth_frames")
