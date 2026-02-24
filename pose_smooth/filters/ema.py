"""Score-weighted EMA smoothing with missing-joint handling."""

from __future__ import annotations

from multiprocessing import process
from shlex import join
from typing import List

from pose_smooth.config import SmoothConfig
from pose_smooth.types import FrameRecord
import numpy as np


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
        self.config = config.validate()
        self.smoothed = np.zeros((num_keypoints, 3))
        self.seen = np.zeros(num_keypoints, dtype=bool)
        
       
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
        joints = frame["keypoints"]
        missing = np.zeros(len(joints), dtype=bool)
        used_observation = np.zeros(len(joints), dtype=bool)
        teleport_rejected = np.zeros(len(joints), dtype=bool)
        for joint_indx in range(len(joints)):
            x, y, score = joints[joint_indx]
            observation = np.array([x, y])
            if score < self.config.min_score:
                  missing[joint_indx] = True
            # when seen
            if self.seen[joint_indx]:
                s_x, s_y, s_score = self.smoothed[joint_indx]
                current_smoothed = np.array([s_x, s_y])
                jump = np.linalg.norm(observation - current_smoothed)
                # teleport gating
                if jump > self.config.max_jump_px and score < self.config.teleport_trust_score:
                    missing[joint_indx] = True
                    teleport_rejected[joint_indx] = True
                if missing[joint_indx]:
                    self.smoothed[joint_indx, 2] = s_score * self.config.score_decay
                else:
                  # EMA update 
                  self.smoothed[joint_indx, 0] = self.config.alpha * x + (1.0 - self.config.alpha) * s_x
                  self.smoothed[joint_indx, 1] = self.config.alpha * y + (1.0 - self.config.alpha) * s_y
                  self.smoothed[joint_indx, 2] = self.config.alpha * score + (1.0 - self.config.alpha) * s_score
                  used_observation[joint_indx] = True
            # when unseen and valid
            elif not missing[joint_indx]:
                  self.smoothed[joint_indx] = joints[joint_indx]
                  self.seen[joint_indx] = True
                  missing[joint_indx] = False
                  used_observation[joint_indx] = True

        frameRecord = {}
        frameRecord["frame_idx"] = frame["frame_idx"]
        if "timestamp_s" in frame:
            frameRecord["timestamp_s"] = frame["timestamp_s"]
        frameRecord["keypoints"] = self.smoothed.tolist()
        frameRecord["meta"] = {
                          "used_observation": used_observation.tolist(),
                          "teleport_rejected": teleport_rejected.tolist(),
                          "missing": missing.tolist(),
                          }
        return frameRecord
                    

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
    processed_frames = []
    if len(frames) > 0:
        num_keypoints = len(frames[0]["keypoints"])
        poseEMASmoother = PoseEMASmoother(num_keypoints, config)
        for frame in frames:
            processed_frames.append(poseEMASmoother.process_frame(frame))
    return processed_frames