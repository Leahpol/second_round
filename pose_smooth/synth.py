"""Synthetic pose dataset generation for demo/testing."""

from __future__ import annotations

from typing import List

from pose_smooth.types import FrameRecord

import numpy as np


def generate_synthetic_frames(
    frames: int = 240,
    num_keypoints: int = 17,
    jitter_std: float = 8.0,
    dropout_prob: float = 0.08,
    teleport_prob: float = 0.03,
    seed: int = 7,
) -> List[FrameRecord]:
    """
    Generate synthetic moving skeleton frames for demo and tests.

    Inputs:
    - frames: number of frames to generate.
    - num_keypoints: joints per frame.
    - jitter_std: Gaussian noise sigma for x/y.
    - dropout_prob: probability a joint drops out in a frame.
    - teleport_prob: probability of a sudden large jump for a non-dropped joint.
    - seed: RNG seed for reproducibility.

    Returns:
    - List[FrameRecord] following assignment JSONL schema.

    Goal:
    - Simulate smooth motion + noise.
    - Add random dropouts (low score + NaN or zero coordinates).
    - Add random teleport spikes (large coordinate jumps).

    Hints:
    - Start from a moving center trajectory and per-joint offsets.
    - Sample scores high by default, low for dropouts.
    - Keep output deterministic for a fixed seed.
    """
    # deterministic random number generator
    rnd = np.random.RandomState(seed)

    # moving center trajectory only
    t = np.linspace(0.0, 1.0, frames, dtype=float)
    angle = 2.0 * np.pi * t
    center_x = 320.0 + 80.0 * np.cos(angle)
    center_y = 240.0 + 40.0 * np.sin(angle)
    center_trajectory = np.stack([center_x, center_y], axis=1)  

    # per-joint offsets 
    k = np.arange(num_keypoints, dtype=float)
    base_ang = 2.0 * np.pi * (k / max(num_keypoints, 1))
    r = 40.0 + 10.0 * np.sin(2.0 * base_ang)
    per_joint_offsets = np.stack([r * np.cos(base_ang), r * np.sin(base_ang)], axis=1)  

    # smooth motion + noise
    frames_record = []
    for i in range(frames):
        base_xy = center_trajectory[i][None, :] + per_joint_offsets  
        # noise 
        jitter = rnd.normal(loc=0.0, scale=jitter_std, size=(num_keypoints, 2))
        base_noise_xy = base_xy + jitter

        score = np.full((num_keypoints,), 0.99, dtype=float)
        keypoints = np.concatenate([base_noise_xy, score[:, None]], axis=1)  

        # dropout
        dropped = rnd.rand(num_keypoints) < dropout_prob
        keypoints[dropped, 0] = np.nan
        keypoints[dropped, 1] = np.nan
        keypoints[dropped, 2] = 0.01

        # teleport 
        teleported = (~dropped) & (rnd.rand(num_keypoints) < teleport_prob)
        if np.any(teleported):
            n = int(np.sum(teleported))
            keypoints[teleported, 0] += rnd.normal(loc=0.0, scale=80.0, size=n)
            keypoints[teleported, 1] += rnd.normal(loc=0.0, scale=80.0, size=n)

        frames_record.append({"frame_idx": i, "keypoints": keypoints.tolist()})

    return frames_record