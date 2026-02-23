"""Synthetic pose dataset generation for demo/testing."""

from __future__ import annotations

from typing import List

from pose_smooth.types import FrameRecord


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
    raise NotImplementedError("Implement generate_synthetic_frames")
