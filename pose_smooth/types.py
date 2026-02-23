"""Shared runtime types for pose smoothing."""

from typing import List, NotRequired, TypedDict


class FrameMeta(TypedDict):
    used_observation: List[bool]
    teleport_rejected: List[bool]
    missing: List[bool]


class FrameRecord(TypedDict):
    frame_idx: int
    keypoints: List[List[float]]
    timestamp_s: NotRequired[float]
    meta: NotRequired[FrameMeta]
