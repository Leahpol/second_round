"""Jitter and missingness metrics for pose sequences."""

from __future__ import annotations

from doctest import ELLIPSIS_MARKER
from typing import Dict, List, Optional

from pose_smooth.types import FrameRecord

import numpy as np


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
		jitter = 0.0
		frame_num = len(frames)
		pairs_num = frame_num - 1
		if frame_num >= 2:
				avarage_across_frames = 0.0
				for frame_indx in range(frame_num - 1):
						# only joints that are included
						joint_means = 0.0
						included_joints_num = 0
						curr_frame = frames[frame_indx]
						next_frame = frames[frame_indx + 1]
						for joint_indx in range(len(curr_frame["keypoints"])):
								x_curr, y_curr, score_curr = curr_frame["keypoints"][joint_indx]
								x_next, y_next, score_next = next_frame["keypoints"][joint_indx]
								if (score_curr >= min_score and score_next >= min_score):
										joint_means += np.linalg.norm(np.array([x_next, y_next]) - np.array([x_curr, y_curr]))
										included_joints_num += 1
						if included_joints_num > 0 :
								avarage_across_frames += joint_means / included_joints_num		
				jitter = avarage_across_frames / pairs_num		
		return jitter


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
		fraction_missing = 0.0
		total_joints = 0
		missing_count = 0
		if len(frames) > 0:
				for frame in frames:
						total_joints += len(frame["keypoints"])
						for joint in frame["keypoints"]:
							if joint[2] < min_score:
									missing_count += 1
				if total_joints > 0:
						fraction_missing = missing_count / total_joints
		return fraction_missing


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
		result = {}
		result["jitter_before"] = jitter(raw_frames, min_score)
		if smoothed_frames is not None:
				result["jitter_after"] = jitter(smoothed_frames, min_score)
		else:
				result["jitter_after"] = None
		result["missing_rate"] = missing_rate(raw_frames, min_score)
		return result
		
