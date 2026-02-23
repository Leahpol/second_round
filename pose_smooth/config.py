"""Configuration and validation helpers."""

from dataclasses import dataclass

DEFAULT_NUM_KEYPOINTS = 17
DEFAULT_ALPHA = 0.6
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MAX_JUMP_PX = 35.0
DEFAULT_SCORE_DECAY = 0.95
DEFAULT_TELEPORT_TRUST_SCORE = 0.9


@dataclass(frozen=True)
class SmoothConfig:
    """Smoothing parameters for score-weighted EMA."""

    alpha: float = DEFAULT_ALPHA
    min_score: float = DEFAULT_MIN_SCORE
    max_jump_px: float = DEFAULT_MAX_JUMP_PX
    score_decay: float = DEFAULT_SCORE_DECAY
    teleport_trust_score: float = DEFAULT_TELEPORT_TRUST_SCORE

    def validate(self) -> "SmoothConfig":
        """
        Validate numeric ranges for smoothing hyperparameters.

        Inputs:
        - self: SmoothConfig instance.

        Returns:
        - SmoothConfig: usually `self` if all fields are valid.

        Goal:
        - Fail fast when a parameter is invalid so runtime logic stays simple.

        Hints:
        - `alpha` should be in (0, 1].
        - `min_score` and `teleport_trust_score` should be in [0, 1].
        - `max_jump_px` should be >= 0.
        - `score_decay` should be in [0, 1].
        - Raise `ValueError` with a clear message.
        """
        if not (0 < self.alpha <= 1):
            raise ValueError(f"alpha should be in (0, 1]")
        if not (0 <= self.min_score <= 1) or not (0 <= self.teleport_trust_score <= 1):
            raise ValueError(f"min_score and teleport_trust_score should be in [0, 1]")
        if self.max_jump_px < 0:
            raise ValueError(f"max_jump_px should be >= 0")
        if not (0 <= self.score_decay <= 1):
            raise ValueError(f"score_decay should be in [0, 1]")
        return self
