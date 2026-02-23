#!/usr/bin/env python3
"""Generate a synthetic pose JSONL dataset."""

from __future__ import annotations


def main() -> int:
    """
    CLI wrapper to generate and save synthetic pose data.

    Inputs:
    - command-line flags such as `--out`, `--frames`, `--jitter`, `--dropout-prob`.

    Returns:
    - int exit code.

    Goal:
    - Parse arguments.
    - Call `pose_smooth.synth.generate_synthetic_frames`.
    - Write output JSONL.

    Hints:
    - Reuse `pose_smooth.io.jsonl.write_jsonl`.
    - Ensure parent directory exists before writing.
    """
    raise NotImplementedError("Implement scripts/make_synth_dataset.py")


if __name__ == "__main__":
    raise SystemExit(main())
