#!/usr/bin/env python3
"""Generate a synthetic pose JSONL dataset."""

from __future__ import annotations

import argparse
import os

from pose_smooth import synth as ps_synth
from pose_smooth.io import write_jsonl


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
    parser = argparse.ArgumentParser(prog="make-synth-dataset")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--jitter", type=float, default=8.0)
    parser.add_argument("--dropout-prob", type=float, default=0.08)
    parser.add_argument("--teleport-prob", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-keypoints", type=int, default=17)
    args = parser.parse_args()

    frames = ps_synth.generate_synthetic_frames(
        frames=args.frames,
        num_keypoints=args.num_keypoints,
        jitter_std=args.jitter,
        dropout_prob=args.dropout_prob,
        teleport_prob=args.teleport_prob,
        seed=args.seed,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_jsonl(args.out, frames)
    print(f"Wrote {len(frames)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
