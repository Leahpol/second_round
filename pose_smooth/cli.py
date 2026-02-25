"""Argparse CLI for pose smoothing and evaluation."""

from __future__ import annotations

import argparse
from typing import Optional

from pathlib import Path

from pose_smooth.config import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_JUMP_PX,
    DEFAULT_SCORE_DECAY,
    DEFAULT_TELEPORT_TRUST_SCORE,
    SmoothConfig,
)
from pose_smooth.filters.ema import smooth_frames
from pose_smooth.metrics.jitter import compute_metrics
from pose_smooth.synth import generate_synthetic_frames
from pose_smooth.io.jsonl import read_jsonl, write_jsonl



def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser with subcommands: smooth, metrics, demo.

    Inputs:
    - none

    Returns:
    - argparse.ArgumentParser configured with required flags.

    Goal:
    - Expose commands:
      - `smooth --in --out [smoothing params]`
      - `metrics --in [--smoothed] [--min-score]`
      - `demo --out-dir [data+smoothing params]`

    Hints:
    - Use `set_defaults(func=...)` per subcommand.
    - Keep argument names aligned with assignment prompt.
    """
    
    parser = argparse.ArgumentParser(prog="pose-smooth")
    subparser = parser.add_subparsers(dest="command", required=True)

    # smooth
    smooth_p = subparser.add_parser("smooth", help="Smooth an input JSONL sequence")
    smooth_p.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    smooth_p.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    smooth_p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    smooth_p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    smooth_p.add_argument("--max-jump-px", type=float, default=DEFAULT_MAX_JUMP_PX)
    smooth_p.add_argument("--score-decay", type=float, default=DEFAULT_SCORE_DECAY)
    smooth_p.add_argument("--teleport-trust-score", type=float, default=DEFAULT_TELEPORT_TRUST_SCORE)
    smooth_p.set_defaults(func=_smooth)

    # metrics
    metrics_p = subparser.add_parser("metrics", help="Compute jitter/missingness metrics")
    metrics_p.add_argument("--in", dest="in_path", required=True, help="Raw input JSONL path")
    metrics_p.add_argument("--smoothed", dest="smoothed_path", default=None, help="Optional smoothed JSONL path")
    metrics_p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    metrics_p.set_defaults(func=_metrics)

    # demo
    demo_p = subparser.add_parser("demo", help="Generate synthetic data and run smoothing")
    demo_p.add_argument("--out-dir", required=True, help="Output directory for demo artifacts")
    demo_p.add_argument("--frames", type=int, default=240)
    demo_p.add_argument("--jitter", type=float, default=8.0)
    demo_p.add_argument("--dropout-prob", type=float, default=0.08)
    demo_p.add_argument("--teleport-prob", type=float, default=0.03)
    demo_p.add_argument("--seed", type=int, default=7)
    demo_p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    demo_p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    demo_p.add_argument("--max-jump-px", type=float, default=DEFAULT_MAX_JUMP_PX)
    demo_p.add_argument("--score-decay", type=float, default=DEFAULT_SCORE_DECAY)
    demo_p.add_argument("--teleport-trust-score", type=float, default=DEFAULT_TELEPORT_TRUST_SCORE)
    demo_p.set_defaults(func=_demo)

    return parser



def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse CLI arguments and execute selected command.

    Inputs:
    - argv: optional CLI args list for testing.

    Returns:
    - int process exit code (0 for success).

    Goal:
    - Dispatch to command handlers (`smooth`, `metrics`, `demo`).
    - Print useful outputs (paths + metrics).

    Hints:
    - `args = parser.parse_args(argv)` then call `args.func(args)`.
    - Convert returned value to `int`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0

def _smooth(args):
    config = SmoothConfig(
        alpha=args.alpha,
        min_score=args.min_score,
        max_jump_px=args.max_jump_px,
        score_decay=args.score_decay,
        teleport_trust_score=args.teleport_trust_score,
    )
    frames = read_jsonl(args.in_path)
    smoothed = smooth_frames(frames, config=config)
    write_jsonl(args.out_path, smoothed)


def _metrics(args):
    raw = read_jsonl(args.in_path)
    smooth = read_jsonl(args.smoothed_path) if args.smoothed_path else None
    metrics = compute_metrics(raw, smooth, args.min_score)
    for k, v in metrics.items():
        print(f"{k}: {v}")


def _demo(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"
    output_path = out_dir / "output.jsonl"

    raw_frames = generate_synthetic_frames(
        frames=args.frames,
        jitter_std=args.jitter,
        dropout_prob=args.dropout_prob,
        teleport_prob=args.teleport_prob,
        seed=args.seed,
    )
    write_jsonl(input_path, raw_frames)

    config = SmoothConfig(
        alpha=args.alpha,
        min_score=args.min_score,
        max_jump_px=args.max_jump_px,
        score_decay=args.score_decay,
        teleport_trust_score=args.teleport_trust_score,
    )
    smoothed_frames = smooth_frames(raw_frames, config=config)
    write_jsonl(output_path, smoothed_frames)

    metrics = compute_metrics(raw_frames, smoothed_frames, args.min_score)
    for k, v in metrics.items():
        print(f"{k}: {v}")