"""Argparse CLI for pose smoothing and evaluation."""

from __future__ import annotations

import argparse
from typing import Optional


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
    raise NotImplementedError("Implement build_parser")


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
    raise NotImplementedError("Implement main")
