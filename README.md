# Intern Assignment Starter (No Solution Included)

This folder is a starter template for the temporal pose smoothing assignment. Core implementation is intentionally omitted.

## Goal
Implement a small installable Python package that smooths per-frame 2D pose keypoints over time, handles missing/low-confidence joints, and provides CLI + tests.

## Required CLI
- `pose-smooth smooth --in input.jsonl --out output.jsonl --alpha 0.6 --min-score 0.3 --max-jump-px 35 --score-decay 0.95`
- `pose-smooth metrics --in input.jsonl [--smoothed output.jsonl]`
- `pose-smooth demo --out-dir demo_out/ --frames 240 --jitter 8 --dropout-prob 0.08`
- `python -m pose_smooth ...` should also work.

## Input schema
Each JSONL line:
```json
{
  "frame_idx": 0,
  "timestamp_s": 0.0,
  "keypoints": [[x, y, score], ...]
}
```

## Where You Should Code
Open this file first:
- `FILES_TO_EDIT.md`

Implement TODOs in package code and script files listed there.

Important:
- Tests are already written for you.
- Do not edit tests; run them to validate your code.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Validate your work
```bash
pytest -q
pose-smooth demo --out-dir demo_out
```

## Success criteria
- `pip install -e .` works
- `pose-smooth demo --out-dir demo_out` runs end-to-end
- `demo_out/input.jsonl` and `demo_out/output.jsonl` exist
- metrics show `jitter_after < jitter_before`
- `pytest -q` passes

## Required submission notes (anti-bullshit)
Include these in your submission:
1. Three real bugs you hit and how you fixed them.
2. Three failure cases you observed or expect (occlusion, low confidence bursts, jitter spikes, frame drops, etc.).
3. One concrete next improvement (for example: One Euro filter or Kalman filter).
