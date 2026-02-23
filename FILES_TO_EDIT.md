# Exactly What To Edit

Implement TODOs in these files only.

## 1) `pose_smooth/config.py`
- Implement: `SmoothConfig.validate`
- Input: config fields (`alpha`, `min_score`, `max_jump_px`, `score_decay`, `teleport_trust_score`)
- Output: validated `SmoothConfig` (or raises `ValueError`)
- Purpose: ensure bad params fail early.

## 2) `pose_smooth/io/jsonl.py`
- Implement: `read_jsonl`, `write_jsonl`
- Input: JSONL frames from disk
- Output: list of sanitized frame dicts / JSONL file output
- Purpose: robust parsing + NaN handling + consistent schema.

## 3) `pose_smooth/filters/ema.py`
- Implement: `PoseEMASmoother.__init__`, `PoseEMASmoother.process_frame`, `smooth_frames`
- Input: frame(s) with `keypoints` shape (V,3)
- Output: smoothed frame(s) + `meta` flags
- Purpose: core temporal reasoning (EMA + missing policy + teleport gating).

## 4) `pose_smooth/metrics/jitter.py`
- Implement: `jitter`, `missing_rate`, `compute_metrics`
- Input: raw/smoothed frame sequences and `min_score`
- Output: jitter and missingness stats
- Purpose: verify smoothing quality numerically.

## 5) `pose_smooth/synth.py`
- Implement: `generate_synthetic_frames`
- Input: frame count, jitter/dropout/teleport probabilities, seed
- Output: synthetic frame list in assignment schema
- Purpose: reproducible local data for demo + testing.

## 6) `pose_smooth/cli.py`
- Implement: `build_parser`, `main`
- Input: CLI args
- Output: command dispatch and user-visible metrics/output paths
- Purpose: end-to-end usability.

## 7) `scripts/make_synth_dataset.py`
- Implement: `main`
- Input: command-line generation args
- Output: JSONL dataset file
- Purpose: standalone dataset generation helper.

# Do Not Edit (provided for validation)
- `tests/test_io_roundtrip.py`
- `tests/test_ema_smoothing.py`
- `tests/test_metrics.py`

# Usually Leave Unchanged
- `pyproject.toml`
- `pose_smooth/types.py`
- `pose_smooth/__init__.py`
- `pose_smooth/__main__.py`
- package `__init__.py` files
- `.gitignore`

# Done Checklist
- `pip install -e .` succeeds in a fresh venv
- `pose-smooth demo --out-dir demo_out` creates:
  - `demo_out/input.jsonl`
  - `demo_out/output.jsonl`
- Demo prints metrics where `jitter_after < jitter_before`
- `pytest -q` passes
