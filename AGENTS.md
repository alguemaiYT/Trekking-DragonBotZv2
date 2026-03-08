# Repository Guidelines

## Project Structure & Module Organization
This repository is versioned by implementation stage under `versions/`:
- `versions/v2-legacy/`: legacy color-based tracker (`cone_tracker/`) and tests.
- `versions/v3/`: v2 tracker plus ONNX/CNN experiments in `CNN/` and `geracnn/`.
- `versions/v4/`: YOLO ONNX detector (`cone_detector_v4.py`) and model assets.

Top-level files:
- `requirements.txt`: shared Python dependencies.
- `scripts/cleanup_generated.sh`: removes generated outputs and caches.
- `README.md`: root overview; version-specific READMEs document execution details.

## Build, Test, and Development Commands
Use Python 3 in a virtual environment.

```bash
pip install -r requirements.txt
pip install -r versions/v3/requirements.txt   # when working on v3 CNN/ONNX flows
pytest versions/v2-legacy/tests
pytest versions/v3/tests
python3 versions/v3/start.py
python3 versions/v4/cone_detector_v4.py --source versions/v4/dataset --profile balanced
bash scripts/cleanup_generated.sh
```

`pytest` validates tracker/detector behavior and integration paths. Run only the version you changed, then run both suites for cross-version changes.

## Coding Style & Naming Conventions
Follow existing Python style in the repo:
- 4-space indentation, snake_case for functions/modules, PascalCase for classes.
- Keep files and CLI scripts descriptive (`run_batch_detection.sh`, `test_video_output.py`).
- Prefer small, testable functions in `cone_tracker/` and explicit CLI args in entry scripts.

No formatter/linter config is currently enforced; keep style consistent with neighboring code and avoid mixing unrelated refactors in the same commit.

## Testing Guidelines
- Framework: `pytest` (see `requirements.txt`).
- Add tests in the matching version folder: `versions/<version>/tests/`.
- Name files `test_*.py`; name test functions `test_*`.
- For bug fixes, add or update a regression test near the affected module.

## Commit & Pull Request Guidelines
Recent history favors short, descriptive commit subjects (often Portuguese) focused on one change area. Use imperative, scoped messages, for example:
- `v4: ajusta follow error e telemetria JSONL`
- `tests(v3): adiciona cobertura para video output`

For pull requests, include:
- What changed and why.
- Affected version(s) (`v2-legacy`, `v3`, `v4`).
- Test evidence (`pytest ...` output summary) and sample commands used.
- Screenshots or output paths when visual detection behavior changes.
