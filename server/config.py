"""Runtime paths and tunables, resolved once at import."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _ensure_runtime_dir(preferred_path: str, fallback_name: str) -> Path:
    try:
        os.makedirs(preferred_path, exist_ok=True)
        return Path(preferred_path)
    except OSError:
        fallback_path = Path(tempfile.gettempdir()) / fallback_name
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


INPUT_DIR = _ensure_runtime_dir(os.environ.get("A2SB_INPUT_DIR", "/app/inputs"), "a2sb-inputs")
OUTPUT_DIR = _ensure_runtime_dir(os.environ.get("A2SB_OUTPUT_DIR", "/app/outputs"), "a2sb-outputs")

#: Per-run artefacts live under here so re-running with different settings no
#: longer overwrites earlier results.
RUNS_DIR = _ensure_runtime_dir(str(OUTPUT_DIR / "runs"), "a2sb-runs")

#: Scratch space for the filtered channel WAVs handed to the inference process.
WORK_DIR = _ensure_runtime_dir(str(INPUT_DIR / ".work"), "a2sb-work")

INFERENCE_CWD = os.environ.get("A2SB_INFERENCE_CWD", "/app/inference")
INFERENCE_SCRIPT = "A2SB_upsample_api.py"
PYTHONPATH_FOR_INFERENCE = os.environ.get("A2SB_PYTHONPATH", "/app")

#: The model configs specify 44.1 kHz; feeding anything else only adds a lossy
#: resample round-trip inside the data loader.
MODEL_SAMPLE_RATE = 44100

BATCH_MIN = 1
BATCH_MAX = max(BATCH_MIN, _read_int_env("A2SB_UI_BATCH_MAX", 64))
BATCH_DEFAULT = min(max(_read_int_env("A2SB_DEFAULT_BATCH_SIZE", 16), BATCH_MIN), BATCH_MAX)

STEPS_MIN = 10
STEPS_MAX = 200
STEPS_DEFAULT = 50

CUTOFF_MIN_HZ = 1000
CUTOFF_MAX_HZ = 20000
CUTOFF_DEFAULT_HZ = 14000

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus", ".wma"}

#: Kept small enough to stream to the browser but dense enough to resolve the
#: 14-22 kHz band the tool exists to reconstruct.
SPECTROGRAM_WIDTH = 900
SPECTROGRAM_HEIGHT = 448

#: Rolling in-memory log retained per job; the full log is also written to disk.
LOG_RING_SIZE = _read_int_env("A2SB_LOG_RING_SIZE", 400)
