"""Training orchestration: dataset vetting, finetune subprocess, and checkpoint management.

`run_finetune` streams live output from `finetune.py` via the same tqdm-aware
line splitter used by the inference runner.  Progress is reported per-split.
`vet_dataset` reuses `spectral_scan` from `server/analysis.py` — the same
algorithm as the CLI tool — so the Train tab and `training/vet_dataset.py`
always agree.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from server.analysis import spectral_scan
from server.config import (
    ENSEMBLE_CONFIG_PATH,
    FINETUNED_CKPT_DIR,
    TRAIN_MIN_FREE_BYTES,
    TRAINING_APP_ROOT,
    TRAINING_CKPT_DIR,
    TRAINING_OUTPUT_DIR,
    TRAINING_SCRIPT,
    VET_CHECK_HZ,
    VET_PASS_HZ,
)
from server.process import (
    iter_output_lines,
    parse_eta_seconds,
    parse_progress,
    terminate_tree,
)

import librosa

LogSink = Callable[[str], None]
ProgressSink = Callable[[float, Optional[float]], None]

# Split names as finetune.py expects them
SPLIT_BOTH = "both"
SPLIT_FIRST = "0.0-0.5"
SPLIT_SECOND = "0.5-1.0"

_FINETUNED_NAMES = [
    "A2SB_twosplit_0.0_0.5_finetuned.ckpt",
    "A2SB_twosplit_0.5_1.0_finetuned.ckpt",
]

# finetune.py announces per-split progress like:
#   "Split 0.0-0.5 starts at global_step=N; training until M"
import re
_SPLIT_HEADER_RE = re.compile(
    r"Split\s+([\d.]+)-([\d.]+)\s+starts at global_step=(\d+);\s+training until (\d+)"
)


class TrainingCancelled(Exception):
    """Raised when the training job was cancelled mid-run."""


class TrainingError(RuntimeError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


# ---------------------------------------------------------------------------
# Dataset vetting
# ---------------------------------------------------------------------------

@dataclass
class VetResult:
    path: str
    name: str
    size_bytes: int
    duration_sec: float
    sample_rate: int
    hf_edge_hz: float
    shelf: bool
    verdict: str   # "pass" | "check" | "reject"
    note: str


def vet_file(path: str) -> VetResult:
    """Vet a single audio file for training suitability.

    Reuses `spectral_scan` from `server/analysis.py` with vet_dataset.py's
    thresholds so the UI and the CLI tool always agree.
    """
    p = Path(path)
    y, sr = librosa.load(path, sr=None, mono=True, duration=120.0)
    edge_hz, shelf = spectral_scan(y, int(sr))
    size_bytes = p.stat().st_size
    duration_sec = float(len(y) / sr)

    if edge_hz >= VET_PASS_HZ:
        verdict = "pass"
        note = f"Content runs to {edge_hz / 1000:.1f} kHz — genuinely full-bandwidth."
    elif edge_hz >= VET_CHECK_HZ:
        verdict = "check"
        note = (
            f"Content to {edge_hz / 1000:.1f} kHz — could be a genuine dark master "
            f"or a 320 kbps transcode. Check the shelf flag and listen."
        )
    else:
        verdict = "reject"
        if shelf:
            note = (
                f"Brickwall cliff at {edge_hz / 1000:.1f} kHz — this is a lossy transcode. "
                f"Training on it teaches the model to output silence above the cutoff."
            )
        else:
            note = (
                f"Content only to {edge_hz / 1000:.1f} kHz — too band-limited to be useful "
                f"training material for bandwidth extension."
            )

    return VetResult(
        path=path,
        name=p.name,
        size_bytes=size_bytes,
        duration_sec=duration_sec,
        sample_rate=int(sr),
        hf_edge_hz=edge_hz,
        shelf=shelf,
        verdict=verdict,
        note=note,
    )


def vet_dataset(paths: list[str]) -> list[VetResult]:
    """Vet a list of audio file paths for training suitability."""
    results = []
    for path in paths:
        try:
            results.append(vet_file(path))
        except Exception as exc:  # noqa: BLE001
            p = Path(path)
            results.append(VetResult(
                path=path,
                name=p.name,
                size_bytes=p.stat().st_size if p.exists() else 0,
                duration_sec=0.0,
                sample_rate=0,
                hf_edge_hz=0.0,
                shelf=False,
                verdict="reject",
                note=f"Could not read file: {exc}",
            ))
    return results


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight(splits: str, training_data_dir: Optional[str] = None) -> list[str]:
    """Return a list of problems. Empty list means OK to proceed."""
    problems = []

    # Release checkpoints
    ckpt_dir = Path(TRAINING_CKPT_DIR)
    needed = []
    if splits in (SPLIT_BOTH, SPLIT_FIRST):
        needed.append(ckpt_dir / "A2SB_twosplit_0.0_0.5_release.ckpt")
    if splits in (SPLIT_BOTH, SPLIT_SECOND):
        needed.append(ckpt_dir / "A2SB_twosplit_0.5_1.0_release.ckpt")
    missing = [str(c) for c in needed if not c.is_file()]
    if missing:
        problems.append(
            f"Release checkpoints not found: {', '.join(missing)}. "
            f"They are downloaded by the container entrypoint — did it run?"
        )

    # A2SB framework
    main_py = Path(TRAINING_APP_ROOT) / "main.py"
    if not main_py.is_file():
        problems.append(
            f"A2SB main.py not found at {main_py}. "
            f"Set A2SB_APP_ROOT to the directory containing main.py."
        )

    # finetune.py
    if not Path(TRAINING_SCRIPT).is_file():
        problems.append(
            f"finetune.py not found at {TRAINING_SCRIPT}. "
            f"Rebuild the image with COPY training/ /app/training/."
        )

    # Training data
    if training_data_dir is not None:
        td = Path(training_data_dir)
        if not td.is_dir():
            problems.append(
                f"Training data directory does not exist: {training_data_dir}."
            )

    # Disk space
    stat = shutil.disk_usage(str(TRAINING_OUTPUT_DIR))
    if stat.free < TRAIN_MIN_FREE_BYTES:
        gb_free = stat.free / 1024 ** 3
        gb_needed = TRAIN_MIN_FREE_BYTES / 1024 ** 3
        problems.append(
            f"Only {gb_free:.1f} GB free; {gb_needed:.0f} GB needed. "
            f"Each checkpoint is ~2.3 GB and training keeps up to 4 per split."
        )

    return problems


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    split: str                          # e.g. "0.0-0.5"
    split_index: int                    # 0 or 1
    split_total: int                    # 1 or 2
    step: int
    max_steps: int
    stage: str
    fraction: float                     # overall 0..1
    eta_sec: Optional[float]


def run_finetune(
    data_dir: str,
    output_dir: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    splits: str,
    val_frac: float,
    val_every: Optional[int],
    val_samples: Optional[int],
    restart: bool,
    on_log: LogSink,
    on_progress: Callable[[TrainingProgress], None],
    cancel_event: threading.Event,
) -> None:
    """Launch finetune.py as a subprocess and stream its output.

    Raises `TrainingCancelled` if cancelled, `TrainingError` on failure.
    """
    command = [
        sys.executable, "-u", TRAINING_SCRIPT,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--splits", splits,
        "--val-frac", str(val_frac),
    ]
    if val_every is not None:
        command += ["--val-every", str(val_every)]
    if val_samples is not None:
        command += ["--val-samples", str(val_samples)]
    if restart:
        command.append("--restart")

    # Force CSVLogger so metrics.csv is guaranteed (configs set logger: null,
    # which Lightning resolves to TensorBoard if installed, CSV otherwise).
    command += ["--", "--trainer.logger=CSVLogger"]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["A2SB_APP_ROOT"] = TRAINING_APP_ROOT
    env["A2SB_CKPT_DIR"] = TRAINING_CKPT_DIR

    on_log(f"$ {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise TrainingError(
            f"Could not launch finetune.py at {TRAINING_SCRIPT}.",
            detail=str(exc),
        ) from exc

    n_splits = 1 if splits != SPLIT_BOTH else 2
    state = {
        "split": "",
        "split_index": 0,
        "split_step": 0,
        "split_max": steps,
        "eta_sec": None,
    }
    tail: list[str] = []
    cancelled = False

    def watch_for_cancel() -> None:
        nonlocal cancelled
        while process.poll() is None:
            if cancel_event.wait(timeout=0.5):
                cancelled = True
                terminate_tree(process)
                return

    watcher = threading.Thread(target=watch_for_cancel, daemon=True)
    watcher.start()

    assert process.stdout is not None
    for raw_line in iter_output_lines(process.stdout):
        line = raw_line.strip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 120:
            tail.pop(0)

        # Detect split header to update split index and max_steps.
        header = _SPLIT_HEADER_RE.search(line)
        if header:
            lo, hi = header.group(1), header.group(2)
            state["split"] = f"{lo}-{hi}"
            state["split_step"] = int(header.group(3))
            state["split_max"] = int(header.group(4))
            # Derive split index from lo: "0.0" -> 0, "0.5" -> 1.
            state["split_index"] = 0 if lo == "0.0" else 1
            on_log(line)
            continue

        frac = parse_progress(line)
        if frac is not None:
            state["eta_sec"] = parse_eta_seconds(line)
            # Overall = splits done + current fraction of current split.
            done_splits = state["split_index"]
            overall = (done_splits + frac) / n_splits
            on_progress(TrainingProgress(
                split=state["split"],
                split_index=state["split_index"],
                split_total=n_splits,
                step=state["split_step"],
                max_steps=state["split_max"],
                stage=f"Split {state['split']} — step {state['split_step']}/{state['split_max']}",
                fraction=min(max(overall, 0.0), 1.0),
                eta_sec=state["eta_sec"],
            ))
        else:
            on_log(line)

    returncode = process.wait()
    watcher.join(timeout=1.0)

    if cancelled or cancel_event.is_set():
        raise TrainingCancelled()

    if returncode != 0:
        raise TrainingError(
            f"finetune.py exited with code {returncode}.",
            detail="\n".join(tail[-40:]),
        )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

@dataclass
class CheckpointStatus:
    active: str                        # "release" | "finetuned" | "mixed"
    finetuned_paths: dict[str, str]    # name -> path, only if the file exists
    release_paths: dict[str, str]      # name -> path
    ensemble_config: str


def checkpoint_status() -> CheckpointStatus:
    """Read the ensemble YAML to report which checkpoints are active."""
    config_path = Path(ENSEMBLE_CONFIG_PATH)
    release = {
        "A2SB_twosplit_0.0_0.5_release.ckpt": str(
            Path(TRAINING_CKPT_DIR) / "A2SB_twosplit_0.0_0.5_release.ckpt"
        ),
        "A2SB_twosplit_0.5_1.0_release.ckpt": str(
            Path(TRAINING_CKPT_DIR) / "A2SB_twosplit_0.5_1.0_release.ckpt"
        ),
    }
    finetuned_dir = Path(FINETUNED_CKPT_DIR)
    finetuned = {
        name: str(finetuned_dir / name)
        for name in _FINETUNED_NAMES
        if (finetuned_dir / name).is_file()
    }

    active = "release"
    if config_path.is_file():
        try:
            data = yaml.safe_load(config_path.read_text())
            ckpts = (data.get("model") or {}).get("pretrained_checkpoints") or []
            n_finetuned = sum(1 for c in ckpts if "finetuned" in str(c))
            if n_finetuned == len(ckpts) and len(ckpts) > 0:
                active = "finetuned"
            elif n_finetuned > 0:
                active = "mixed"
        except Exception:  # noqa: BLE001
            pass

    return CheckpointStatus(
        active=active,
        finetuned_paths=finetuned,
        release_paths=release,
        ensemble_config=str(config_path),
    )


def activate_checkpoints(finetuned_dir: Optional[str] = None) -> int:
    """Point the ensemble config at whatever finetuned checkpoints exist.

    Returns the number of splits activated.  This mirrors update_ckpt_config.py
    so activation works without restarting the container (each restore spawns
    a fresh subprocess that re-reads the config).
    """
    src_dir = Path(finetuned_dir or FINETUNED_CKPT_DIR)
    config_path = Path(ENSEMBLE_CONFIG_PATH)
    if not config_path.is_file():
        raise TrainingError(f"Ensemble config not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    ckpts = list((data.get("model") or {}).get("pretrained_checkpoints") or [])
    if len(ckpts) != 2:
        raise TrainingError(
            "Expected 2 pretrained_checkpoints in ensemble config; "
            f"found {len(ckpts)}."
        )

    activated = 0
    for i, name in enumerate(_FINETUNED_NAMES):
        p = src_dir / name
        if p.is_file():
            ckpts[i] = str(p)
            activated += 1

    data["model"]["pretrained_checkpoints"] = ckpts
    config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return activated


def revert_to_release() -> None:
    """Restore the ensemble config to the release checkpoints."""
    config_path = Path(ENSEMBLE_CONFIG_PATH)
    if not config_path.is_file():
        raise TrainingError(f"Ensemble config not found: {config_path}")

    ckpt_dir = Path(TRAINING_CKPT_DIR)
    data = yaml.safe_load(config_path.read_text())
    data.setdefault("model", {})["pretrained_checkpoints"] = [
        str(ckpt_dir / "A2SB_twosplit_0.0_0.5_release.ckpt"),
        str(ckpt_dir / "A2SB_twosplit_0.5_1.0_release.ckpt"),
    ]
    config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


# ---------------------------------------------------------------------------
# CSV metrics reader
# ---------------------------------------------------------------------------

def read_training_metrics(output_dir: str, splits: str = SPLIT_BOTH) -> dict[str, list[dict]]:
    """Read loss curves from Lightning's CSV logger output.

    Returns {split_tag: [{"step": int, "train_loss": float, ...}, ...]}
    """
    result: dict[str, list[dict]] = {}
    base = Path(output_dir)

    dirs = []
    if splits in (SPLIT_BOTH, SPLIT_FIRST):
        dirs.append(("0.0-0.5", base / "split_0.0_0.5"))
    if splits in (SPLIT_BOTH, SPLIT_SECOND):
        dirs.append(("0.5-1.0", base / "split_0.5_1.0"))

    for tag, split_dir in dirs:
        csv_files = sorted(
            (split_dir / "lightning_logs").glob("*/metrics.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if (split_dir / "lightning_logs").exists() else []

        if not csv_files:
            result[tag] = []
            continue

        rows: list[dict] = []
        try:
            text = csv_files[0].read_text(encoding="utf-8")
            lines = text.splitlines()
            if not lines:
                result[tag] = []
                continue
            headers = lines[0].split(",")
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) != len(headers):
                    continue
                row: dict = {}
                for h, v in zip(headers, parts):
                    v = v.strip()
                    if v == "":
                        continue
                    try:
                        row[h.strip()] = float(v)
                    except ValueError:
                        row[h.strip()] = v
                if row:
                    rows.append(row)
        except OSError:
            pass
        result[tag] = rows

    return result
