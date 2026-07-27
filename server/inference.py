"""Driving the A2SB inference subprocess with live progress and cancellation."""
from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from typing import Callable, Optional

from server.audio import is_likely_corrupted_audio
from server.config import (
    INFERENCE_CWD,
    INFERENCE_SCRIPT,
    PYTHONPATH_FOR_INFERENCE,
)
from server.process import iter_output_lines, parse_eta_seconds, parse_progress, terminate_tree

LogSink = Callable[[str], None]
#: (fraction complete in 0..1 or None when unknown, seconds remaining or None)
ProgressSink = Callable[[Optional[float], Optional[float]], None]


class InferenceCancelled(Exception):
    """Raised when a job was cancelled while the subprocess was running."""


class InferenceError(RuntimeError):
    def __init__(self, message: str, tail: str = ""):
        super().__init__(message)
        self.tail = tail


def run_a2sb_inference(
    input_path: str,
    output_path: str,
    steps: int,
    cutoff_hz: int,
    batch_size: int,
    on_log: LogSink,
    on_progress: ProgressSink,
    cancel_event: threading.Event,
) -> None:
    """Run one channel through A2SB, streaming progress until it finishes.

    Raises `InferenceCancelled` if `cancel_event` is set mid-run, and
    `InferenceError` if the process fails or produces unusable audio.
    """
    # UpsampleMask computes FFT bin indices via (n_fft * freq / sampling_rate),
    # so the cutoff MUST be in Hz to produce the correct bin boundary.
    command = [
        "python3",
        "-u",
        INFERENCE_SCRIPT,
        "-f", input_path,
        "-o", output_path,
        "-n", str(int(steps)),
        "-c", str(int(cutoff_hz)),
        "-b", str(int(batch_size)),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH_FOR_INFERENCE
    env["PYTHONUNBUFFERED"] = "1"

    on_log(f"$ {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=INFERENCE_CWD,
            env=env,
            # Own process group so cancellation can reach the nested Lightning
            # process that A2SB_upsample_api.py spawns, not just the wrapper.
            start_new_session=True,
        )
    except OSError as exc:
        raise InferenceError(
            f"Could not launch the inference process in {INFERENCE_CWD}. "
            f"Set A2SB_INFERENCE_CWD if the A2SB source lives elsewhere.",
            tail=str(exc),
        ) from exc

    tail: deque[str] = deque(maxlen=60)
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
    for line in iter_output_lines(process.stdout):
        stripped = line.strip()
        if not stripped:
            continue
        tail.append(stripped)
        fraction = parse_progress(stripped)
        if fraction is not None:
            on_progress(fraction, parse_eta_seconds(stripped))
        else:
            on_log(stripped)

    returncode = process.wait()
    watcher.join(timeout=1.0)

    if cancelled or cancel_event.is_set():
        raise InferenceCancelled()

    if returncode != 0:
        raise InferenceError(
            f"Inference exited with code {returncode}.",
            tail="\n".join(tail),
        )

    if not os.path.exists(output_path):
        raise InferenceError(
            "Inference finished without writing an output file. This is usually a "
            "permissions problem on /app/outputs or /debug.",
            tail="\n".join(tail),
        )

    if is_likely_corrupted_audio(output_path):
        raise InferenceError(
            "Inference produced audio that failed validation (silent, clipped, or "
            "noise-like). The cutoff may be masking the entire spectrum.",
            tail="\n".join(tail),
        )
