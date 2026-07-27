"""Subprocess streaming utilities shared by inference and training runners.

Splitting tqdm's \\r redraws and killing nested Lightning process groups are
non-obvious enough that they must not be duplicated.  Both `inference.py` and
`training.py` import from here.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
from typing import Generator, Optional


#: Lightning's progress bar, e.g. "Predicting DataLoader 0:  45%|####  | 9/20 [00:12<00:15,  1.4s/it]".
_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_RATIO_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_ETA_RE = re.compile(r"\[\d+:\d+<(\d+):(\d+)(?::(\d+))?")


def parse_eta_seconds(line: str) -> Optional[float]:
    """Parse tqdm ETA from a progress line, returning seconds or None."""
    match = _ETA_RE.search(line)
    if not match:
        return None
    a, b, c = match.group(1), match.group(2), match.group(3)
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def parse_progress(line: str) -> Optional[float]:
    """Parse tqdm progress fraction (0..1) from a progress line, or None."""
    percent = _PERCENT_RE.search(line)
    if percent:
        return min(max(int(percent.group(1)) / 100.0, 0.0), 1.0)
    ratio = _RATIO_RE.search(line)
    if ratio:
        done, total = int(ratio.group(1)), int(ratio.group(2))
        if total > 0:
            return min(max(done / total, 0.0), 1.0)
    return None


def iter_output_lines(stream) -> Generator[str, None, None]:
    """Split a byte stream on both newline and carriage return.

    tqdm redraws with ``\\r``, so line-buffered reads would return one enormous
    line at the end of the run instead of a progress feed.
    """
    buffer = b""
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        if chunk in (b"\n", b"\r"):
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
                buffer = b""
            continue
        buffer += chunk
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


def terminate_tree(process: subprocess.Popen) -> None:
    """Stop a subprocess and any children it spawned in the same process group."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
