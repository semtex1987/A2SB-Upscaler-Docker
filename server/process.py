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
from pathlib import Path
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


def process_group_members(pgid: int) -> list[int]:
    """PIDs that currently belong to ``pgid`` (Linux ``/proc``; empty elsewhere)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    members: list[int] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
        except OSError:
            continue
        close = text.rfind(")")
        if close == -1:
            continue
        fields = text[close + 2 :].split()
        # After ``comm``: state, ppid, pgrp.
        if len(fields) < 3:
            continue
        try:
            if int(fields[2]) == pgid:
                members.append(int(entry.name))
        except ValueError:
            continue
    return members


def terminate_tree(process: subprocess.Popen, grace_sec: float = 10.0) -> None:
    """Stop a subprocess and any children it spawned in the same process group.

    SIGTERM is sent to the group first. After the parent exits — or after a
    bounded grace period — remaining group members are SIGKILL'd. A child that
    ignores SIGTERM must not keep the GPU after the job is marked cancelled.
    """
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        pass
    remaining = process_group_members(pgid)
    if not remaining:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        process.wait(timeout=min(grace_sec, 2.0))
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
