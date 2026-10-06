"""Process-group cancellation must reap children that ignore SIGTERM."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from server.process import process_group_members, terminate_tree


def test_terminate_tree_kills_a_child_that_ignores_sigterm():
    script = """
import os, signal, subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, "-c",
     "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
)
print(child.pid, flush=True)
time.sleep(20)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert parent.stdout is not None
    line = parent.stdout.readline()
    child_pid = int(line.strip())
    pgid = os.getpgid(parent.pid)
    assert child_pid in process_group_members(pgid)
    terminate_tree(parent)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        remaining = process_group_members(pgid)
        if not remaining:
            break
        time.sleep(0.05)
    assert process_group_members(pgid) == []
    try:
        os.kill(child_pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive
