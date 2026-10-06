"""On-demand TensorBoard child process.

TensorBoard is started only when someone asks for it and is bound to loopback,
because deployments publish a single port. Browser traffic reaches it through the
reverse proxy in `server.main`, which is why it runs under a path prefix.

Only one instance is ever running; `ensure_running` is idempotent and safe to
call concurrently.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from server.config import (
    TENSORBOARD_HOST,
    TENSORBOARD_PATH_PREFIX,
    TENSORBOARD_PORT,
    TENSORBOARD_RELOAD_INTERVAL_SEC,
    TENSORBOARD_STARTUP_TIMEOUT_SEC,
    TRAINING_OUTPUT_DIR,
)
from server.process import terminate_tree


@dataclass
class TensorBoardStatus:
    """What the Train tab needs to decide what to render."""

    available: bool          # the tensorboard package is importable
    running: bool
    ready: bool              # bound its port and answering requests
    #: Path the browser should load, prefixed and trailing-slashed. TensorBoard
    #: resolves its assets relatively, so dropping the trailing slash 404s them.
    url: str
    pid: Optional[int]
    error: Optional[str]
    log_dir: str
    #: True when at least one event file exists, so the UI can say "nothing to
    #: show yet" instead of rendering an empty TensorBoard.
    has_event_files: bool


def tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def find_event_files(log_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(log_dir)
    if not root.is_dir():
        return []
    return sorted(root.rglob("events.out.tfevents.*"))


class TensorBoardManager:
    def __init__(
        self,
        log_dir: str | os.PathLike[str] = TRAINING_OUTPUT_DIR,
        port: int = TENSORBOARD_PORT,
        path_prefix: str = TENSORBOARD_PATH_PREFIX,
    ) -> None:
        self.log_dir = str(log_dir)
        self.port = port
        self.path_prefix = path_prefix
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._drain_thread: Optional[threading.Thread] = None
        self._output: deque[str] = deque(maxlen=40)
        self._error: Optional[str] = None
        self._ready = False

    # -- introspection -----------------------------------------------------

    @property
    def upstream_base(self) -> str:
        """Where the proxy should forward to, prefix included."""
        return f"http://{TENSORBOARD_HOST}:{self.port}{self.path_prefix}"

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> TensorBoardStatus:
        with self._lock:
            alive = self._is_alive()
            if not alive:
                # A process that exited on its own leaves stale ready state.
                self._ready = False
            return TensorBoardStatus(
                available=tensorboard_available(),
                running=alive,
                ready=alive and self._ready,
                url=f"{self.path_prefix}/",
                pid=self._process.pid if alive and self._process else None,
                error=self._error,
                log_dir=self.log_dir,
                has_event_files=bool(find_event_files(self.log_dir)),
            )

    # -- lifecycle ---------------------------------------------------------

    def ensure_running(self) -> TensorBoardStatus:
        """Start TensorBoard if it is not already up, and return without waiting.

        Startup routinely takes 20s or more while TensorBoard scans the log tree.
        Blocking that long risks tripping an ingress timeout, so readiness is
        tracked by a background thread and callers poll `status()` instead.
        """
        with self._lock:
            if self._is_alive():
                return self.status()
            if not tensorboard_available():
                self._error = (
                    "The tensorboard package is not installed in this image. "
                    "Rebuild with `docker compose up --build`, or "
                    "`pip install tensorboard` for a local run."
                )
                return self.status()
            self._spawn()
            return self.status()

    def wait_until_ready(self, timeout: float = TENSORBOARD_STARTUP_TIMEOUT_SEC) -> bool:
        """Block until ready or timeout. For tests and non-HTTP callers."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready:
                return True
            if not self._is_alive():
                return False
            time.sleep(0.25)
        return self._ready

    def _spawn(self) -> None:
        self._error = None
        self._ready = False
        self._output.clear()
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", self.log_dir,
            "--host", TENSORBOARD_HOST,
            "--port", str(self.port),
            # Served under a prefix so the app's own proxy can forward to it
            # without rewriting TensorBoard's asset URLs.
            f"--path_prefix={self.path_prefix}",
            "--reload_interval", str(TENSORBOARD_RELOAD_INTERVAL_SEC),
        ]

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Own process group, so shutdown reaches any helper it spawns.
                start_new_session=True,
            )
        except OSError as exc:
            self._process = None
            self._error = f"Could not launch TensorBoard: {exc}"
            return

        self._drain_thread = threading.Thread(target=self._drain_output, daemon=True)
        self._drain_thread.start()
        threading.Thread(target=self._watch_readiness, daemon=True).start()

    def _drain_output(self) -> None:
        """Keep TensorBoard's output for error reporting and off the pipe buffer.

        Left unread, a full stdout pipe would eventually block TensorBoard itself.
        """
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._output.append(line)

    def _watch_readiness(self) -> None:
        """Poll the child until it answers, then publish that it is ready.

        Deliberately does not take the manager lock: this runs for tens of
        seconds, and `status()` has to stay responsive throughout so the UI can
        poll it.
        """
        deadline = time.monotonic() + TENSORBOARD_STARTUP_TIMEOUT_SEC
        probe = f"{self.upstream_base}/"
        while time.monotonic() < deadline:
            if not self._is_alive():
                self._error = (
                    "TensorBoard exited during startup"
                    + (f":\n{self._output_tail()}" if self._output_tail() else ".")
                )
                self._ready = False
                return
            try:
                with urllib.request.urlopen(probe, timeout=2.0) as response:
                    if response.status < 500:
                        self._ready = True
                        self._error = None
                        return
            except (urllib.error.URLError, OSError):
                pass  # Not bound yet; keep waiting.
            time.sleep(0.5)

        self._ready = False
        tail = self._output_tail()
        self._error = (
            f"TensorBoard did not become ready within "
            f"{TENSORBOARD_STARTUP_TIMEOUT_SEC:.0f}s"
            + (f":\n{tail}" if tail else ".")
        )

    def _output_tail(self) -> str:
        """Snapshot the child's output, letting the drain thread catch up first.

        A child that dies immediately often has its diagnosis still sitting in
        the pipe. The drain loop ends at EOF, which the exit already caused, so
        this join returns promptly.
        """
        if self._drain_thread is not None and not self._is_alive():
            self._drain_thread.join(timeout=1.0)
        return "\n".join(self._output)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._ready = False
            if process is None or process.poll() is not None:
                self._process = None
                return
            terminate_tree(process)
            self._process = None


manager = TensorBoardManager()
