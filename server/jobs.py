"""Job queue.

Work is submitted, not awaited. A single worker thread drains the queue so the
GPU is never double-booked, every state change is written to disk under the run
directory, and progress is broadcast to subscribers. Closing the browser has no
effect on a running job, and reopening it reattaches to the same state.

Two job kinds share the queue:
  - ``kind="restore"``  per-file restoration via the A2SB inference subprocess.
  - ``kind="train"``    fine-tuning via finetune.py (takes hours; blocks restores).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, Optional

from server.config import LOG_RING_SIZE, RUNS_DIR
from server.inference import InferenceCancelled
from server.pipeline import FileProgress, FileResult, PipelineError, restore_file
from server.serialization import camelize, snakeize
from server.training import (
    TrainingCancelled,
    TrainingError,
    TrainingProgress,
    run_finetune,
)

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"

TERMINAL_STATES = {COMPLETED, FAILED, CANCELLED, INTERRUPTED}

#: Progress broadcasts are coalesced to this interval; status changes bypass it.
BROADCAST_INTERVAL_SEC = 0.35
PERSIST_INTERVAL_SEC = 3.0

JOB_KIND_RESTORE = "restore"
JOB_KIND_TRAIN = "train"


@dataclass
class JobFile:
    name: str
    source_path: str
    cutoff_hz: int
    status: str = QUEUED
    stage: str = "Queued"
    fraction: Optional[float] = None
    eta_sec: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None


@dataclass
class TrainParams:
    data_dir: str
    output_dir: str
    steps: int
    batch_size: int
    learning_rate: float
    splits: str
    val_frac: float
    val_every: Optional[int]
    val_samples: Optional[int]
    restart: bool


@dataclass
class Job:
    id: str
    created_at: float
    steps: int
    batch_size: int
    files: list[JobFile]
    kind: str = JOB_KIND_RESTORE
    status: str = QUEUED
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    #: Present only for kind="train"; stored so the tab can display params.
    train_params: Optional[TrainParams] = None
    #: Current training stage text, e.g. "Split 0.0-0.5 — step 1200/5000".
    train_stage: str = ""
    train_fraction: Optional[float] = None
    train_eta_sec: Optional[float] = None

    def to_dict(self) -> dict:
        data = camelize(asdict(self))
        data["progress"] = self.overall_progress()
        return data

    def overall_progress(self) -> Optional[float]:
        if self.kind == JOB_KIND_TRAIN:
            return self.train_fraction
        if not self.files:
            return None
        total = 0.0
        known = False
        for entry in self.files:
            if entry.status in (COMPLETED, FAILED, CANCELLED, INTERRUPTED):
                total += 1.0
                known = True
            elif entry.fraction is not None:
                total += entry.fraction
                known = True
        return (total / len(self.files)) if known else None


class EventBroker:
    """Fan-out to SSE subscribers across the thread boundary."""

    def __init__(self) -> None:
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._lock = threading.Lock()

    def subscribe(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.add((loop, queue))

    def unsubscribe(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard((loop, queue))

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for loop, queue in targets:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # The subscriber's loop has closed; its unsubscribe will follow.
                pass


class JobStore:
    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self.runs_dir = Path(runs_dir)
        self.broker = EventBroker()
        self._jobs: dict[str, Job] = {}
        self._logs: dict[str, deque[str]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._queue: "Queue[str]" = Queue()
        self._worker: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._last_broadcast = 0.0
        self._last_persist: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._restore_from_disk()
        self._worker = threading.Thread(target=self._run_worker, name="a2sb-worker", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
        # Give the worker time to actually kill the inference process group.
        # Returning immediately would let the interpreter exit first and leave a
        # detached job holding the GPU against the next container start.
        if self._worker is not None:
            self._worker.join(timeout=15.0)

    def _restore_from_disk(self) -> None:
        """Reload past jobs so history survives a container restart.

        Anything recorded as running was killed with the previous process, so it
        is reported as interrupted rather than silently resurrected.
        """
        if not self.runs_dir.exists():
            return
        for job_file in sorted(self.runs_dir.glob("*/job.json")):
            try:
                raw = snakeize(json.loads(job_file.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
            known_fields = set(JobFile.__dataclass_fields__)
            files = [
                JobFile(**{k: v for k, v in entry.items() if k in known_fields})
                for entry in raw.get("files", [])
            ]
            kind = raw.get("kind", JOB_KIND_RESTORE)
            train_params_raw = raw.get("train_params")
            train_params: Optional[TrainParams] = None
            if train_params_raw and isinstance(train_params_raw, dict):
                known_tp = set(TrainParams.__dataclass_fields__)
                train_params = TrainParams(**{k: v for k, v in train_params_raw.items() if k in known_tp})
            job = Job(
                id=raw["id"],
                created_at=raw.get("created_at", 0.0),
                steps=raw.get("steps", 50),
                batch_size=raw.get("batch_size", 16),
                files=files,
                kind=kind,
                status=raw.get("status", INTERRUPTED),
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                error=raw.get("error"),
                train_params=train_params,
                train_stage=raw.get("train_stage", ""),
                train_fraction=raw.get("train_fraction"),
                train_eta_sec=raw.get("train_eta_sec"),
            )
            if job.status in (QUEUED, RUNNING):
                job.status = INTERRUPTED
                job.finished_at = job.finished_at or time.time()
                if job.kind == JOB_KIND_RESTORE:
                    for entry in job.files:
                        if entry.status in (QUEUED, RUNNING):
                            entry.status = INTERRUPTED
                            entry.stage = "Interrupted by restart"
                else:
                    job.train_stage = "Interrupted by restart"
            with self._lock:
                self._jobs[job.id] = job
                self._logs[job.id] = deque(self._read_log(job.id), maxlen=LOG_RING_SIZE)

    # -- submission --------------------------------------------------------

    def submit(self, files: Iterable[dict], steps: int, batch_size: int) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_files = [
            JobFile(
                name=entry["name"],
                source_path=entry["source_path"],
                cutoff_hz=int(entry["cutoff_hz"]),
            )
            for entry in files
        ]
        job = Job(
            id=job_id,
            created_at=time.time(),
            steps=int(steps),
            batch_size=int(batch_size),
            files=job_files,
            kind=JOB_KIND_RESTORE,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._logs[job_id] = deque(maxlen=LOG_RING_SIZE)
            self._cancel_events[job_id] = threading.Event()
        self._persist(job, force=True)
        self._broadcast_job(job, force=True)
        self._queue.put(job_id)
        return job

    def submit_training(self, params: TrainParams) -> Job:
        """Enqueue a fine-tuning job."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            created_at=time.time(),
            steps=params.steps,
            batch_size=params.batch_size,
            files=[],
            kind=JOB_KIND_TRAIN,
            train_params=params,
            train_stage="Queued",
        )
        with self._lock:
            self._jobs[job_id] = job
            self._logs[job_id] = deque(maxlen=LOG_RING_SIZE)
            self._cancel_events[job_id] = threading.Event()
        self._persist(job, force=True)
        self._broadcast_job(job, force=True)
        self._queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATES:
                return False
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        self.append_log(job_id, "Cancellation requested.")
        if job.status == QUEUED:
            self._finish_job(job, CANCELLED)
        return True

    # -- reads -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_log(self, job_id: str) -> list[str]:
        with self._lock:
            return list(self._logs.get(job_id, []))

    def active_job_id(self) -> Optional[str]:
        with self._lock:
            for job in self._jobs.values():
                if job.status == RUNNING:
                    return job.id
        return None

    def active_job_kind(self) -> Optional[str]:
        with self._lock:
            for job in self._jobs.values():
                if job.status == RUNNING:
                    return job.kind
        return None

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status == QUEUED)

    def run_dir(self, job_id: str) -> Path:
        return self.runs_dir / job_id

    # -- worker ------------------------------------------------------------

    def _run_worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            job = self.get(job_id)
            if job is None or job.status in TERMINAL_STATES:
                continue
            try:
                if job.kind == JOB_KIND_TRAIN:
                    self._process_training_job(job)
                else:
                    self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - the worker must never die
                self.append_log(job.id, f"Worker error: {exc}")
                job.error = str(exc)
                self._finish_job(job, FAILED)

    def _process_job(self, job: Job) -> None:
        cancel_event = self._cancel_events.setdefault(job.id, threading.Event())
        job.status = RUNNING
        job.started_at = time.time()
        self._broadcast_job(job, force=True)
        self.append_log(job.id, f"Starting job {job.id}: {len(job.files)} file(s), {job.steps} steps, batch {job.batch_size}.")

        run_dir = self.run_dir(job.id)
        run_dir.mkdir(parents=True, exist_ok=True)

        any_failed = False
        for entry in job.files:
            if cancel_event.is_set():
                entry.status = CANCELLED
                entry.stage = "Cancelled"
                continue

            entry.status = RUNNING
            entry.started_at = time.time()
            entry.stage = "Starting"
            entry.fraction = None
            self._broadcast_job(job, force=True)

            def on_progress(progress: FileProgress, _entry=entry) -> None:
                _entry.stage = progress.stage
                _entry.fraction = progress.fraction
                _entry.eta_sec = progress.eta_sec
                self._broadcast_job(job)

            def on_log(line: str) -> None:
                self.append_log(job.id, line)

            try:
                result: FileResult = restore_file(
                    source_path=entry.source_path,
                    run_dir=run_dir / Path(entry.name).stem.replace(" ", "_"),
                    steps=job.steps,
                    cutoff_hz=entry.cutoff_hz,
                    batch_size=job.batch_size,
                    on_progress=on_progress,
                    on_log=on_log,
                    cancel_event=cancel_event,
                )
            except InferenceCancelled:
                entry.status = CANCELLED
                entry.stage = "Cancelled"
                entry.finished_at = time.time()
                self.append_log(job.id, f"{entry.name}: cancelled.")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
                detail = exc.detail if isinstance(exc, PipelineError) else repr(exc)
                message = str(exc) or exc.__class__.__name__
                any_failed = True
                entry.status = FAILED
                entry.stage = "Failed"
                entry.error = message
                entry.error_detail = detail or None
                entry.finished_at = time.time()
                self.append_log(job.id, f"{entry.name}: FAILED - {message}")
                for line in (detail or "").splitlines():
                    self.append_log(job.id, line)
                self._broadcast_job(job, force=True)
                continue

            entry.status = COMPLETED
            entry.stage = "Done"
            entry.fraction = 1.0
            entry.eta_sec = None
            entry.finished_at = time.time()
            # Stored snake_case like every other field, because `Job.to_dict`
            # camelises on the way out and `_restore_from_disk` snakeises on the
            # way in. Storing the camelCase form here would leave a fresh job
            # and a reloaded one holding different keys for the same result.
            entry.result = asdict(result)
            self.append_log(
                job.id,
                f"{entry.name}: energy >={entry.cutoff_hz} Hz went "
                f"{result.high_band_in_db:.1f} dB -> {result.high_band_out_db:.1f} dB "
                f"({result.high_band_delta_db:+.1f} dB) in {result.elapsed_sec:.0f}s.",
            )
            self._broadcast_job(job, force=True)

        if cancel_event.is_set():
            self._finish_job(job, CANCELLED)
        elif any_failed:
            # Files that did succeed keep their results; the job status reflects
            # that something needs attention rather than quietly reading "done".
            failed = [entry.name for entry in job.files if entry.status == FAILED]
            job.error = (
                f"{len(failed)} of {len(job.files)} files failed: {', '.join(failed)}"
            )
            self._finish_job(job, FAILED)
        else:
            self._finish_job(job, COMPLETED)

    def _process_training_job(self, job: Job) -> None:
        cancel_event = self._cancel_events.setdefault(job.id, threading.Event())
        params = job.train_params
        assert params is not None

        job.status = RUNNING
        job.started_at = time.time()
        job.train_stage = "Starting"
        job.train_fraction = 0.0
        self._broadcast_job(job, force=True)
        self.append_log(
            job.id,
            f"Starting training job {job.id}: {params.steps} steps, "
            f"batch {params.batch_size}, splits={params.splits}.",
        )

        def on_log(line: str) -> None:
            self.append_log(job.id, line)

        def on_progress(progress: TrainingProgress) -> None:
            job.train_stage = progress.stage
            job.train_fraction = progress.fraction
            job.train_eta_sec = progress.eta_sec
            self._broadcast_job(job)

        try:
            run_finetune(
                data_dir=params.data_dir,
                output_dir=params.output_dir,
                steps=params.steps,
                batch_size=params.batch_size,
                learning_rate=params.learning_rate,
                splits=params.splits,
                val_frac=params.val_frac,
                val_every=params.val_every,
                val_samples=params.val_samples,
                restart=params.restart,
                on_log=on_log,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
        except TrainingCancelled:
            job.train_stage = "Cancelled"
            self._finish_job(job, CANCELLED)
            return
        except Exception as exc:  # noqa: BLE001
            detail = exc.detail if isinstance(exc, TrainingError) else repr(exc)
            job.error = str(exc) or exc.__class__.__name__
            job.train_stage = "Failed"
            self.append_log(job.id, f"Training FAILED: {job.error}")
            if detail:
                for line in detail.splitlines():
                    self.append_log(job.id, line)
            self._finish_job(job, FAILED)
            return

        job.train_fraction = 1.0
        job.train_stage = "Done"
        self._finish_job(job, COMPLETED)

    def _finish_job(self, job: Job, status: str) -> None:
        job.status = status
        job.finished_at = time.time()
        if job.kind == JOB_KIND_RESTORE:
            for entry in job.files:
                if entry.status in (QUEUED, RUNNING):
                    entry.status = CANCELLED if status == CANCELLED else status
                    entry.stage = "Cancelled" if status == CANCELLED else entry.stage
        self.append_log(job.id, f"Job {job.id} {status}.")
        self._persist(job, force=True)
        self._broadcast_job(job, force=True)

    # -- logging and broadcast --------------------------------------------

    def append_log(self, job_id: str, line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')}  {line}"
        with self._lock:
            ring = self._logs.setdefault(job_id, deque(maxlen=LOG_RING_SIZE))
            ring.append(stamped)
        self._append_log_file(job_id, stamped)
        self.broker.publish({"type": "log", "jobId": job_id, "line": stamped})

    def _log_path(self, job_id: str) -> Path:
        return self.run_dir(job_id) / "job.log"

    def _append_log_file(self, job_id: str, line: str) -> None:
        path = self._log_path(job_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def _read_log(self, job_id: str) -> list[str]:
        try:
            return self._log_path(job_id).read_text(encoding="utf-8").splitlines()[-LOG_RING_SIZE:]
        except OSError:
            return []

    def _broadcast_job(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_broadcast) < BROADCAST_INTERVAL_SEC:
            return
        self._last_broadcast = now
        self.broker.publish({"type": "job", "job": job.to_dict()})
        self._persist(job, force=force)

    def _persist(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_persist.get(job.id, 0.0)) < PERSIST_INTERVAL_SEC:
            return
        self._last_persist[job.id] = now
        path = self.run_dir(job.id) / "job.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job.to_dict(), indent=2))
            tmp.replace(path)
        except OSError:
            pass


store = JobStore()
