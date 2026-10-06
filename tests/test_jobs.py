"""Queue behaviour, driven with a stubbed pipeline so no GPU is involved."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from server import jobs as jobs_module
from server.inference import InferenceCancelled
from server.jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    JobStore,
)
from server.pipeline import FileProgress, FileResult, PipelineError


def _result(name: str, cutoff_hz: int) -> FileResult:
    return FileResult(
        name=name,
        source_path=f"/inputs/{name}",
        restored_path=f"/outputs/{name}",
        filtered_path=f"/outputs/filtered_{name}",
        channels=2,
        duration_sec=4.0,
        cutoff_hz=cutoff_hz,
        steps=10,
        batch_size=4,
        high_band_in_db=-40.0,
        high_band_out_db=-12.0,
        high_band_delta_db=28.0,
        elapsed_sec=1.0,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store with its own runs directory and a stubbed `restore_file`."""
    created = JobStore(runs_dir=tmp_path / "runs")
    created.start()
    try:
        yield created
    finally:
        created.shutdown()


def _stub_pipeline(monkeypatch, behaviour):
    monkeypatch.setattr(jobs_module, "restore_file", behaviour)


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


FILES = [{"name": "a.wav", "source_path": "/inputs/a.wav", "cutoff_hz": 12000}]


def test_a_successful_job_records_the_measurement(store, monkeypatch):
    def behaviour(*, source_path, cutoff_hz, on_progress, **_kwargs):
        on_progress(FileProgress(stage="Left: diffusion", fraction=0.5, eta_sec=8))
        return _result("a.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: job.status == COMPLETED), job.status
    entry = job.files[0]
    assert entry.status == COMPLETED
    assert entry.fraction == 1.0
    assert entry.result["high_band_delta_db"] == 28.0
    assert job.overall_progress() == 1.0


def test_progress_updates_reach_the_job_before_it_finishes(store, monkeypatch):
    release = threading.Event()
    seen: list[float] = []

    def behaviour(*, cutoff_hz, on_progress, **_kwargs):
        on_progress(FileProgress(stage="Left: diffusion", fraction=0.25, eta_sec=30))
        seen.append(0.25)
        release.wait(timeout=5)
        return _result("a.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: job.files[0].fraction == 0.25)
    assert job.files[0].stage == "Left: diffusion"
    assert job.files[0].eta_sec == 30
    assert store.active_job_id() == job.id
    release.set()
    assert _wait_for(lambda: job.status == COMPLETED)


def test_a_failed_file_is_reported_with_its_detail(store, monkeypatch):
    def behaviour(**_kwargs):
        raise PipelineError("Could not decode a.wav.", "ffmpeg returned error code: 1")

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: job.status == FAILED)
    assert job.files[0].error == "Could not decode a.wav."
    assert "ffmpeg" in job.files[0].error_detail
    # The job-level message exists so the UI has something to show above the
    # per-file rows.
    assert "a.wav" in job.error


def test_an_unexpected_exception_still_produces_a_message(store, monkeypatch):
    def behaviour(**_kwargs):
        raise RuntimeError()

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: job.status == FAILED)
    assert job.files[0].error == "RuntimeError"


def test_one_bad_file_does_not_stop_the_rest_of_the_batch(store, monkeypatch):
    def behaviour(*, source_path, cutoff_hz, **_kwargs):
        if source_path.endswith("bad.wav"):
            raise PipelineError("Could not decode bad.wav.")
        return _result("good.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(
        [
            {"name": "bad.wav", "source_path": "/inputs/bad.wav", "cutoff_hz": 12000},
            {"name": "good.wav", "source_path": "/inputs/good.wav", "cutoff_hz": 12000},
        ],
        steps=10,
        batch_size=4,
    )

    assert _wait_for(lambda: job.status == FAILED)
    assert job.files[0].status == FAILED
    assert job.files[1].status == COMPLETED
    assert job.files[1].result is not None


def test_cancelling_a_running_job_signals_the_pipeline(store, monkeypatch):
    entered = threading.Event()

    def behaviour(*, cancel_event, **_kwargs):
        entered.set()
        if cancel_event.wait(timeout=10):
            raise InferenceCancelled()
        return _result("a.wav", 12000)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(FILES, steps=10, batch_size=4)

    assert entered.wait(timeout=5)
    assert store.cancel(job.id) is True
    assert _wait_for(lambda: job.status == CANCELLED)
    assert job.files[0].status == CANCELLED


def test_cancelling_a_queued_job_never_starts_it(store, monkeypatch):
    release = threading.Event()
    started: list[str] = []

    def behaviour(*, source_path, cutoff_hz, **_kwargs):
        started.append(source_path)
        release.wait(timeout=5)
        return _result("a.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    first = store.submit(FILES, steps=10, batch_size=4)
    second = store.submit(
        [{"name": "b.wav", "source_path": "/inputs/b.wav", "cutoff_hz": 12000}],
        steps=10,
        batch_size=4,
    )

    assert _wait_for(lambda: first.status == RUNNING)
    assert second.status == QUEUED
    assert store.cancel(second.id) is True
    assert second.status == CANCELLED

    release.set()
    assert _wait_for(lambda: first.status == COMPLETED)
    assert "/inputs/b.wav" not in started


def test_a_finished_job_cannot_be_cancelled(store, monkeypatch):
    _stub_pipeline(monkeypatch, lambda *, cutoff_hz, **_k: _result("a.wav", cutoff_hz))
    job = store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: job.status == COMPLETED)
    assert store.cancel(job.id) is False


def test_only_one_job_runs_at_a_time(store, monkeypatch):
    """There is one GPU; overlapping runs would fight over it."""
    concurrent = 0
    peak = 0
    guard = threading.Lock()

    def behaviour(*, cutoff_hz, **_kwargs):
        nonlocal concurrent, peak
        with guard:
            concurrent += 1
            peak = max(peak, concurrent)
        time.sleep(0.2)
        with guard:
            concurrent -= 1
        return _result("a.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    submitted = [store.submit(FILES, steps=10, batch_size=4) for _ in range(3)]

    assert _wait_for(lambda: all(j.status == COMPLETED for j in submitted), timeout=15)
    assert peak == 1


def test_history_and_logs_survive_a_restart(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    first = JobStore(runs_dir=runs)
    first.start()
    _stub_pipeline(monkeypatch, lambda *, cutoff_hz, **_k: _result("a.wav", cutoff_hz))
    job = first.submit(FILES, steps=10, batch_size=4)
    assert _wait_for(lambda: job.status == COMPLETED)
    first.shutdown()

    second = JobStore(runs_dir=runs)
    second.start()
    try:
        restored = second.get(job.id)
        assert restored is not None
        assert restored.status == COMPLETED
        assert restored.files[0].result["high_band_delta_db"] == 28.0
        assert any("Starting job" in line for line in second.get_log(job.id))
    finally:
        second.shutdown()


def test_a_job_killed_mid_run_comes_back_as_interrupted(tmp_path, monkeypatch):
    """A hard kill leaves `running` on disk; resurrecting it would be a lie."""
    runs = tmp_path / "runs"
    release = threading.Event()
    first = JobStore(runs_dir=runs)
    first.start()
    _stub_pipeline(monkeypatch, lambda **_k: (release.wait(timeout=10), _result("a.wav", 12000))[1])
    job = first.submit(FILES, steps=10, batch_size=4)
    assert _wait_for(lambda: job.status == RUNNING)
    # No shutdown(): simulate the process disappearing while the job ran.

    second = JobStore(runs_dir=runs)
    second.start()
    try:
        restored = second.get(job.id)
        assert restored.status == INTERRUPTED
        assert restored.files[0].status == INTERRUPTED
        assert restored.files[0].stage == "Interrupted by restart"
    finally:
        second.shutdown()
        release.set()
        first.shutdown()


def test_queue_depth_counts_only_waiting_jobs(store, monkeypatch):
    release = threading.Event()
    _stub_pipeline(
        monkeypatch,
        lambda *, cutoff_hz, **_k: (release.wait(timeout=5), _result("a.wav", cutoff_hz))[1],
    )
    running = store.submit(FILES, steps=10, batch_size=4)
    store.submit(FILES, steps=10, batch_size=4)

    assert _wait_for(lambda: running.status == RUNNING)
    assert store.queue_depth() == 1

    release.set()
    assert _wait_for(lambda: store.queue_depth() == 0, timeout=15)


def test_job_serialises_to_camel_case(store, monkeypatch):
    _stub_pipeline(monkeypatch, lambda *, cutoff_hz, **_k: _result("a.wav", cutoff_hz))
    job = store.submit(FILES, steps=10, batch_size=4)
    assert _wait_for(lambda: job.status == COMPLETED)

    payload = job.to_dict()
    assert payload["batchSize"] == 4
    assert payload["files"][0]["cutoffHz"] == 12000
    assert payload["progress"] == 1.0
    # The nested result is stored snake_case, so this is where a missed
    # conversion would leak through to the browser.
    assert payload["files"][0]["result"]["highBandDeltaDb"] == 28.0
    assert not any("_" in key for key in payload["files"][0]["result"])


def test_equal_basenames_write_distinct_run_directories(store, monkeypatch):
    seen: list[str] = []

    def behaviour(*, source_path, run_dir, cutoff_hz, **_kwargs):
        seen.append(str(run_dir))
        return _result("same.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(
        [
            {"name": "same.wav", "source_path": "/inputs/album_a/same.wav", "cutoff_hz": 12000},
            {"name": "same.wav", "source_path": "/inputs/album_b/same.wav", "cutoff_hz": 12000},
        ],
        steps=10,
        batch_size=4,
    )

    assert _wait_for(lambda: job.status == COMPLETED)
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert job.files[0].id != job.files[1].id
    assert job.files[0].name == "same.wav"
    assert Path(seen[0]).name == job.files[0].id
    assert Path(seen[1]).name == job.files[1].id


def test_stems_that_normalize_identically_write_distinct_run_directories(store, monkeypatch):
    seen: list[str] = []

    def behaviour(*, run_dir, cutoff_hz, **_kwargs):
        seen.append(str(run_dir))
        return _result("x.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(
        [
            {"name": "foo bar.wav", "source_path": "/inputs/foo bar.wav", "cutoff_hz": 12000},
            {"name": "foo_bar.wav", "source_path": "/inputs/foo_bar.wav", "cutoff_hz": 12000},
            {"name": "same.wav", "source_path": "/inputs/same.wav", "cutoff_hz": 12000},
            {"name": "same.flac", "source_path": "/inputs/same.flac", "cutoff_hz": 12000},
        ],
        steps=10,
        batch_size=4,
    )

    assert _wait_for(lambda: job.status == COMPLETED)
    assert len(set(seen)) == 4
    assert [entry.name for entry in job.files] == [
        "foo bar.wav",
        "foo_bar.wav",
        "same.wav",
        "same.flac",
    ]


def test_colliding_display_names_get_distinct_output_directories(store, monkeypatch, tmp_path):
    """Equal basenames, equal stems, and underscore-normalized names must not share a folder."""
    seen: list = []

    def behaviour(*, source_path, run_dir, cutoff_hz, **_kwargs):
        seen.append(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "owned.txt").write_text(source_path)
        return _result(source_path.rsplit("/", 1)[-1], cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(
        [
            {"name": "same.wav", "source_path": "/inputs/album1/same.wav", "cutoff_hz": 12000},
            {"name": "same.wav", "source_path": "/inputs/album2/same.wav", "cutoff_hz": 12000},
            {"name": "same.flac", "source_path": "/inputs/same.flac", "cutoff_hz": 12000},
            {"name": "foo bar.wav", "source_path": "/inputs/foo bar.wav", "cutoff_hz": 12000},
            {"name": "foo_bar.wav", "source_path": "/inputs/foo_bar.wav", "cutoff_hz": 12000},
        ],
        steps=10,
        batch_size=4,
    )

    assert _wait_for(lambda: job.status == COMPLETED), job.status
    assert len({str(path) for path in seen}) == 5
    assert len({path.name for path in seen}) == 5
    owners = [path.joinpath("owned.txt").read_text() for path in seen]
    assert len(set(owners)) == 5
    assert [entry.name for entry in job.files] == [
        "same.wav",
        "same.wav",
        "same.flac",
        "foo bar.wav",
        "foo_bar.wav",
    ]
    assert all(entry.id for entry in job.files)
    assert len({entry.id for entry in job.files}) == 5


def test_display_name_is_basename_only(store, monkeypatch):
    def behaviour(*, cutoff_hz, **_kwargs):
        return _result("safe.wav", cutoff_hz)

    _stub_pipeline(monkeypatch, behaviour)
    job = store.submit(
        [{"name": "../etc/passwd.wav", "source_path": "/inputs/real.wav", "cutoff_hz": 12000}],
        steps=10,
        batch_size=4,
    )
    assert job.files[0].name == "passwd.wav"
    assert _wait_for(lambda: job.status == COMPLETED)


def _train_params(tmp_path: Path, **overrides) -> TrainParams:
    values = dict(
        data_dir=str(tmp_path),
        output_dir="unused",
        steps=10,
        batch_size=2,
        learning_rate=5e-5,
        splits="both",
        val_frac=0.1,
        val_every=None,
        val_samples=None,
        restart=False,
    )
    values.update(overrides)
    return TrainParams(**values)


def test_training_jobs_get_isolated_output_directories(store, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "TRAINING_OUTPUT_DIR", tmp_path / "training")
    monkeypatch.setattr(jobs_module, "run_finetune", lambda **_kwargs: None)
    first = store.submit_training(_train_params(tmp_path / "data_a"))
    second = store.submit_training(_train_params(tmp_path / "data_b"))
    assert first.train_params is not None and second.train_params is not None
    assert first.train_params.output_dir != second.train_params.output_dir
    assert first.train_params.output_dir.endswith(f"runs/{first.id}")
    assert second.train_params.output_dir.endswith(f"runs/{second.id}")
    assert Path(first.train_params.output_dir).is_dir()
    assert Path(second.train_params.output_dir).is_dir()
    assert _wait_for(lambda: first.status == COMPLETED)
    assert _wait_for(lambda: second.status == COMPLETED)


def test_training_resume_reuses_the_selected_run_directory(store, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "TRAINING_OUTPUT_DIR", tmp_path / "training")
    monkeypatch.setattr(jobs_module, "run_finetune", lambda **_kwargs: None)
    first = store.submit_training(_train_params(tmp_path / "data_a"))
    assert _wait_for(lambda: first.status == COMPLETED)
    resumed = store.submit_training(_train_params(tmp_path / "data_a", resume_from=first.id, steps=20))
    assert resumed.train_params is not None and first.train_params is not None
    assert resumed.train_params.output_dir == first.train_params.output_dir
    assert resumed.train_params.steps == 20
    assert _wait_for(lambda: resumed.status == COMPLETED)


def test_training_resume_is_opt_in(store, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "TRAINING_OUTPUT_DIR", tmp_path / "training")
    monkeypatch.setattr(jobs_module, "run_finetune", lambda **_kwargs: None)
    first = store.submit_training(_train_params(tmp_path / "data_a"))
    assert _wait_for(lambda: first.status == COMPLETED)
    second = store.submit_training(_train_params(tmp_path / "data_b"))
    assert second.train_params is not None and first.train_params is not None
    assert second.train_params.output_dir != first.train_params.output_dir
    assert _wait_for(lambda: second.status == COMPLETED)


def test_unknown_training_resume_target_is_rejected(store, tmp_path):
    with pytest.raises(ValueError, match="Cannot resume"):
        store.submit_training(_train_params(tmp_path, resume_from="missing-run"))


def test_restore_job_pins_one_ensemble_snapshot_for_every_file(store, monkeypatch, tmp_path):
    seen: list[str | None] = []

    def fake_snapshot(dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("pinned: true\n", encoding="utf-8")
        return {
            "active": "release",
            "config_path": str(dest),
            "checkpoints": ["split1.ckpt", "split2.ckpt"],
        }

    def fake_restore(*, ensemble_config, cutoff_hz, **_kwargs):
        seen.append(ensemble_config)
        return _result("a.wav", cutoff_hz)

    monkeypatch.setattr(jobs_module, "snapshot_ensemble_config", fake_snapshot)
    monkeypatch.setattr(jobs_module, "restore_file", fake_restore)

    job = store.submit(
        [
            {"name": "a.wav", "source_path": "/inputs/a.wav", "cutoff_hz": 12000},
            {"name": "b.wav", "source_path": "/inputs/b.wav", "cutoff_hz": 11000},
        ],
        steps=10,
        batch_size=4,
    )
    assert _wait_for(lambda: job.status == COMPLETED)
    assert job.model_identity is not None
    assert job.model_identity["active"] == "release"
    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert seen[0] is not None and seen[0].endswith("ensemble.yaml")


def test_malformed_history_does_not_block_startup(tmp_path):
    runs = tmp_path / "runs"
    (runs / "empty").mkdir(parents=True)
    (runs / "empty" / "job.json").write_text("{}", encoding="utf-8")
    (runs / "legacy").mkdir()
    (runs / "legacy" / "job.json").write_text(
        json.dumps({"id": 123, "status": "completed"}),
        encoding="utf-8",
    )
    (runs / "good").mkdir()
    (runs / "good" / "job.json").write_text(
        json.dumps(
            {
                "id": "abc123def456",
                "created_at": 1.0,
                "steps": 10,
                "batch_size": 4,
                "files": [{"id": "f1", "name": "a.wav", "source_path": "/in/a.wav", "cutoff_hz": 12000}],
                "kind": "restore",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    created = JobStore(runs_dir=runs)
    created.start()
    try:
        ids = {job.id for job in created.list_jobs()}
        assert ids == {"abc123def456"}
        assert not (runs / "empty" / "job.json").exists()
        assert (runs / "empty" / "job.json.invalid").is_file()
        assert not (runs / "legacy" / "job.json").exists()
        assert (runs / "legacy" / "job.json.invalid").is_file()
        assert (runs / "good" / "job.json").is_file()
    finally:
        created.shutdown()
