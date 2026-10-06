"""HTTP contract, including the path confinement the media routes rely on."""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.requests import Request

from server import jobs as jobs_module
from server.config import CUTOFF_MAX_HZ, STEPS_MAX
from server.jobs import EventBroker, enqueue_event
from server.main import create_app
from server.pipeline import FileResult

from .conftest import SAMPLE_RATE, brickwalled, write_wav


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as created:
        yield created


def test_config_exposes_the_bounds_the_ui_renders(client):
    body = client.get("/api/config").json()

    assert body["cutoffHz"]["min"] < body["cutoffHz"]["default"] < body["cutoffHz"]["max"]
    assert body["steps"]["max"] == STEPS_MAX
    assert ".wav" in body["audioExtensions"]
    assert body["inputDir"] and body["outputDir"]


def test_health_reports_the_active_job(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert "activeJobId" in body


# -- staging ---------------------------------------------------------------


def test_browsing_a_directory_lists_the_audio_inside_it(client, input_dir, transcode_wav):
    """Pasting a folder is the obvious thing to do, so it has to work."""
    nested = write_wav(input_dir / "nested" / "deep.wav", brickwalled(12000))

    entries = client.get("/api/browse", params={"pattern": str(input_dir)}).json()["entries"]

    paths = {entry["path"] for entry in entries}
    assert str(transcode_wav) in paths
    assert str(nested) in paths
    assert all(entry["sizeBytes"] > 0 for entry in entries)


def test_browsing_skips_non_audio_and_hidden_scratch(client, input_dir):
    (input_dir / "notes.txt").write_text("not audio")
    write_wav(input_dir / ".work" / "scratch.wav", brickwalled(12000))

    paths = {
        entry["path"]
        for entry in client.get("/api/browse", params={"pattern": str(input_dir)}).json()["entries"]
    }

    assert not any(path.endswith("notes.txt") for path in paths)
    assert not any("/.work/" in path for path in paths)


def test_browsing_accepts_a_glob(client, input_dir, transcode_wav):
    entries = client.get("/api/browse", params={"pattern": f"{input_dir}/transcode.*"}).json()
    assert [entry["name"] for entry in entries["entries"]] == ["transcode.wav"]


def test_browsing_outside_the_staging_tree_returns_nothing(client):
    assert client.get("/api/browse", params={"pattern": "/etc"}).json()["entries"] == []


def test_analyze_reports_per_file_errors_without_failing_the_batch(client, transcode_wav, input_dir):
    broken = input_dir / "broken.wav"
    broken.write_text("not a wav")

    body = client.post(
        "/api/analyze", json={"paths": [str(transcode_wav), str(broken), "/etc/passwd"]}
    ).json()

    assert [entry["name"] for entry in body["files"]] == ["transcode.wav"]
    assert {entry["name"] for entry in body["errors"]} == {"broken.wav", "passwd"}
    assert all(entry["error"] for entry in body["errors"])


def test_upload_stages_and_measures_the_file(client):
    buffer = io.BytesIO()
    sf.write(buffer, brickwalled(11000), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    body = client.post(
        "/api/uploads", files={"files": ("upload.wav", buffer, "audio/wav")}
    ).json()

    assert body["errors"] == []
    assert body["files"][0]["verdict"] == "transcode"
    assert body["files"][0]["suggestedCutoffHz"] > 0


def test_upload_rejects_a_non_audio_extension_with_a_readable_reason(client):
    body = client.post(
        "/api/uploads", files={"files": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    ).json()

    assert body["files"] == []
    assert "Not an audio file" in body["errors"][0]["error"]


def test_upload_of_an_undecodable_file_never_reports_a_blank_error(client):
    body = client.post(
        "/api/uploads", files={"files": ("fake.wav", io.BytesIO(b"nope"), "audio/wav")}
    ).json()

    assert body["files"] == []
    assert body["errors"][0]["error"].strip()


def _wav_bytes(cutoff_hz: float) -> io.BytesIO:
    buffer = io.BytesIO()
    sf.write(buffer, brickwalled(cutoff_hz), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


def test_upload_keeps_duplicate_filenames_as_distinct_files(client):
    """Two parts named same.wav must not share a destination path."""
    body = client.post(
        "/api/uploads",
        files=[
            ("files", ("same.wav", _wav_bytes(8000), "audio/wav")),
            ("files", ("same.wav", _wav_bytes(16000), "audio/wav")),
        ],
    ).json()

    assert body["errors"] == []
    assert len(body["files"]) == 2
    paths = [entry["path"] for entry in body["files"]]
    assert paths[0] != paths[1]
    assert all(Path(path).is_file() for path in paths)
    assert Path(paths[0]).read_bytes() != Path(paths[1]).read_bytes()
    assert [entry["name"] for entry in body["files"]] == ["same.wav", "same.wav"]
    # The original filename is a display label, not the on-disk identity.
    assert Path(paths[0]).name != "same.wav" or Path(paths[0]).parent != Path(paths[1]).parent


# -- jobs ------------------------------------------------------------------


def test_submitting_a_job_returns_it_queued(client, transcode_wav, monkeypatch):
    monkeypatch.setattr(
        jobs_module,
        "restore_file",
        lambda *, cutoff_hz, **_k: FileResult(
            name="transcode.wav",
            source_path=str(transcode_wav),
            restored_path=str(transcode_wav),
            filtered_path=str(transcode_wav),
            channels=2,
            duration_sec=4.0,
            cutoff_hz=cutoff_hz,
            steps=10,
            batch_size=4,
            high_band_in_db=-40.0,
            high_band_out_db=-10.0,
            high_band_delta_db=30.0,
            elapsed_sec=1.0,
        ),
    )

    response = client.post(
        "/api/jobs",
        json={
            "files": [
                {"name": "transcode.wav", "sourcePath": str(transcode_wav), "cutoffHz": 11000}
            ],
            "steps": 10,
            "batchSize": 4,
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in {"queued", "running"}
    assert body["files"][0]["cutoffHz"] == 11000
    assert client.get(f"/api/jobs/{body['id']}").status_code == 200


def test_an_empty_job_is_rejected(client):
    assert client.post("/api/jobs", json={"files": []}).status_code == 400


def test_an_out_of_range_cutoff_is_rejected(client, transcode_wav):
    response = client.post(
        "/api/jobs",
        json={
            "files": [
                {
                    "name": "transcode.wav",
                    "sourcePath": str(transcode_wav),
                    "cutoffHz": CUTOFF_MAX_HZ + 5000,
                }
            ]
        },
    )
    assert response.status_code == 422


def test_a_job_referencing_a_file_outside_the_tree_is_refused(client):
    response = client.post(
        "/api/jobs",
        json={"files": [{"name": "passwd", "sourcePath": "/etc/passwd", "cutoffHz": 12000}]},
    )
    assert response.status_code == 403


def test_unknown_jobs_are_404_and_uncancellable(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 409


# -- media -----------------------------------------------------------------


@pytest.mark.parametrize("route", ["/api/audio", "/api/download", "/api/spectrogram", "/api/waveform"])
def test_media_routes_serve_files_inside_the_tree(client, transcode_wav, route):
    assert client.get(route, params={"path": str(transcode_wav)}).status_code == 200


@pytest.mark.parametrize("route", ["/api/audio", "/api/download", "/api/spectrogram", "/api/waveform"])
def test_media_routes_refuse_paths_outside_the_tree(client, route):
    """These take a caller-supplied path, so traversal is the obvious attack."""
    assert client.get(route, params={"path": "/etc/passwd"}).status_code == 403
    assert client.get(route, params={"path": "/tmp/../etc/passwd"}).status_code == 403


def test_media_routes_report_a_missing_file_distinctly(client, input_dir):
    response = client.get("/api/audio", params={"path": str(input_dir / "absent.wav")})
    assert response.status_code == 404


def test_spectrogram_returns_a_grid_the_browser_can_draw(client, transcode_wav):
    body = client.get("/api/spectrogram", params={"path": str(transcode_wav)}).json()

    assert body["width"] > 0 and body["height"] > 0
    assert body["maxFrequencyHz"] == SAMPLE_RATE / 2
    assert isinstance(body["data"], str)


def test_the_event_stream_opens_with_a_snapshot(client, monkeypatch):
    """A reconnecting tab must be consistent without a second round trip.

    The handler streams until the client goes away, which never happens under
    TestClient, so disconnect is forced after the snapshot is written.
    """

    async def already_disconnected(_self) -> bool:
        return True

    monkeypatch.setattr(Request, "is_disconnected", already_disconnected)

    with client.stream("GET", "/api/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        payload = json.loads(response.read().decode().split("data: ", 1)[1])

    assert payload["type"] == "snapshot"
    assert isinstance(payload["jobs"], list)


def test_the_broker_fans_events_out_to_every_subscriber():
    """Events cross from the worker thread into each request's event loop."""

    async def collect() -> list[dict]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        first: asyncio.Queue = asyncio.Queue()
        second: asyncio.Queue = asyncio.Queue()
        broker.subscribe(loop, first)
        broker.subscribe(loop, second)

        await asyncio.to_thread(broker.publish, {"type": "log", "line": "hello"})
        received = [await asyncio.wait_for(q.get(), timeout=2) for q in (first, second)]

        broker.unsubscribe(loop, first)
        await asyncio.to_thread(broker.publish, {"type": "log", "line": "second"})
        assert await asyncio.wait_for(second.get(), timeout=2) == {
            "type": "log",
            "line": "second",
        }
        assert first.empty()
        return received

    assert asyncio.run(collect()) == [
        {"type": "log", "line": "hello"},
        {"type": "log", "line": "hello"},
    ]


def test_enqueue_keeps_terminal_job_when_the_queue_is_full():
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    enqueue_event(queue, {"type": "log", "line": "1"})
    enqueue_event(queue, {"type": "log", "line": "2"})
    enqueue_event(queue, {"type": "log", "line": "3"})
    enqueue_event(queue, {"type": "job", "status": "completed"})
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert any(item.get("type") == "job" for item in events)
    logs = [item for item in events if item.get("type") == "log"]
    assert logs
    assert logs[-1]["line"] == "3"


def test_training_preflight_keeps_structured_problems(client, input_dir):
    """The GUI must receive `detail.problems`, not a flattened status string."""
    response = client.post(
        "/api/training/jobs",
        json={"dataDir": str(input_dir), "steps": 100, "batchSize": 2, "splits": "both"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert isinstance(detail["problems"], list)
    assert detail["problems"]
