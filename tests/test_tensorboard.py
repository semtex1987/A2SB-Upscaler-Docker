"""The TensorBoard child process, the proxy in front of it, and logger wiring.

No test here starts a real TensorBoard: it takes 15-30s to bind, which would
dominate the suite. The spawn is stubbed and the command line asserted instead.
"""
from __future__ import annotations

import importlib.util
import io
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from server.main import create_app
from server.tensorboard import TensorBoardManager, find_event_files

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeProcess:
    """Stand-in for a TensorBoard child. `returncode=None` means still running."""

    def __init__(self, output: bytes = b"", returncode: int | None = None) -> None:
        self.pid = 4242
        self.stdout = io.BytesIO(output)
        self._returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode or 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


@pytest.fixture
def stub_spawn(monkeypatch):
    """Replace Popen and shorten the readiness window."""
    captured: dict = {}

    def install(process: FakeProcess, timeout: float = 0.4):
        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return process

        monkeypatch.setattr("server.tensorboard.subprocess.Popen", fake_popen)
        monkeypatch.setattr("server.tensorboard.TENSORBOARD_STARTUP_TIMEOUT_SEC", timeout)
        return captured

    return install


# ---------------------------------------------------------------------------
# Event file discovery
# ---------------------------------------------------------------------------

def test_event_files_are_found_at_any_depth(tmp_path):
    nested = tmp_path / "split_0.0_0.5" / "tensorboard" / "version_0"
    nested.mkdir(parents=True)
    (nested / "events.out.tfevents.1700000000.host.1.0").touch()
    (tmp_path / "split_0.0_0.5" / "lightning_logs" / "version_0").mkdir(parents=True)
    (tmp_path / "split_0.0_0.5" / "lightning_logs" / "version_0" / "metrics.csv").touch()

    found = find_event_files(tmp_path)

    assert [p.name for p in found] == ["events.out.tfevents.1700000000.host.1.0"]


def test_a_missing_log_directory_is_not_an_error(tmp_path):
    assert find_event_files(tmp_path / "never-created") == []


def test_status_reports_whether_any_event_files_exist(tmp_path):
    manager = TensorBoardManager(log_dir=tmp_path)
    assert manager.status().has_event_files is False

    events = tmp_path / "split_0.0_0.5" / "tensorboard" / "version_0"
    events.mkdir(parents=True)
    (events / "events.out.tfevents.1.host.1.0").touch()

    assert manager.status().has_event_files is True


# ---------------------------------------------------------------------------
# Manager lifecycle
# ---------------------------------------------------------------------------

def test_status_before_anything_starts(tmp_path):
    status = TensorBoardManager(log_dir=tmp_path).status()

    assert status.running is False
    assert status.ready is False
    assert status.pid is None


def test_the_advertised_url_keeps_its_trailing_slash(tmp_path):
    # Without it TensorBoard resolves its assets one level too high and 404s.
    assert TensorBoardManager(log_dir=tmp_path).status().url.endswith("/")


def test_the_upstream_target_carries_the_path_prefix(tmp_path):
    manager = TensorBoardManager(log_dir=tmp_path, port=6099, path_prefix="/tb")
    assert manager.upstream_base == "http://127.0.0.1:6099/tb"


def test_a_missing_package_gives_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("server.tensorboard.tensorboard_available", lambda: False)
    manager = TensorBoardManager(log_dir=tmp_path)

    status = manager.ensure_running()

    assert status.running is False
    assert "pip install tensorboard" in (status.error or "")


def test_the_spawn_command_pins_logdir_port_and_prefix(tmp_path, stub_spawn):
    captured = stub_spawn(FakeProcess())
    manager = TensorBoardManager(log_dir=tmp_path, port=6099, path_prefix="/tb")

    manager.ensure_running()
    command = captured["command"]

    assert "--logdir" in command
    assert str(tmp_path) in command
    assert "6099" in command
    # A prefix is what lets the app proxy TensorBoard on its own single port.
    assert "--path_prefix=/tb" in command


def test_starting_does_not_block_until_ready(tmp_path, stub_spawn):
    # Readiness takes tens of seconds in reality; the call must return anyway so
    # the request does not outlive a deployment's ingress timeout.
    stub_spawn(FakeProcess(), timeout=30.0)
    manager = TensorBoardManager(log_dir=tmp_path, port=6099)

    started = time.monotonic()
    status = manager.ensure_running()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert status.running is True
    assert status.ready is False
    manager.stop()


def test_a_second_start_does_not_spawn_a_duplicate(tmp_path, stub_spawn):
    captured = stub_spawn(FakeProcess(), timeout=30.0)
    manager = TensorBoardManager(log_dir=tmp_path, port=6099)

    manager.ensure_running()
    first = captured["command"]
    captured.clear()
    manager.ensure_running()

    assert "command" not in captured, "already-running manager spawned a second child"
    assert first is not None
    manager.stop()


def test_a_child_that_exits_surfaces_its_own_output(tmp_path, stub_spawn):
    stub_spawn(FakeProcess(output=b"ERROR: Port 6099 is already in use\n", returncode=1))
    manager = TensorBoardManager(log_dir=tmp_path, port=6099)

    manager.ensure_running()
    assert manager.wait_until_ready(timeout=3.0) is False
    status = manager.status()

    assert status.running is False
    assert "exited during startup" in (status.error or "")
    # The reason has to reach the user, not just the container log.
    assert "already in use" in (status.error or "")


def test_readiness_times_out_rather_than_hanging(tmp_path, stub_spawn):
    # Alive but never binding: the poll must give up and say so.
    stub_spawn(FakeProcess(), timeout=0.5)
    manager = TensorBoardManager(log_dir=tmp_path, port=6099)

    manager.ensure_running()
    assert manager.wait_until_ready(timeout=4.0) is False

    assert "did not become ready" in (manager.status().error or "")
    manager.stop()


def test_stopping_is_safe_when_nothing_is_running(tmp_path):
    manager = TensorBoardManager(log_dir=tmp_path)
    manager.stop()
    manager.stop()
    assert manager.status().running is False


def test_stop_clears_the_running_state(tmp_path, stub_spawn):
    stub_spawn(FakeProcess(), timeout=30.0)
    manager = TensorBoardManager(log_dir=tmp_path, port=6099)
    manager.ensure_running()

    manager.stop()

    status = manager.status()
    assert status.running is False
    assert status.pid is None


# ---------------------------------------------------------------------------
# Proxy route
# ---------------------------------------------------------------------------

def test_the_proxy_explains_itself_when_nothing_is_running():
    with TestClient(create_app()) as client:
        response = client.get("/tensorboard/")

    assert response.status_code == 503
    assert "Train tab" in response.json()["detail"]


def test_the_bare_prefix_redirects_to_a_trailing_slash():
    with TestClient(create_app()) as client:
        response = client.get("/tensorboard", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/tensorboard/"


def test_the_spa_catch_all_does_not_swallow_tensorboard_paths():
    # Route order is load-bearing: registered after the catch-all, every
    # TensorBoard request would come back as index.html with a 200.
    with TestClient(create_app()) as client:
        response = client.get("/tensorboard/data/runs")

    assert response.status_code == 503
    assert "text/html" not in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Logger configuration written for the training subprocess
# ---------------------------------------------------------------------------

def _load_finetune():
    spec = importlib.util.spec_from_file_location(
        "finetune_under_test", REPO_ROOT / "training" / "finetune.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base_config(tmp_path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "mix_dataset_config": {
                        "CURATED": {"root_folder": "/old", "filename": "old.csv"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _loggers_written(base_config: Path, run_dir: Path, **kwargs) -> list[dict]:
    finetune = _load_finetune()
    dest = run_dir / "data_override.yaml"
    finetune.write_run_override(
        base_config, dest, "/data", "manifest.csv", log_dir=run_dir, **kwargs
    )
    return yaml.safe_load(dest.read_text(encoding="utf-8"))["trainer"]["logger"]


def test_the_csv_logger_is_configured_even_when_nothing_asked_for_it(base_config, tmp_path):
    # Lightning's default logger flips to TensorBoard once that package is
    # installed, which would take metrics.csv -- and the in-app loss curve --
    # away. Stating CSVLogger explicitly is what prevents that.
    loggers = _loggers_written(base_config, tmp_path / "run")

    assert [entry["class_path"] for entry in loggers] == [
        "lightning.pytorch.loggers.CSVLogger"
    ]


def test_the_csv_logger_still_writes_where_the_metrics_reader_looks(base_config, tmp_path):
    run_dir = tmp_path / "run"
    loggers = _loggers_written(base_config, run_dir)

    init = loggers[0]["init_args"]
    assert init["save_dir"] == str(run_dir)
    assert init["name"] == "lightning_logs"


def test_tensorboard_is_added_without_displacing_the_csv_logger(base_config, tmp_path):
    loggers = _loggers_written(base_config, tmp_path / "run", tensorboard=True)

    assert [entry["class_path"] for entry in loggers] == [
        "lightning.pytorch.loggers.CSVLogger",
        "lightning.pytorch.loggers.TensorBoardLogger",
    ]


def test_the_two_loggers_do_not_share_a_version_directory(base_config, tmp_path):
    loggers = _loggers_written(base_config, tmp_path / "run", tensorboard=True)

    names = [entry["init_args"]["name"] for entry in loggers]
    assert len(set(names)) == len(names), f"loggers would collide under {names}"


def test_wandb_no_longer_replaces_the_csv_logger(base_config, tmp_path):
    loggers = _loggers_written(
        base_config, tmp_path / "run", wandb_init={"project": "p", "name": "n"}
    )

    classes = [entry["class_path"] for entry in loggers]
    assert "lightning.pytorch.loggers.CSVLogger" in classes
    assert "lightning.pytorch.loggers.WandbLogger" in classes


def test_the_manifest_override_survives_the_logger_changes(base_config, tmp_path):
    finetune = _load_finetune()
    dest = tmp_path / "run" / "data_override.yaml"
    finetune.write_run_override(
        base_config, dest, "/data", "manifest.csv", log_dir=tmp_path / "run"
    )

    written = yaml.safe_load(dest.read_text(encoding="utf-8"))
    entry = written["data"]["mix_dataset_config"]["CURATED"]
    assert entry["root_folder"] == "/data"
    assert entry["filename"] == "manifest.csv"
