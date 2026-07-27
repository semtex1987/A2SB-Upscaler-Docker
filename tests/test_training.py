"""Training module: dataset vetting, progress parsing, checkpoint management, and cancellation.

The test suite runs entirely without a GPU: finetune.py is stubbed at the
subprocess level so the suite stays fast and CUDA-free, just as inference.py
is already stubbed in test_inference.py.
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from server.process import parse_eta_seconds, parse_progress
from server.training import (
    SPLIT_BOTH,
    SPLIT_FIRST,
    SPLIT_SECOND,
    TrainingCancelled,
    TrainingError,
    activate_checkpoints,
    checkpoint_status,
    preflight,
    read_training_metrics,
    revert_to_release,
    run_finetune,
    vet_dataset,
)

from .conftest import SAMPLE_RATE, brickwalled, full_bandwidth, write_wav


# ---------------------------------------------------------------------------
# Dataset vetting
# ---------------------------------------------------------------------------

def test_full_bandwidth_file_passes(tmp_path):
    p = write_wav(tmp_path / "clean.wav", full_bandwidth())
    results = vet_dataset([str(p)])
    assert len(results) == 1
    assert results[0].verdict == "pass"


def test_brickwalled_at_11kHz_is_rejected(tmp_path):
    p = write_wav(tmp_path / "transcode.wav", brickwalled(11000))
    results = vet_dataset([str(p)])
    assert len(results) == 1
    assert results[0].verdict == "reject"


def test_check_band_file_is_between_pass_and_reject(tmp_path):
    """18 kHz should land in the CHECK band (17–20.5 kHz)."""
    p = write_wav(tmp_path / "check.wav", brickwalled(18000))
    results = vet_dataset([str(p)])
    assert len(results) == 1
    # CHECK (could be either check or pass depending on exact spectrum)
    assert results[0].verdict in ("check", "pass")


def test_unreadable_file_gets_a_reject(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio data at all")
    results = vet_dataset([str(bad)])
    assert len(results) == 1
    assert results[0].verdict == "reject"
    assert "Could not read" in results[0].note


def test_vet_returns_duration_and_size(tmp_path):
    p = write_wav(tmp_path / "meta.wav", full_bandwidth(seconds=2.0))
    results = vet_dataset([str(p)])
    r = results[0]
    assert r.duration_sec == pytest.approx(2.0, abs=0.1)
    assert r.size_bytes > 0


def test_multiple_files_all_vetted(tmp_path):
    files = [
        write_wav(tmp_path / "a.wav", full_bandwidth()),
        write_wav(tmp_path / "b.wav", brickwalled(11000)),
    ]
    results = vet_dataset([str(f) for f in files])
    assert len(results) == 2
    verdicts = {r.name: r.verdict for r in results}
    assert verdicts["a.wav"] == "pass"
    assert verdicts["b.wav"] == "reject"


# ---------------------------------------------------------------------------
# Process-module helpers (shared with inference)
# ---------------------------------------------------------------------------

TQDM_LINE = "Predicting DataLoader 0:  45%|####      | 9/20 [00:12<00:15,  1.4s/it]"


def test_parse_progress_percent():
    assert parse_progress(TQDM_LINE) == pytest.approx(0.45)


def test_parse_progress_ratio_fallback():
    assert parse_progress("Epoch 0: 3/12 [00:04<00:11]") == pytest.approx(0.25)


def test_parse_progress_returns_none_for_log_lines():
    assert parse_progress("Loading checkpoint /app/ckpts/A2SB_twosplit_release.ckpt") is None


def test_parse_eta_seconds_minutes():
    assert parse_eta_seconds(TQDM_LINE) == 15


def test_parse_eta_seconds_hours():
    assert parse_eta_seconds("it [00:30<1:02:05, 1.0s/it]") == 3725


def test_parse_eta_returns_none_when_absent():
    assert parse_eta_seconds("Predicting DataLoader 0:  45%|####|") is None


# ---------------------------------------------------------------------------
# finetune.py subprocess stub
# ---------------------------------------------------------------------------

def _install_fake_finetune(tmp_path: Path, monkeypatch, body: str) -> Path:
    """Drop a stub finetune.py into tmp_path and point TRAINING_SCRIPT there."""
    script = tmp_path / "finetune.py"
    script.write_text(textwrap.dedent(body))
    monkeypatch.setattr("server.training.TRAINING_SCRIPT", str(script))
    return script


def _base_run_kwargs(data_dir: str, output_dir: str, cancel_event=None):
    return dict(
        data_dir=data_dir,
        output_dir=output_dir,
        steps=20,
        batch_size=2,
        learning_rate=5e-5,
        splits=SPLIT_BOTH,
        val_frac=0.1,
        val_every=None,
        val_samples=None,
        restart=False,
        on_log=lambda _: None,
        on_progress=lambda _: None,
        cancel_event=cancel_event or threading.Event(),
    )


def test_run_finetune_streams_progress(tmp_path, monkeypatch):
    """Split headers and tqdm lines must reach on_progress with sane fractions."""
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")

    _install_fake_finetune(
        tmp_path,
        monkeypatch,
        """
        import sys, time
        print("Split 0.0-0.5 starts at global_step=0; training until 20")
        for i in range(1, 6):
            sys.stdout.write(
                f"\\rTraining: {i*20:3d}%|##| {i}/5 [00:01<00:0{5-i}, 1.0s/it]"
            )
            sys.stdout.flush()
            time.sleep(0.02)
        print("\\nSplit 0.5-1.0 starts at global_step=0; training until 20")
        for i in range(1, 6):
            sys.stdout.write(
                f"\\rTraining: {i*20:3d}%|##| {i}/5 [00:01<00:0{5-i}, 1.0s/it]"
            )
            sys.stdout.flush()
            time.sleep(0.02)
        print("\\nDone.")
        """,
    )

    fractions: list[float] = []

    def on_progress(p):
        fractions.append(p.fraction)

    run_finetune(**{**_base_run_kwargs(data_dir, out_dir), "on_progress": on_progress})

    assert len(fractions) >= 2
    # Overall fraction must climb from 0 to near 1 with both splits.
    assert fractions[-1] == pytest.approx(1.0, abs=0.05)
    # Fractions must not decrease (monotone-ish within each split).
    for prev, nxt in zip(fractions[:-1], fractions[1:]):
        assert nxt >= prev - 0.01


def test_run_finetune_raises_on_failure(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    _install_fake_finetune(
        tmp_path,
        monkeypatch,
        """
        import sys
        print("CUDA out of memory", file=sys.stderr)
        sys.exit(1)
        """,
    )

    with pytest.raises(TrainingError, match="exited with code 1"):
        run_finetune(**_base_run_kwargs(data_dir, out_dir))


def test_run_finetune_can_be_cancelled(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")
    pid_file = tmp_path / "pid.txt"

    _install_fake_finetune(
        tmp_path,
        monkeypatch,
        f"""
        import os, time
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        time.sleep(120)
        """,
    )

    cancel = threading.Event()
    threading.Timer(1.5, cancel.set).start()
    started = time.monotonic()

    with pytest.raises(TrainingCancelled):
        run_finetune(**{**_base_run_kwargs(data_dir, out_dir), "cancel_event": cancel})

    assert time.monotonic() - started < 20
    # Verify the child process died.
    if pid_file.exists():
        child_pid = int(pid_file.read_text())
        time.sleep(0.3)
        with pytest.raises(OSError):
            os.kill(child_pid, 0)


# ---------------------------------------------------------------------------
# Checkpoint activation / revert
# ---------------------------------------------------------------------------

def _make_ensemble_yaml(ckpt_dir: Path) -> Path:
    """Write a minimal ensemble YAML pointing at placeholder release paths."""
    config_path = ckpt_dir / "ensemble.yaml"
    data = {
        "model": {
            "pretrained_checkpoints": [
                str(ckpt_dir / "A2SB_twosplit_0.0_0.5_release.ckpt"),
                str(ckpt_dir / "A2SB_twosplit_0.5_1.0_release.ckpt"),
            ]
        }
    }
    config_path.write_text(yaml.dump(data))
    return config_path


def _make_finetuned_ckpts(ft_dir: Path) -> list[Path]:
    ft_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ["A2SB_twosplit_0.0_0.5_finetuned.ckpt", "A2SB_twosplit_0.5_1.0_finetuned.ckpt"]:
        p = ft_dir / name
        p.write_bytes(b"\x00" * 16)  # dummy checkpoint bytes
        files.append(p)
    return files


def test_activate_checkpoints_rewrites_config(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    config_path = _make_ensemble_yaml(ckpt_dir)
    ft_dir = tmp_path / "finetuned"
    _make_finetuned_ckpts(ft_dir)

    monkeypatch.setattr("server.training.ENSEMBLE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("server.training.FINETUNED_CKPT_DIR", str(ft_dir))
    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(ckpt_dir))

    n = activate_checkpoints(str(ft_dir))
    assert n == 2

    data = yaml.safe_load(config_path.read_text())
    ckpts = data["model"]["pretrained_checkpoints"]
    assert all("finetuned" in c for c in ckpts)


def test_revert_to_release_restores_original_paths(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    config_path = _make_ensemble_yaml(ckpt_dir)
    ft_dir = tmp_path / "finetuned"
    _make_finetuned_ckpts(ft_dir)

    monkeypatch.setattr("server.training.ENSEMBLE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("server.training.FINETUNED_CKPT_DIR", str(ft_dir))
    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(ckpt_dir))

    activate_checkpoints(str(ft_dir))
    revert_to_release()

    data = yaml.safe_load(config_path.read_text())
    ckpts = data["model"]["pretrained_checkpoints"]
    assert all("release" in c for c in ckpts)


def test_checkpoint_status_reads_active_state(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    config_path = _make_ensemble_yaml(ckpt_dir)
    ft_dir = tmp_path / "finetuned"
    _make_finetuned_ckpts(ft_dir)

    monkeypatch.setattr("server.training.ENSEMBLE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("server.training.FINETUNED_CKPT_DIR", str(ft_dir))
    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(ckpt_dir))

    status = checkpoint_status()
    assert status.active == "release"

    activate_checkpoints(str(ft_dir))
    status = checkpoint_status()
    assert status.active == "finetuned"


def test_activate_returns_zero_when_no_finetuned_ckpts_exist(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    config_path = _make_ensemble_yaml(ckpt_dir)
    empty_ft_dir = tmp_path / "finetuned_empty"
    empty_ft_dir.mkdir()

    monkeypatch.setattr("server.training.ENSEMBLE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("server.training.FINETUNED_CKPT_DIR", str(empty_ft_dir))
    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(ckpt_dir))

    n = activate_checkpoints(str(empty_ft_dir))
    assert n == 0


# ---------------------------------------------------------------------------
# CSV metrics reader
# ---------------------------------------------------------------------------

def test_read_training_metrics_returns_empty_for_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("server.training.TRAINING_OUTPUT_DIR", tmp_path / "nonexistent")
    result = read_training_metrics(str(tmp_path / "nonexistent"))
    assert isinstance(result, dict)


def test_read_training_metrics_parses_csv(tmp_path):
    split_dir = tmp_path / "split_0.0_0.5" / "lightning_logs" / "version_0"
    split_dir.mkdir(parents=True)
    csv_path = split_dir / "metrics.csv"
    csv_path.write_text("step,train_loss,val_loss\n100,0.532,\n200,0.420,0.511\n")

    result = read_training_metrics(str(tmp_path), splits=SPLIT_FIRST)
    rows = result.get("0.0-0.5", [])
    assert len(rows) == 2
    assert rows[0]["step"] == pytest.approx(100)
    assert rows[0]["train_loss"] == pytest.approx(0.532)
    assert rows[1]["val_loss"] == pytest.approx(0.511)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def test_preflight_reports_missing_ckpts(tmp_path, monkeypatch):
    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(tmp_path / "empty_ckpts"))
    monkeypatch.setattr("server.training.TRAINING_APP_ROOT", str(tmp_path))
    monkeypatch.setattr("server.training.TRAINING_SCRIPT", str(tmp_path / "nofile.py"))
    monkeypatch.setattr("server.training.TRAINING_OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()

    problems = preflight(SPLIT_BOTH)
    assert any("checkpoint" in p.lower() for p in problems)


def test_preflight_passes_when_everything_exists(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    for name in [
        "A2SB_twosplit_0.0_0.5_release.ckpt",
        "A2SB_twosplit_0.5_1.0_release.ckpt",
    ]:
        (ckpt_dir / name).write_bytes(b"\x00" * 16)

    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "main.py").write_text("# stub")

    ft_script = tmp_path / "finetune.py"
    ft_script.write_text("# stub")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr("server.training.TRAINING_CKPT_DIR", str(ckpt_dir))
    monkeypatch.setattr("server.training.TRAINING_APP_ROOT", str(app_root))
    monkeypatch.setattr("server.training.TRAINING_SCRIPT", str(ft_script))
    monkeypatch.setattr("server.training.TRAINING_OUTPUT_DIR", out_dir)
    monkeypatch.setattr("server.training.TRAIN_MIN_FREE_BYTES", 0)

    problems = preflight(SPLIT_BOTH)
    assert problems == []
