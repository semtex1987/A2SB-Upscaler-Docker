"""Subprocess driving: progress parsing, argument units, and cancellation."""
from __future__ import annotations

import os
import textwrap
import threading
import time

import pytest

from server import inference
from server.inference import (
    InferenceCancelled,
    InferenceError,
    run_a2sb_inference,
)
from server.process import parse_eta_seconds as _parse_eta_seconds, parse_progress as _parse_progress

from .conftest import brickwalled, write_wav

TQDM_LINE = "Predicting DataLoader 0:  45%|####      | 9/20 [00:12<00:15,  1.4s/it]"


def test_percentage_is_read_from_the_progress_bar():
    assert _parse_progress(TQDM_LINE) == pytest.approx(0.45)


def test_step_ratio_is_used_when_there_is_no_percentage():
    assert _parse_progress("Epoch 0: 3/12 [00:04<00:11]") == pytest.approx(0.25)


def test_ordinary_log_lines_report_no_progress():
    assert _parse_progress("Loading checkpoint /app/ckpts/A2SB_twosplit_0.0_0.5_release.ckpt") is None


def test_progress_is_clamped_to_the_unit_interval():
    assert _parse_progress("Predicting: 120%|###|") == 1.0


@pytest.mark.parametrize(
    "line,expected",
    [
        (TQDM_LINE, 15),
        ("it [00:30<02:05, 1.0s/it]", 125),
        ("it [00:30<1:02:05, 1.0s/it]", 3725),
    ],
)
def test_eta_is_read_from_the_remaining_field(line, expected):
    assert _parse_eta_seconds(line) == expected


def test_eta_is_absent_when_the_bar_has_none():
    assert _parse_eta_seconds("Predicting DataLoader 0:  45%|####|") is None


def _install_fake_runner(tmp_path, monkeypatch, body: str) -> None:
    """Point the runner at a throwaway script with the real CLI contract."""
    script = tmp_path / inference.INFERENCE_SCRIPT
    script.write_text(textwrap.dedent(body))
    monkeypatch.setattr(inference, "INFERENCE_CWD", str(tmp_path))


ARGPARSE_PREAMBLE = """
    import argparse, sys, time
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="infile")
    parser.add_argument("-o", dest="outfile")
    parser.add_argument("-n", dest="steps", type=int)
    parser.add_argument("-c", dest="cutoff", type=int)
    parser.add_argument("-b", dest="batch", type=int)
    args = parser.parse_args()
"""


def test_cutoff_is_passed_in_hz_not_normalised(tmp_path, monkeypatch, input_dir):
    """UpsampleMask turns this into an FFT bin index; a 0-1 value masks everything."""
    source = write_wav(input_dir / "cli_args.wav", brickwalled(11000))
    output = tmp_path / "out.wav"
    _install_fake_runner(
        tmp_path,
        monkeypatch,
        ARGPARSE_PREAMBLE
        + """
    open("args.txt", "w").write(repr(vars(args)))
    import shutil; shutil.copy(args.infile, args.outfile)
    """,
    )
    monkeypatch.setattr(inference, "is_likely_corrupted_audio", lambda _path: False)

    run_a2sb_inference(
        input_path=str(source),
        output_path=str(output),
        steps=50,
        cutoff_hz=14000,
        batch_size=8,
        on_log=lambda _line: None,
        on_progress=lambda _f, _e: None,
        cancel_event=threading.Event(),
    )

    recorded = eval((tmp_path / "args.txt").read_text())  # noqa: S307 - our own literal
    assert recorded["cutoff"] == 14000
    assert recorded["steps"] == 50
    assert recorded["batch"] == 8


def test_progress_streams_while_the_process_runs(tmp_path, monkeypatch, input_dir):
    """tqdm redraws with carriage returns; line buffering would hide all of this."""
    source = write_wav(input_dir / "cli_progress.wav", brickwalled(11000))
    _install_fake_runner(
        tmp_path,
        monkeypatch,
        ARGPARSE_PREAMBLE
        + """
    for i in range(1, 6):
        sys.stdout.write(f"\\rPredicting DataLoader 0: {i*20:3d}%|##| {i}/5 [00:01<00:0{5-i}, 1.0s/it]")
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\\ndone\\n")
    import shutil; shutil.copy(args.infile, args.outfile)
    """,
    )
    monkeypatch.setattr(inference, "is_likely_corrupted_audio", lambda _path: False)

    seen: list[tuple] = []
    run_a2sb_inference(
        input_path=str(source),
        output_path=str(tmp_path / "out.wav"),
        steps=5,
        cutoff_hz=12000,
        batch_size=1,
        on_log=lambda _line: None,
        on_progress=lambda fraction, eta: seen.append((fraction, eta)),
        cancel_event=threading.Event(),
    )

    fractions = [fraction for fraction, _ in seen if fraction is not None]
    assert fractions == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert any(eta is not None for _, eta in seen)


def test_a_failing_process_raises_with_the_output_tail(tmp_path, monkeypatch, input_dir):
    source = write_wav(input_dir / "cli_fail.wav", brickwalled(11000))
    _install_fake_runner(
        tmp_path,
        monkeypatch,
        ARGPARSE_PREAMBLE
        + """
    print("CUDA out of memory. Tried to allocate 2.00 GiB")
    sys.exit(1)
    """,
    )

    with pytest.raises(InferenceError) as caught:
        run_a2sb_inference(
            input_path=str(source),
            output_path=str(tmp_path / "out.wav"),
            steps=5,
            cutoff_hz=12000,
            batch_size=1,
            on_log=lambda _line: None,
            on_progress=lambda _f, _e: None,
            cancel_event=threading.Event(),
        )

    assert "exited with code 1" in str(caught.value)
    assert "CUDA out of memory" in caught.value.tail


def test_a_silent_success_without_output_is_still_an_error(tmp_path, monkeypatch, input_dir):
    source = write_wav(input_dir / "cli_nooutput.wav", brickwalled(11000))
    _install_fake_runner(tmp_path, monkeypatch, ARGPARSE_PREAMBLE + '\n    print("finished")\n')

    with pytest.raises(InferenceError, match="without writing an output file"):
        run_a2sb_inference(
            input_path=str(source),
            output_path=str(tmp_path / "missing.wav"),
            steps=5,
            cutoff_hz=12000,
            batch_size=1,
            on_log=lambda _line: None,
            on_progress=lambda _f, _e: None,
            cancel_event=threading.Event(),
        )


def test_a_missing_inference_tree_names_the_override(tmp_path, monkeypatch, input_dir):
    source = write_wav(input_dir / "cli_nocwd.wav", brickwalled(11000))
    monkeypatch.setattr(inference, "INFERENCE_CWD", str(tmp_path / "does-not-exist"))

    with pytest.raises(InferenceError, match="A2SB_INFERENCE_CWD"):
        run_a2sb_inference(
            input_path=str(source),
            output_path=str(tmp_path / "out.wav"),
            steps=5,
            cutoff_hz=12000,
            batch_size=1,
            on_log=lambda _line: None,
            on_progress=lambda _f, _e: None,
            cancel_event=threading.Event(),
        )


def test_cancelling_stops_the_process_group(tmp_path, monkeypatch, input_dir):
    """Cancellation has to reach the Lightning child, not just the wrapper."""
    source = write_wav(input_dir / "cli_cancel.wav", brickwalled(11000))
    _install_fake_runner(
        tmp_path,
        monkeypatch,
        ARGPARSE_PREAMBLE
        + """
    open("pid.txt", "w").write(str(__import__("os").getpid()))
    time.sleep(120)
    """,
    )

    cancel_event = threading.Event()
    threading.Timer(1.5, cancel_event.set).start()
    started = time.monotonic()

    with pytest.raises(InferenceCancelled):
        run_a2sb_inference(
            input_path=str(source),
            output_path=str(tmp_path / "out.wav"),
            steps=5,
            cutoff_hz=12000,
            batch_size=1,
            on_log=lambda _line: None,
            on_progress=lambda _f, _e: None,
            cancel_event=cancel_event,
        )

    assert time.monotonic() - started < 20
    child_pid = int((tmp_path / "pid.txt").read_text())
    time.sleep(0.5)
    with pytest.raises(OSError):
        os.kill(child_pid, 0)
