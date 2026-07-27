"""Per-file restoration, with the diffusion step replaced by a cheap stand-in."""
from __future__ import annotations

import shutil
import threading

import numpy as np
import pytest
import soundfile as sf

from server import pipeline as pipeline_module
from server.inference import InferenceCancelled
from server.pipeline import PipelineError, cleanup_run_dir, restore_file

from .conftest import SAMPLE_RATE, brickwalled, write_wav


def _fake_inference(add_high_band: bool = True):
    """Stand in for A2SB: copy the channel through, optionally adding HF energy."""

    def run(*, input_path, output_path, cutoff_hz, on_log, on_progress, cancel_event, **_kwargs):
        on_log(f"fake inference at {cutoff_hz} Hz")
        on_progress(0.5, 4.0)
        samples, rate = sf.read(input_path)
        if add_high_band:
            rng = np.random.default_rng(5)
            spectrum = np.fft.rfft(rng.normal(0, 1, samples.shape[0]))
            freqs = np.fft.rfftfreq(samples.shape[0], 1 / rate)
            spectrum[(freqs < cutoff_hz) | (freqs > 20000)] = 0
            high = np.fft.irfft(spectrum, n=samples.shape[0])
            high *= (np.max(np.abs(samples)) * 0.15) / (np.max(np.abs(high)) or 1.0)
            samples = np.clip(samples + high, -1.0, 1.0)
        sf.write(output_path, samples, rate)
        on_progress(1.0, 0.0)

    return run


@pytest.fixture
def stereo_source(input_dir):
    return write_wav(input_dir / "pipeline_stereo.wav", brickwalled(11000), channels=2)


@pytest.fixture
def mono_source(input_dir):
    return write_wav(input_dir / "pipeline_mono.wav", brickwalled(11000))


def _restore(source, run_dir, monkeypatch, runner=None, **overrides):
    monkeypatch.setattr(pipeline_module, "run_a2sb_inference", runner or _fake_inference())
    calls = []
    result = restore_file(
        source_path=str(source),
        run_dir=run_dir,
        steps=overrides.get("steps", 10),
        cutoff_hz=overrides.get("cutoff_hz", 11000),
        batch_size=4,
        on_progress=calls.append,
        on_log=lambda _line: None,
        cancel_event=overrides.get("cancel_event", threading.Event()),
    )
    return result, calls


def test_a_stereo_file_is_restored_channel_by_channel(tmp_path, stereo_source, monkeypatch):
    """Both channels must be processed; a mono result would collapse the image."""
    result, progress = _restore(stereo_source, tmp_path / "run", monkeypatch)

    assert result.channels == 2
    assert sf.info(result.restored_path).channels == 2
    assert sf.info(result.filtered_path).channels == 2
    stages = [entry.stage for entry in progress]
    assert any("Left" in stage for stage in stages)
    assert any("Right" in stage for stage in stages)


def test_a_mono_file_reports_a_single_channel(tmp_path, mono_source, monkeypatch):
    result, progress = _restore(mono_source, tmp_path / "run", monkeypatch)

    assert result.channels == 1
    assert sf.info(result.restored_path).channels == 1
    assert any("Mono" in entry.stage for entry in progress)


def test_progress_rises_monotonically_and_ends_at_one(tmp_path, stereo_source, monkeypatch):
    _, progress = _restore(stereo_source, tmp_path / "run", monkeypatch)

    fractions = [entry.fraction for entry in progress if entry.fraction is not None]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert progress[-1].stage == "Done"


def test_added_high_band_energy_is_measured(tmp_path, stereo_source, monkeypatch):
    result, _ = _restore(stereo_source, tmp_path / "run", monkeypatch)

    assert result.high_band_delta_db > 3
    assert result.high_band_delta_db == pytest.approx(
        result.high_band_out_db - result.high_band_in_db, abs=0.02
    )
    assert result.warnings == []


def test_a_model_that_adds_nothing_produces_a_warning(tmp_path, stereo_source, monkeypatch):
    """The release checkpoints are weak above ~12 kHz; silence about that is worse."""
    result, _ = _restore(
        stereo_source, tmp_path / "run", monkeypatch, runner=_fake_inference(add_high_band=False)
    )

    assert result.high_band_delta_db < 1
    assert len(result.warnings) == 1
    assert "lower cutoff" in result.warnings[0]


def test_an_undecodable_source_names_the_file(tmp_path, input_dir, monkeypatch):
    broken = input_dir / "pipeline_broken.wav"
    broken.write_text("not audio")

    with pytest.raises(PipelineError, match="pipeline_broken.wav"):
        _restore(broken, tmp_path / "run", monkeypatch)


def test_more_than_two_channels_is_refused(tmp_path, input_dir, monkeypatch):
    samples = np.stack([brickwalled(11000)] * 3, axis=1)
    path = input_dir / "pipeline_surround.wav"
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")

    with pytest.raises(PipelineError, match="only mono and stereo"):
        _restore(path, tmp_path / "run", monkeypatch)


def test_cancellation_propagates_out_of_the_pipeline(tmp_path, stereo_source, monkeypatch):
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(InferenceCancelled):
        _restore(stereo_source, tmp_path / "run", monkeypatch, cancel_event=cancel_event)


def test_channel_scratch_files_are_cleaned_up(tmp_path, stereo_source, monkeypatch):
    from server.config import WORK_DIR

    before = set(WORK_DIR.iterdir())
    result, _ = _restore(stereo_source, tmp_path / "run", monkeypatch)

    assert set(WORK_DIR.iterdir()) == before
    # Only the two rendered files a user can act on remain in the run directory.
    remaining = sorted(p.name for p in (tmp_path / "run").iterdir())
    assert remaining == sorted(
        [result.restored_path.rsplit("/", 1)[1], result.filtered_path.rsplit("/", 1)[1]]
    )


def test_cleanup_removes_the_run_directory(tmp_path, stereo_source, monkeypatch):
    run_dir = tmp_path / "run"
    _restore(stereo_source, run_dir, monkeypatch)
    assert run_dir.exists()

    cleanup_run_dir(run_dir)

    assert not run_dir.exists()
    cleanup_run_dir(run_dir)  # idempotent


def test_a_source_name_with_spaces_yields_a_safe_stem(tmp_path, input_dir, monkeypatch):
    spaced = input_dir / "my track 01.wav"
    shutil.copy(write_wav(input_dir / "pipeline_spaces.wav", brickwalled(11000)), spaced)

    result, _ = _restore(spaced, tmp_path / "run", monkeypatch)

    assert result.name == "my track 01.wav"
    assert " " not in result.restored_path.rsplit("/", 1)[1]
