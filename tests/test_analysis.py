"""Bandwidth detection: the measurement the suggested cutoff is built on."""
from __future__ import annotations

import base64
import importlib.util

import librosa
import numpy as np
import pytest

from server.analysis import (
    analyze_source,
    median_smooth,
    peak_envelope,
    spectral_scan,
    spectrogram_payload,
)

from .conftest import REPO_ROOT, SAMPLE_RATE, brickwalled, full_bandwidth, write_wav


@pytest.fixture(scope="module")
def vet_dataset():
    """The standalone vetting script, loaded by path since it is not a package."""
    spec = importlib.util.spec_from_file_location(
        "vet_dataset", REPO_ROOT / "training" / "vet_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cutoff_hz", [8000, 11000, 15000, 19000])
def test_brickwall_is_found_near_its_true_cutoff(input_dir, cutoff_hz):
    path = write_wav(input_dir / f"scan_brickwall_{cutoff_hz}.wav", brickwalled(cutoff_hz))
    y, sr = librosa.load(path, sr=None, mono=True)

    edge_hz, shelf = spectral_scan(y, sr)

    assert shelf is True
    assert cutoff_hz - 500 <= edge_hz <= cutoff_hz + 1200


def test_full_bandwidth_material_is_not_reported_as_band_limited(input_dir):
    """The regression that matters: a level-only floor lands mid-band here.

    Ordinary material already slopes ~60 dB from bass to Nyquist, so a detector
    that thresholds against a percentile of the spectrum calls a full-bandwidth
    master an 8 kHz transcode.
    """
    path = write_wav(input_dir / "scan_full.wav", full_bandwidth())
    y, sr = librosa.load(path, sr=None, mono=True)

    edge_hz, shelf = spectral_scan(y, sr)

    assert shelf is False
    assert edge_hz > 20000


def test_gentle_rolloff_is_not_mistaken_for_a_cliff(input_dir):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(1, 6000, btype="low", fs=SAMPLE_RATE, output="sos")
    gentle = sosfiltfilt(sos, full_bandwidth())
    path = write_wav(input_dir / "scan_gentle.wav", (gentle / np.max(np.abs(gentle)) * 0.85))
    y, sr = librosa.load(path, sr=None, mono=True)

    _, shelf = spectral_scan(y, sr)

    assert shelf is False


def test_a_narrow_notch_is_not_a_bandwidth_limit(input_dir):
    """A crossover null drops steeply but the band above it comes back."""
    samples = full_bandwidth()
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1 / SAMPLE_RATE)
    spectrum[(freqs > 9000) & (freqs < 9600)] = 0
    notched = np.fft.irfft(spectrum, n=samples.size)
    path = write_wav(input_dir / "scan_notch.wav", (notched / np.max(np.abs(notched)) * 0.85))
    y, sr = librosa.load(path, sr=None, mono=True)

    edge_hz, shelf = spectral_scan(y, sr)

    assert shelf is False
    assert edge_hz > 20000


def test_silence_reports_no_content(input_dir):
    path = write_wav(input_dir / "scan_silence.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))
    y, sr = librosa.load(path, sr=None, mono=True)

    assert spectral_scan(y, sr) == (0.0, False)


@pytest.mark.parametrize("cutoff_hz", [9000, 13000, 17000])
def test_scan_agrees_with_the_dataset_vetting_tool(input_dir, vet_dataset, cutoff_hz):
    """Both tools answer the same question and must not drift apart."""
    path = write_wav(input_dir / f"parity_{cutoff_hz}.wav", brickwalled(cutoff_hz))
    y, sr = librosa.load(path, sr=None, mono=True)

    assert spectral_scan(y, sr) == vet_dataset.spectral_scan(y, sr, np, librosa)


def test_transcode_analysis_suggests_a_cutoff_under_the_cliff(transcode_wav):
    analysis = analyze_source(str(transcode_wav))

    assert analysis.verdict == "transcode"
    assert analysis.shelf is True
    assert analysis.channels == 2
    assert analysis.sample_rate == SAMPLE_RATE
    # Restoring from just under the cliff avoids feeding the model the
    # transition slope, where the source is already attenuated.
    assert analysis.suggested_cutoff_hz < analysis.hf_edge_hz
    assert "transcode" in analysis.note.lower()


def test_master_analysis_does_not_advertise_a_restoration(master_wav):
    analysis = analyze_source(str(master_wav))

    assert analysis.verdict == "full-bandwidth"
    assert analysis.shelf is False
    assert "fine-tuning source" in analysis.note


def test_analysis_serialises_to_camel_case(master_wav):
    payload = analyze_source(str(master_wav)).to_dict()

    assert "suggestedCutoffHz" in payload
    assert "hfEdgeHz" in payload
    assert not any("_" in key for key in payload)


def test_median_smooth_flattens_scatter_but_keeps_a_step():
    rng = np.random.default_rng(3)
    values = np.concatenate([np.zeros(200), np.full(200, -80.0)]) + rng.normal(0, 5, 400)

    smoothed = median_smooth(values, 31)

    assert smoothed.shape == values.shape
    assert abs(smoothed[100]) < 3
    assert abs(smoothed[300] + 80) < 3


def test_median_smooth_is_a_noop_for_a_degenerate_window():
    values = np.arange(5, dtype=float)
    assert np.array_equal(median_smooth(values, 1), values)
    assert np.array_equal(median_smooth(values, 99), values)


def test_spectrogram_payload_decodes_to_the_declared_grid(transcode_wav):
    payload = spectrogram_payload(str(transcode_wav))

    grid = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    assert grid.size == payload["width"] * payload["height"]
    assert payload["maxFrequencyHz"] == SAMPLE_RATE / 2
    assert payload["floorDb"] < 0


def test_spectrogram_puts_high_frequencies_at_the_top(transcode_wav):
    """Row 0 must be Nyquist; a flipped image makes the cutoff line point at bass."""
    payload = spectrogram_payload(str(transcode_wav))
    grid = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8).reshape(
        payload["height"], payload["width"]
    )

    assert grid[0].mean() < grid[-1].mean()


def test_spectrogram_honours_the_duration_limit(master_wav):
    clipped = spectrogram_payload(str(master_wav), max_seconds=1.0)
    assert clipped["durationSec"] == pytest.approx(1.0, abs=0.05)


def test_peak_envelope_is_normalised_and_bounded(master_wav):
    envelope = peak_envelope(str(master_wav), buckets=256)

    assert len(envelope["peaks"]) == 256
    assert max(envelope["peaks"]) == pytest.approx(1.0)
    assert min(envelope["peaks"]) >= 0.0
    assert envelope["durationSec"] > 0
