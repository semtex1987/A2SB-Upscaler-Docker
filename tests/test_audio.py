"""DSP behaviour that the restoration result depends on."""
from __future__ import annotations

import numpy as np
import pytest
from pydub import AudioSegment

from server.audio import (
    apply_lowpass_to_segment,
    butter_lowpass_filter,
    ensure_a2sb_input_format,
    high_band_rms_db,
    is_likely_corrupted_audio,
)
from server.config import MODEL_SAMPLE_RATE

from .conftest import SAMPLE_RATE, brickwalled, write_wav


def _two_tone(low_hz: int = 1000, high_hz: int = 10000, seconds: float = 0.2) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return np.sin(2 * np.pi * low_hz * t) + np.sin(2 * np.pi * high_hz * t)


def test_stereo_channels_filter_independently():
    """Filtering must run down the time axis, not across the channel axis.

    Filtering across channels mixes left into right and is inaudible in a quick
    listen, which is exactly why it needs a test.
    """
    signal = _two_tone()
    stereo = np.column_stack((signal, np.sin(2 * np.pi * 1000 * np.arange(signal.size) / SAMPLE_RATE)))

    filtered = butter_lowpass_filter(stereo, 4000, SAMPLE_RATE)
    left = butter_lowpass_filter(stereo[:, 0], 4000, SAMPLE_RATE)
    right = butter_lowpass_filter(stereo[:, 1], 4000, SAMPLE_RATE)

    assert np.allclose(filtered[:, 0], left)
    assert np.allclose(filtered[:, 1], right)


def test_lowpass_removes_the_stopband_tone():
    signal = _two_tone()
    filtered = butter_lowpass_filter(signal, 4000, SAMPLE_RATE)

    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(filtered.size, 1 / SAMPLE_RATE)
    passband = spectrum[np.argmin(np.abs(freqs - 1000))]
    stopband = spectrum[np.argmin(np.abs(freqs - 10000))]

    assert stopband < passband * 0.01


def test_integer_input_is_clipped_not_wrapped():
    """int16 overflow wraps to the opposite rail and sounds like harsh clicks."""
    loud = np.full(2048, np.iinfo(np.int16).max, dtype=np.int16)
    loud[::2] = np.iinfo(np.int16).min

    filtered = butter_lowpass_filter(loud, 4000, SAMPLE_RATE)

    assert filtered.dtype == np.int16
    assert filtered.min() >= np.iinfo(np.int16).min
    assert filtered.max() <= np.iinfo(np.int16).max


def test_cutoff_at_or_above_nyquist_is_a_passthrough():
    signal = _two_tone()
    assert np.array_equal(butter_lowpass_filter(signal, SAMPLE_RATE, SAMPLE_RATE), signal)


def test_segment_lowpass_preserves_frame_count_and_channels(input_dir):
    path = write_wav(input_dir / "segment_lowpass.wav", brickwalled(16000), channels=2)
    segment = AudioSegment.from_file(path)

    filtered = apply_lowpass_to_segment(segment, 6000)

    assert filtered.channels == segment.channels
    assert filtered.frame_count() == segment.frame_count()
    assert filtered.frame_rate == segment.frame_rate


def test_input_format_matches_the_model_configs(input_dir):
    path = write_wav(input_dir / "resample_me.wav", brickwalled(8000))
    segment = AudioSegment.from_file(path).set_frame_rate(22050)

    normalized = ensure_a2sb_input_format(segment)

    assert normalized.frame_rate == MODEL_SAMPLE_RATE
    assert normalized.sample_width == 2


def test_high_band_rms_separates_a_transcode_from_a_master(input_dir, master_wav):
    transcode = write_wav(input_dir / "hb_transcode.wav", brickwalled(11000))

    quiet = high_band_rms_db(str(transcode), 12000)
    loud = high_band_rms_db(str(master_wav), 12000)

    assert quiet < loud - 20


def test_high_band_rms_above_nyquist_is_the_empty_sentinel(master_wav):
    assert high_band_rms_db(str(master_wav), 30000) == pytest.approx(-200.0)


@pytest.mark.parametrize(
    "name,samples",
    [
        ("silence", np.zeros(SAMPLE_RATE, dtype=np.float32)),
        ("square_clipping", np.sign(np.sin(np.linspace(0, 400, SAMPLE_RATE))).astype(np.float32)),
    ],
)
def test_corruption_check_rejects_degenerate_audio(input_dir, name, samples):
    path = write_wav(input_dir / f"corrupt_{name}.wav", samples)
    assert is_likely_corrupted_audio(str(path)) is True


def test_corruption_check_accepts_ordinary_audio(master_wav):
    assert is_likely_corrupted_audio(str(master_wav)) is False


def test_corruption_check_rejects_an_unreadable_file(input_dir):
    path = input_dir / "not_audio.wav"
    path.write_text("this is not a wav file")
    assert is_likely_corrupted_audio(str(path)) is True
