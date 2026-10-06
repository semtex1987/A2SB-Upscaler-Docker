"""DSP behaviour that the restoration result depends on."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
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


def _tone_segment(hz: float, src_rate: int, seconds: float = 0.25, channels: int = 1) -> AudioSegment:
    """A unit-amplitude sine at `src_rate`, packed as 16-bit PCM."""
    t = np.arange(int(src_rate * seconds), dtype=np.float64) / src_rate
    mono = np.sin(2 * np.pi * hz * t)
    if channels == 2:
        # Right channel is a different frequency so alignment is observable.
        right = np.sin(2 * np.pi * (hz * 0.5) * t)
        samples = np.column_stack((mono, right))
    else:
        samples = mono
    peak = float(np.iinfo(np.int16).max)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * peak).astype(np.int16)
    if channels == 2:
        pcm = np.ascontiguousarray(pcm)
    return AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=src_rate,
        channels=channels,
    )


def _peak_hz(segment: AudioSegment, channel: int = 0) -> float:
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))[:, channel]
    windowed = samples * np.hanning(samples.size)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.size, 1.0 / segment.frame_rate)
    return float(freqs[int(np.argmax(spec))])


def _band_rms(segment: AudioSegment, lo_hz: float, hi_hz: float) -> float:
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))[:, 0]
    spec = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    freqs = np.fft.rfftfreq(samples.size, 1.0 / segment.frame_rate)
    band = spec[(freqs >= lo_hz) & (freqs <= hi_hz)]
    if band.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(band**2)))


def test_downsampling_rejects_content_above_target_nyquist():
    """A 30 kHz tone at 96 kHz must not fold to 14.1 kHz at 44.1 kHz.

    pydub's linear interpolator aliases; a band-limited resampler rejects it.
    """
    source = _tone_segment(30000, src_rate=96000)
    converted = ensure_a2sb_input_format(source)

    assert converted.frame_rate == MODEL_SAMPLE_RATE
    alias_energy = _band_rms(converted, 13000, 15200)
    residual_high = _band_rms(converted, 18000, 22000)
    # A correctly anti-aliased 30 kHz tone leaves essentially nothing in-band.
    assert alias_energy < 50.0
    assert residual_high < 50.0
    assert _peak_hz(converted) != pytest.approx(14100, abs=400)


def test_downsampling_rejects_a_23khz_tone_from_48khz():
    """A 23 kHz tone at 48 kHz must not fold to 21.1 kHz at 44.1 kHz."""
    source = _tone_segment(23000, src_rate=48000)
    converted = ensure_a2sb_input_format(source)

    assert converted.frame_rate == MODEL_SAMPLE_RATE
    alias_energy = _band_rms(converted, 20000, 22000)
    assert alias_energy < 50.0
    assert _peak_hz(converted) != pytest.approx(21100, abs=400)


def test_downsampling_preserves_duration_and_stereo_alignment():
    source = _tone_segment(1000, src_rate=96000, seconds=0.5, channels=2)
    converted = ensure_a2sb_input_format(source)

    assert converted.channels == 2
    assert converted.frame_rate == MODEL_SAMPLE_RATE
    assert converted.sample_width == 2
    assert abs(len(converted) - len(source)) <= 2
    # Left stays the 1 kHz tone; right stays the 500 Hz tone.
    assert _peak_hz(converted, channel=0) == pytest.approx(1000, abs=20)
    assert _peak_hz(converted, channel=1) == pytest.approx(500, abs=20)


def test_high_band_rms_is_calibrated_dbfs_for_a_known_tone(tmp_path):
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32)
    path = write_wav(tmp_path / "tone15k.wav", tone)
    measured = high_band_rms_db(str(path), 12000)
    expected = 20.0 * np.log10(0.5 / np.sqrt(2.0))
    assert measured == pytest.approx(expected, abs=0.4)


def test_high_band_rms_does_not_cancel_antiphase_stereo(tmp_path):
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    left = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32)
    stereo = np.stack([left, -left], axis=1)
    path = write_wav(tmp_path / "side.wav", stereo)
    measured = high_band_rms_db(str(path), 12000)
    assert measured > -20.0


def test_high_band_rms_separates_a_transcode_from_a_master(input_dir, master_wav):
    transcode = write_wav(input_dir / "hb_transcode.wav", brickwalled(11000))

    quiet = high_band_rms_db(str(transcode), 12000)
    loud = high_band_rms_db(str(master_wav), 12000)

    assert quiet < loud - 20


def test_high_band_rms_above_nyquist_is_the_empty_sentinel(master_wav):
    assert high_band_rms_db(str(master_wav), 30000) == pytest.approx(-200.0)


def test_high_band_rms_is_calibrated_dbfs(tmp_path):
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    sine = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32)
    path = write_wav(tmp_path / "tone15k.wav", sine)
    expected = 20.0 * np.log10(0.5 / np.sqrt(2.0))
    assert high_band_rms_db(str(path), 10000) == pytest.approx(expected, abs=0.6)


def test_high_band_rms_does_not_cancel_anti_phase_stereo(tmp_path):
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 15000 * t)
    stereo = np.stack([left, -left], axis=1).astype(np.float32)
    path = tmp_path / "side.wav"
    sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_16")
    expected = 20.0 * np.log10(0.5 / np.sqrt(2.0))
    assert high_band_rms_db(str(path), 10000) == pytest.approx(expected, abs=0.6)


@pytest.mark.parametrize(
    "name,samples",
    [
        ("square_clipping", np.sign(np.sin(np.linspace(0, 400, SAMPLE_RATE))).astype(np.float32)),
    ],
)
def test_corruption_check_rejects_degenerate_audio(input_dir, name, samples):
    path = write_wav(input_dir / f"corrupt_{name}.wav", samples)
    assert is_likely_corrupted_audio(str(path)) is True


def test_corruption_check_accepts_quiet_and_silent_channels(tmp_path):
    silence = write_wav(tmp_path / "silence.wav", np.zeros(SAMPLE_RATE, dtype=np.float32))
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    quiet = write_wav(
        tmp_path / "quiet.wav",
        (1e-3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
    )
    loud = write_wav(tmp_path / "loud.wav", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert is_likely_corrupted_audio(str(silence)) is False
    assert is_likely_corrupted_audio(str(quiet)) is False
    assert is_likely_corrupted_audio(str(silence), reference_path=str(loud)) is True


def test_corruption_check_accepts_ordinary_audio(master_wav):
    assert is_likely_corrupted_audio(str(master_wav)) is False


def test_corruption_check_rejects_an_unreadable_file(input_dir):
    path = input_dir / "not_audio.wav"
    path.write_text("this is not a wav file")
    assert is_likely_corrupted_audio(str(path)) is True
