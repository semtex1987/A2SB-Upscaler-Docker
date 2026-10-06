"""Signal processing shared by the restoration pipeline and the analysis API."""
from __future__ import annotations

import functools
from typing import Optional

import librosa
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt, sosfiltfilt
import soundfile as sf

from server.config import ANALYSIS_MAX_SECONDS, MODEL_SAMPLE_RATE


@functools.lru_cache(maxsize=128)
def _get_butter_sos(order: int, normal_cutoff: float):
    """`butter` is slow and the cutoff/rate pair is usually static across a batch."""
    return butter(order, normal_cutoff, btype="low", analog=False, output="sos")


def butter_lowpass_filter(data, cutoff: float, fs: int, order: int = 10):
    data_arr = np.asarray(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        return data_arr

    # pydub provides PCM integer samples (typically int16). Filtering directly on
    # integers and casting back without clipping can wrap overflow and sound
    # severely distorted. Process in float domain, then clip before reconversion.
    int_dtype = np.issubdtype(data_arr.dtype, np.integer)
    if int_dtype:
        type_info = np.iinfo(data_arr.dtype)
        peak = float(max(abs(type_info.min), type_info.max))
        data_float = data_arr.astype(np.float32) / peak
    else:
        data_float = data_arr.astype(np.float32, copy=False)

    sos = _get_butter_sos(order, normal_cutoff)
    filtered = sosfilt(sos, data_float, axis=0)

    if not int_dtype:
        return filtered

    # Keep int range stable and avoid wrap-around artifacts.
    filtered = np.clip(filtered, -1.0, 1.0 - (1.0 / peak))
    return np.round(filtered * peak).astype(data_arr.dtype)


def apply_lowpass_to_segment(segment: AudioSegment, cutoff_freq_hz: float) -> AudioSegment:
    channel_data = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        channel_data = channel_data.reshape((-1, 2))
    filtered_data = butter_lowpass_filter(channel_data, cutoff_freq_hz, segment.frame_rate)
    return segment._spawn(filtered_data.tobytes())


def _pcm_to_float(samples: np.ndarray) -> np.ndarray:
    """Map integer PCM onto [-1, 1]; leave floating arrays as float32."""
    if np.issubdtype(samples.dtype, np.integer):
        peak = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        return samples.astype(np.float32) / peak
    return samples.astype(np.float32, copy=False)


def _float_to_int16(samples: np.ndarray) -> np.ndarray:
    """Clip float audio and pack as 16-bit PCM without wrap-around."""
    peak = float(np.iinfo(np.int16).max)
    clipped = np.clip(samples, -1.0, 1.0 - (1.0 / peak))
    return np.round(clipped * peak).astype(np.int16)


def ensure_a2sb_input_format(segment: AudioSegment) -> AudioSegment:
    """Match the sampling rate the model configs declare, at 16-bit PCM.

    Downsampling is band-limited in float before the integer conversion.
    pydub's ``set_frame_rate`` uses linear interpolation, which folds energy
    above the target Nyquist into the passband; ``res_type="soxr_hq"`` is
    soxr's high-quality sinc kernel and rejects that band instead.

    Source: https://librosa.org/doc/latest/generated/librosa.resample.html
    """
    samples = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        samples = samples.reshape((-1, 2))
    data = _pcm_to_float(samples)

    orig_sr = int(segment.frame_rate)
    if orig_sr != MODEL_SAMPLE_RATE:
        # Time is axis 0 for both mono (n,) and stereo (n, channels).
        # soxr_hq is librosa 0.11's default band-limited sinc; pin it so a
        # missing extra cannot silently fall through to a weaker backend.
        data = librosa.resample(
            data,
            orig_sr=orig_sr,
            target_sr=MODEL_SAMPLE_RATE,
            res_type="soxr_hq",
            axis=0,
        )

    pcm = _float_to_int16(data)
    return AudioSegment(
        data=np.ascontiguousarray(pcm).tobytes(),
        sample_width=2,
        frame_rate=MODEL_SAMPLE_RATE,
        channels=segment.channels,
    )


def high_band_rms_db(path: str, cutoff_hz: float) -> float:
    """Calibrated high-band RMS in dBFS.

    Each channel is high-passed independently and channel powers are averaged,
    so anti-phase (side-only) stereo is not cancelled by a mono mixdown. The
    result is ``10 * log10(mean_power)`` with full-scale ±1.0 as 0 dBFS. Added
    energy here can be restoration or noise; it is not a quality score.
    """
    y, sr = librosa.load(path, sr=None, mono=False, duration=ANALYSIS_MAX_SECONDS)
    if y.size == 0 or sr <= 0:
        return -200.0
    if y.ndim == 1:
        y = y[np.newaxis, :]
    nyq = 0.5 * float(sr)
    if cutoff_hz >= nyq:
        return -200.0
    sos = butter(8, max(cutoff_hz / nyq, 1e-6), btype="highpass", analog=False, output="sos")
    filtered = sosfiltfilt(sos, y, axis=-1)
    channel_powers = np.mean(np.square(filtered), axis=-1)
    mean_power = float(np.mean(channel_powers))
    if mean_power <= 1e-20:
        return -200.0
    return 10.0 * np.log10(mean_power)


def is_silent_segment(segment: AudioSegment) -> bool:
    """True when a channel has no samples or a zero peak (hard-panned silence)."""
    samples = np.array(segment.get_array_of_samples())
    if samples.size == 0:
        return True
    return float(np.max(np.abs(samples))) == 0.0


def is_likely_corrupted_audio(path: str, reference_path: Optional[str] = None) -> bool:
    """Detect structurally broken or failed-synthesis audio, not quiet content.

    Silence and very quiet channels (around −60 dBFS) are valid. A silent
    output is only treated as a failure when *reference_path* has real energy.
    """
    try:
        segment = AudioSegment.from_file(path)
        samples = np.array(segment.get_array_of_samples())
    except Exception:
        return True

    if samples.size == 0 or not np.isfinite(samples).all():
        return True

    if np.issubdtype(samples.dtype, np.integer):
        full_scale = float(np.iinfo(samples.dtype).max)
    else:
        full_scale = 1.0

    abs_samples = np.abs(samples.astype(np.float64))
    peak = float(np.max(abs_samples)) if abs_samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(abs_samples)))) if abs_samples.size else 0.0
    quiet = peak <= 0.0 or rms < full_scale * 1e-3

    if quiet:
        if reference_path:
            try:
                ref = AudioSegment.from_file(reference_path)
                ref_samples = np.array(ref.get_array_of_samples())
                ref_peak = float(np.max(np.abs(ref_samples))) if ref_samples.size else 0.0
            except Exception:
                ref_peak = 0.0
            if ref_peak > 0.0:
                return True
        return False

    clipped_ratio = float(np.mean(abs_samples >= (full_scale * 0.995)))
    if clipped_ratio > 0.25:
        return True

    # Spectral flatness near 1.0 indicates noise-like content. Only treat that
    # as failed synthesis when the output is loud enough to be a hallucination.
    try:
        y = samples.astype(np.float32) / max(peak, 1.0)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=2048)
        if float(np.mean(flatness)) > 0.6:
            return True
    except Exception:
        pass

    return False


def export_with_linked_headroom(path: str, samples: np.ndarray, sample_rate: int) -> tuple[int, float]:
    """Write 16-bit PCM after stereo-linked peak scaling.

    ``samples`` is float in any range, shape ``(n,)`` or ``(n, channels)``.
    Both channels share one gain so the image is not rebalanced. Returns the
    number of over-range samples before scaling and the original peak.
    """
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    n_over = int(np.sum(np.abs(data) > 1.0))
    if peak > 1.0:
        data = data / peak
    pcm = _float_to_int16(data)
    sf.write(path, pcm, sample_rate, subtype="PCM_16")
    return n_over, peak
