"""Shared training-file eligibility for the GUI, the manifest, and the loader.

The dataset loader drops rows whose ``estimated_true_sr`` is below 32 kHz
(16 kHz of real bandwidth) when ``apply_sr_loss_mask`` is on. Counting or
splitting those files before that filter made the GUI, the manifest, and the
actual MixAudioDataset disagree: a file could PASS vetting and still contribute
zero training samples.
"""
from __future__ import annotations

import librosa
import numpy as np

#: Must match ``datasets.read_standard_csv`` when ``apply_sr_loss_mask`` is true.
LOADER_MIN_TRUE_SR = 32000


def spectral_rolloff_95(y: np.ndarray, sr: int) -> float:
    """95th percentile of per-frame 99% spectral rolloff (Hz)."""
    if y.size == 0 or sr <= 0:
        return 0.0
    rolloff_frames = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.99, n_fft=2048, hop_length=2048
    )
    return float(np.percentile(rolloff_frames, 95))


def estimate_true_sr_from_array(y: np.ndarray, sr: int) -> int:
    """2× the 95th-percentile of per-frame 99% spectral rolloff, capped at 44100."""
    return int(min(2 * spectral_rolloff_95(y, sr), 44100))


def estimate_true_sr(path: str, duration: float = 60.0) -> int:
    """Bandwidth estimate used by the manifest and the GUI trainer-gate."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True, duration=duration)
        return estimate_true_sr_from_array(y, int(sr))
    except Exception:
        return 0


def is_loader_eligible(true_sr: int) -> bool:
    """True when MixAudioDataset would keep this file under apply_sr_loss_mask."""
    return int(true_sr) >= LOADER_MIN_TRUE_SR


def n_loader_segments(
    duration: float,
    segment_length: int = 130560,
    sampling_rate: int = 44100,
) -> int:
    """Match MixAudioDataset.build_file_idx_mapping segment counting."""
    if duration <= 0:
        return 0
    segment_time = float(segment_length) / float(sampling_rate) + 0.001
    return int(np.floor(duration / segment_time))
