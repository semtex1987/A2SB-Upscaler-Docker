"""Source inspection: what bandwidth a file really has, and spectrogram data.

The bandwidth scan mirrors `training/vet_dataset.py::spectral_scan` so the
cutoff this UI suggests agrees with what the dataset vetting tool reports.
The thresholds below must stay in sync with that module.
"""
from __future__ import annotations

import base64
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from server.config import (
    CUTOFF_MAX_HZ,
    CUTOFF_MIN_HZ,
    SPECTROGRAM_HEIGHT,
    SPECTROGRAM_WIDTH,
)

#: A drop steeper than this across ~1 kHz counts as a brickwall cliff.
SHELF_DROP_DB = 40.0
#: A cliff this far below Nyquist means the lowpass is artificial, not musical.
DEAD_BAND_HZ = 2000.0
#: Window for the spectral scan. Long enough to catch intermittent HF, short
#: enough that analysing a dropped folder stays interactive.
ANALYSIS_SEC = 120.0
#: Floor for the displayed spectrogram, relative to the file's peak bin.
SPECTROGRAM_FLOOR_DB = -80.0


@dataclass(frozen=True)
class SourceAnalysis:
    path: str
    name: str
    size_bytes: int
    duration_sec: float
    sample_rate: int
    channels: int
    #: Highest frequency carrying real energy, from the 95th-percentile spectrum.
    hf_edge_hz: float
    #: True when the spectrum cliffs at `hf_edge_hz` (artificial lowpass).
    shelf: bool
    verdict: str
    note: str
    suggested_cutoff_hz: int

    def to_dict(self) -> dict:
        return asdict(self)


def _load_for_analysis(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=None, mono=True, duration=ANALYSIS_SEC)
    return y, int(sr)


def spectral_scan(y: np.ndarray, sr: int) -> tuple[float, bool]:
    """Highest frequency carrying real energy, and whether the spectrum cliffs there.

    The edge comes from the 95th-percentile-over-time spectrum rather than the
    mean: transient HF (pick attacks, brushes, cymbals) is intermittent and a
    mean spectrum averages it below the noise floor, which unfairly judges
    sparse acoustic material as band-limited.
    """
    n_fft = 4096
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=1024))
    p95 = np.percentile(spec, 95, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak = float(p95.max())
    if peak <= 0:
        return 0.0, False
    p95_db = 20.0 * np.log10(np.maximum(p95 / peak, 1e-12))

    # Adaptive floor: the file's own quietest bins. Above an artificial cutoff
    # the dead region sits at this floor; in a genuine file HF energy stays
    # above it toward Nyquist.
    floor = float(np.percentile(p95_db, 10))
    above = np.where(p95_db > floor + 6.0)[0]
    if len(above) == 0:
        return 0.0, False
    edge_bin = int(above.max())
    edge_hz = float(freqs[edge_bin])

    bin_width = sr / n_fft
    look = max(1, int(round(500.0 / bin_width)))
    lo = max(0, edge_bin - look)
    hi = min(len(p95_db) - 1, edge_bin + look)
    drop = float(p95_db[lo] - p95_db[hi])
    return edge_hz, drop > SHELF_DROP_DB


def _suggest_cutoff(edge_hz: float, shelf: bool, sr: int) -> tuple[int, str, str]:
    """Turn the scan into a cutoff, a verdict and a sentence explaining both."""
    nyquist = sr / 2.0

    if edge_hz <= 0:
        return CUTOFF_MIN_HZ, "unknown", "No usable spectral content found; check the file decodes."

    if shelf and (nyquist - edge_hz) > DEAD_BAND_HZ:
        # A brickwall means the content above the cliff is already gone. Sit just
        # under it so the model regenerates from the last band that still has
        # real signal rather than from the transition slope.
        cutoff = int(round((edge_hz - 500.0) / 100.0) * 100)
        note = (
            f"Brickwall cliff at {edge_hz / 1000:.1f} kHz with a "
            f"{(nyquist - edge_hz) / 1000:.1f} kHz dead band above it - a lossy "
            f"transcode. Restore from just below the cliff."
        )
        verdict = "transcode"
    elif (nyquist - edge_hz) <= DEAD_BAND_HZ:
        cutoff = int(round((edge_hz - 1000.0) / 100.0) * 100)
        note = (
            f"Content runs to {edge_hz / 1000:.1f} kHz, essentially full "
            f"bandwidth for {sr / 1000:.1f} kHz. There is little missing to "
            f"restore; this is a better fine-tuning source than a restoration target."
        )
        verdict = "full-bandwidth"
    else:
        cutoff = int(round(edge_hz / 100.0) * 100)
        note = (
            f"Content fades gradually to {edge_hz / 1000:.1f} kHz with no cliff, "
            f"so this looks like a genuine master rather than a transcode."
        )
        verdict = "clean-fade"

    return max(CUTOFF_MIN_HZ, min(CUTOFF_MAX_HZ, cutoff)), verdict, note


def analyze_source(path: str) -> SourceAnalysis:
    """Measure a source file so the UI can pre-fill a cutoff instead of guessing."""
    try:
        info = sf.info(path)
        duration = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    except Exception:
        # Anything ffmpeg can decode but libsndfile cannot (mp3, m4a on old builds).
        duration = float(librosa.get_duration(path=path))
        y_probe, sr_probe = librosa.load(path, sr=None, mono=False, duration=0.1)
        sample_rate = int(sr_probe)
        channels = 1 if y_probe.ndim == 1 else int(y_probe.shape[0])

    y, sr = _load_for_analysis(path)
    edge_hz, shelf = spectral_scan(y, sr)
    cutoff, verdict, note = _suggest_cutoff(edge_hz, shelf, sr)

    return SourceAnalysis(
        path=path,
        name=os.path.basename(path),
        size_bytes=os.path.getsize(path),
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        hf_edge_hz=round(edge_hz, 1),
        shelf=shelf,
        verdict=verdict,
        note=note,
        suggested_cutoff_hz=cutoff,
    )


def _pool_time(matrix: np.ndarray, target_width: int) -> np.ndarray:
    """Reduce the time axis by max-pooling, which keeps transients visible."""
    frames = matrix.shape[1]
    if frames <= target_width:
        return matrix
    edges = np.linspace(0, frames, target_width + 1).astype(int)
    return np.stack(
        [matrix[:, edges[i] : max(edges[i] + 1, edges[i + 1])].max(axis=1) for i in range(target_width)],
        axis=1,
    )


def _pool_freq(matrix: np.ndarray, target_height: int) -> np.ndarray:
    bins = matrix.shape[0]
    if bins <= target_height:
        return matrix
    edges = np.linspace(0, bins, target_height + 1).astype(int)
    return np.stack(
        [matrix[edges[i] : max(edges[i] + 1, edges[i + 1]), :].max(axis=0) for i in range(target_height)],
        axis=0,
    )


def spectrogram_payload(path: str, max_seconds: Optional[float] = None) -> dict:
    """A linear-frequency STFT reduced to a uint8 grid the browser can draw.

    A linear axis is deliberate: a mel scale squashes 14-22 kHz into the top few
    bands, hiding exactly the region A2SB regenerates.
    """
    y, sr = librosa.load(path, sr=None, mono=True, duration=max_seconds)
    duration = float(len(y) / sr) if sr else 0.0

    n_fft = 2048
    hop_length = 512
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(spec, ref=np.max)

    db = _pool_time(db, SPECTROGRAM_WIDTH)
    db = _pool_freq(db, SPECTROGRAM_HEIGHT)

    normalized = np.clip((db - SPECTROGRAM_FLOOR_DB) / (0.0 - SPECTROGRAM_FLOOR_DB), 0.0, 1.0)
    # Row 0 is the top of the image, so flip to put high frequencies up top.
    grid = np.flipud((normalized * 255.0).astype(np.uint8))

    return {
        "width": int(grid.shape[1]),
        "height": int(grid.shape[0]),
        "sampleRate": int(sr),
        "maxFrequencyHz": float(sr / 2.0),
        "durationSec": duration,
        "floorDb": SPECTROGRAM_FLOOR_DB,
        "data": base64.b64encode(grid.tobytes()).decode("ascii"),
    }


def peak_envelope(path: str, buckets: int = 1600) -> dict:
    """Min/max envelope for the waveform display, computed server-side.

    Decoding a 45 MB WAV in the browser to draw a waveform is wasteful when the
    server already has librosa loaded.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0:
        return {"peaks": [], "durationSec": 0.0}

    buckets = max(1, min(buckets, y.size))
    edges = np.linspace(0, y.size, buckets + 1).astype(int)
    peaks: list[float] = []
    for i in range(buckets):
        window = y[edges[i] : max(edges[i] + 1, edges[i + 1])]
        peaks.append(round(float(np.max(np.abs(window))), 4))

    ceiling = max(peaks) or 1.0
    return {
        "peaks": [round(p / ceiling, 4) for p in peaks],
        "durationSec": float(y.size / sr) if sr else 0.0,
    }


def format_hz(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f} kHz"
    return f"{math.floor(value)} Hz"
