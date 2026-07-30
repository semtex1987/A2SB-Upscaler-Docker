"""Source inspection: what bandwidth a file really has, and spectrogram data.

`spectral_scan` here and in `training/vet_dataset.py` answer the same question
and are kept in sync, so the cutoff this UI suggests agrees with what the
dataset vetting tool reports.
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
from server.serialization import camelize

#: A drop steeper than this across ~1 kHz counts as a brickwall cliff.
SHELF_DROP_DB = 30.0
#: How far the whole band above a cliff must sit below the band under it. Stops
#: a narrow notch from being read as a bandwidth limit.
DEAD_PLATEAU_DB = 25.0
#: Median window for the spectral envelope, wide enough to erase per-bin noise
#: scatter and narrow enough to keep a brickwall transition steep.
SMOOTH_HZ = 430.0
#: With no cliff, content is considered present down to this level below the
#: loudest band.
CONTENT_RANGE_DB = 60.0
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
        return camelize(asdict(self))


def _load_for_analysis(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=None, mono=True, duration=ANALYSIS_SEC)
    return y, int(sr)


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Running median with edge padding. Kept in sync with `vet_dataset.py`."""
    if window <= 1 or values.size < window:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def spectral_scan(y: np.ndarray, sr: int) -> tuple[float, bool]:
    """Highest frequency carrying real energy, and whether the spectrum cliffs there.

    The scan looks for a brickwall cliff first and only falls back to a level
    threshold when there is none. Thresholding first does not work: a
    level-based floor assumes the dead band is the quietest part of the
    spectrum, which holds for a transcode but not for ordinary material whose
    spectrum already slopes 60 dB from bass to Nyquist. Against real music that
    approach places the "edge" somewhere in the mid-band.

    The spectrum is the 95th percentile over time rather than the mean:
    transient HF (pick attacks, brushes, cymbals) is intermittent, and a mean
    spectrum averages it below the noise floor, which unfairly judges sparse
    acoustic material as band-limited.
    """
    n_fft = 4096
    # ⚡ Bolt: Reduce STFT overlap to 50% for 2x speedup on macroscopic heuristic
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=2048))
    p95 = np.percentile(spec, 95, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak = float(p95.max())
    if peak <= 0:
        return 0.0, False
    p95_db = 20.0 * np.log10(np.maximum(p95 / peak, 1e-12))

    bin_width = sr / n_fft
    # Median rather than mean: a dead band is not silent, it is dither and
    # coding noise whose per-bin level swings ±10 dB. A mean smooths that into
    # a slope; a median flattens it into the plateau the cliff test needs.
    curve = median_smooth(p95_db, max(3, int(round(SMOOTH_HZ / bin_width)) | 1))

    look = max(1, int(round(500.0 / bin_width)))
    cliff_bin, cliff_drop = _steepest_drop(curve, look, start_bin=int(3000.0 / bin_width))

    if cliff_bin is not None and cliff_drop > SHELF_DROP_DB:
        pre = float(np.median(curve[max(0, cliff_bin - 3 * look) : cliff_bin - look + 1]))
        post = float(np.median(curve[min(len(curve) - 1, cliff_bin + look) :]))
        # Everything above the cliff must stay down. Without this a steep but
        # local dip (a notch, a crossover null) would read as a brickwall.
        if pre - post > DEAD_PLATEAU_DB:
            shoulder = np.where(curve[: cliff_bin + 1] >= pre - 6.0)[0]
            edge_bin = int(shoulder.max()) if len(shoulder) else cliff_bin
            return float(freqs[edge_bin]), True

    # No cliff: report where content fades out relative to the loudest band.
    above = np.where(curve > curve.max() - CONTENT_RANGE_DB)[0]
    if len(above) == 0:
        return 0.0, False
    return float(freqs[int(above.max())]), False


def _steepest_drop(curve: np.ndarray, look: int, start_bin: int) -> tuple[Optional[int], float]:
    """Bin with the largest fall across ±`look` bins, ignoring the low end."""
    if curve.size <= 2 * look + 1:
        return None, 0.0
    drops = curve[: -2 * look] - curve[2 * look :]
    first = max(start_bin, 0)
    if first >= drops.size:
        return None, 0.0
    offset = int(np.argmax(drops[first:])) + first
    return offset + look, float(drops[offset])


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
