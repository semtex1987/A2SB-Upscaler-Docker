"""Shared fixtures.

Every test runs against a private input/output tree so the suite never touches
`/app/inputs` or a developer's real staging directory. The environment has to be
set before `server.config` is imported, which is why this happens at module
import time rather than in a fixture.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SANDBOX = Path(tempfile.mkdtemp(prefix="a2sb-tests-"))
os.environ.setdefault("A2SB_INPUT_DIR", str(_SANDBOX / "inputs"))
os.environ.setdefault("A2SB_OUTPUT_DIR", str(_SANDBOX / "outputs"))

SAMPLE_RATE = 44100


def _broadband(seconds: float = 4.0, seed: int = 11) -> np.ndarray:
    """Pink-ish noise plus a harmonic series: a stand-in for real music.

    White noise is a poor fixture here. Its flat spectrum makes any cutoff
    trivially detectable, which would let a broken detector pass.
    """
    count = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, count))
    freqs = np.fft.rfftfreq(count, 1 / SAMPLE_RATE)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, n=count)
    pink /= np.max(np.abs(pink))

    t = np.arange(count) / SAMPLE_RATE
    tones = sum(
        np.sin(2 * np.pi * freq * t) / (index + 1)
        for index, freq in enumerate([110, 220, 440, 880, 1760, 3520, 7040])
    )
    tones /= np.max(np.abs(tones))

    mixed = 0.65 * pink + 0.35 * tones
    return (mixed / np.max(np.abs(mixed)) * 0.85).astype(np.float32)


def brickwalled(cutoff_hz: float, seconds: float = 4.0) -> np.ndarray:
    """Broadband material with a steep lowpass, i.e. a lossy transcode."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(12, cutoff_hz, btype="low", fs=SAMPLE_RATE, output="sos")
    filtered = sosfiltfilt(sos, _broadband(seconds))
    return (filtered / np.max(np.abs(filtered)) * 0.85).astype(np.float32)


def full_bandwidth(seconds: float = 4.0) -> np.ndarray:
    return _broadband(seconds)


def write_wav(path: Path, samples: np.ndarray, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 2:
        samples = np.stack([samples, np.roll(samples, 137)], axis=1)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def sandbox() -> Path:
    return _SANDBOX


@pytest.fixture(scope="session")
def input_dir() -> Path:
    from server.config import INPUT_DIR

    return INPUT_DIR


@pytest.fixture(scope="session")
def transcode_wav(input_dir: Path) -> Path:
    return write_wav(input_dir / "transcode.wav", brickwalled(11000), channels=2)


@pytest.fixture(scope="session")
def master_wav(input_dir: Path) -> Path:
    return write_wav(input_dir / "master.wav", full_bandwidth())
