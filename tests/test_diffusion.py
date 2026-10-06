"""Multidiffusion padding and inverse-STFT length, without running the model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

VENDOR = Path(__file__).resolve().parent.parent / "nvidia-a2sb-original-repo"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from diffusion import multidiffusion_pad_inputs, multidiffusion_unpad_outputs


def test_short_spectrogram_is_padded_to_the_full_window():
    """An 87-frame input for a 256-frame window used to pad by only 87 frames."""
    x = torch.randn(1, 3, 8, 87)
    padded = multidiffusion_pad_inputs(x, win_length=256, hop_length=128)
    assert padded.shape[-1] == 256
    assert torch.equal(padded[..., :87], x)
    restored = multidiffusion_unpad_outputs(padded, 87)
    assert restored.shape[-1] == 87
    assert torch.equal(restored, x)


def test_padding_constant_fills_the_exact_tail():
    x = torch.ones(1, 1, 4, 87)
    padded = multidiffusion_pad_inputs(x, win_length=256, hop_length=128, padding_constant=0)
    assert padded.shape[-1] == 256
    assert torch.all(padded[..., 87:] == 0)


def test_already_window_sized_input_is_unchanged():
    x = torch.randn(1, 2, 4, 256)
    padded = multidiffusion_pad_inputs(x, win_length=256, hop_length=128)
    assert padded.shape[-1] == 256
    assert torch.equal(padded, x)


def test_padding_past_one_window_lands_on_a_hop_boundary():
    x = torch.randn(1, 3, 8, 300)
    padded = multidiffusion_pad_inputs(x, win_length=256, hop_length=128)
    extra = padded.shape[-1] - 256
    assert extra % 128 == 0
    assert padded.shape[-1] >= 300
    assert torch.equal(padded[..., :300], x)


@pytest.mark.skipif(
    importlib.util.find_spec("torchaudio") is None,
    reason="torchaudio is required for inverse STFT",
)
def test_inverse_stft_keeps_the_original_sample_count():
    from audio_transforms.transforms import ComplexSpectrogram, InverseComplexSpectrogram

    n_fft = 2048
    hop = 512
    length = 44100
    waveform = torch.randn(length)
    spec = ComplexSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop)(waveform)
    inverse = InverseComplexSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop)
    truncated = inverse(spec)
    restored = inverse(spec, length=length)
    assert truncated.numel() < length
    assert restored.numel() == length


def test_istft_length_argument_keeps_the_hop_tail():
    n_fft = 2048
    hop = 512
    length = 44100
    waveform = torch.randn(length)
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True
    )
    truncated = torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window)
    restored = torch.istft(
        spec, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, length=length
    )
    assert truncated.numel() < length
    assert restored.numel() == length
