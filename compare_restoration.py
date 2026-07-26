#!/usr/bin/env python3
"""
Score a restoration against the untouched original.

The point of a lowpass-then-restore test is that the original IS the answer key:
everything the model invents above the cutoff has a correct value to be checked
against. That makes "energy above the cutoff" the wrong metric -- more energy is
only better up to the point where it matches, and a model that overshoots scores
well on a dB delta while sounding wrong. This reports spectral DISTANCE instead.

    python3 compare_restoration.py ORIGINAL.wav RESTORED.wav --cutoff 12000

Reported per band:
  band_rms_db : level of each file in the band (how much energy is there)
  lsd         : log-spectral distance, dB RMS error per time-frequency bin.
                Lower is better; 0 would be identical. This is the number to
                compare between two restorations of the same file.

The below-cutoff band is reported as a sanity check: it is passed through
untouched, so its LSD should be near zero. A large value there means the two
files are misaligned and the above-cutoff number cannot be trusted.
"""
from __future__ import annotations

import argparse
import sys

N_FFT = 4096
HOP = 1024


def load_aligned(path_a: str, path_b: str, np, librosa):
    """Load both files at a common sample rate, trimmed to a common length."""
    ya, sra = librosa.load(path_a, sr=None, mono=True)
    yb, srb = librosa.load(path_b, sr=None, mono=True)
    sr = min(sra, srb)
    if sra != sr:
        ya = librosa.resample(ya, orig_sr=sra, target_sr=sr)
    if srb != sr:
        yb = librosa.resample(yb, orig_sr=srb, target_sr=sr)
    n = min(len(ya), len(yb))
    if n == 0:
        raise SystemExit("one of the files is empty")
    return ya[:n], yb[:n], sr, sra, srb


def band_stats(Sa, Sb, freqs, lo, hi, np):
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return None
    a, b = Sa[sel], Sb[sel]
    ref = max(float(Sa.max()), float(Sb.max()), 1e-12)
    a_db = 20.0 * np.log10(np.maximum(a / ref, 1e-12))
    b_db = 20.0 * np.log10(np.maximum(b / ref, 1e-12))
    return {
        "a_rms_db": 20.0 * np.log10(max(float(np.sqrt((a ** 2).mean())), 1e-12) / ref),
        "b_rms_db": 20.0 * np.log10(max(float(np.sqrt((b ** 2).mean())), 1e-12) / ref),
        "lsd": float(np.sqrt(((a_db - b_db) ** 2).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original", help="untouched full-bandwidth file (the answer key)")
    ap.add_argument("restored", help="model output to score")
    ap.add_argument("--cutoff", type=float, default=12000.0,
                    help="lowpass cutoff used for the restoration (Hz)")
    args = ap.parse_args()

    try:
        import numpy as np
        import librosa
    except ImportError as e:
        print(f"ERROR: needs numpy + librosa ({e})", file=sys.stderr)
        return 2

    ya, yb, sr, sra, srb = load_aligned(args.original, args.restored, np, librosa)
    Sa = np.abs(librosa.stft(ya, n_fft=N_FFT, hop_length=HOP))
    Sb = np.abs(librosa.stft(yb, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    nyq = sr / 2.0

    print(f"original : {args.original}  ({sra} Hz)")
    print(f"restored : {args.restored}  ({srb} Hz)")
    print(f"compared at {sr} Hz, cutoff {args.cutoff:.0f} Hz\n")

    print(f"{'band':<22}{'original':>10}{'restored':>10}{'LSD':>9}")
    print("-" * 51)
    below = band_stats(Sa, Sb, freqs, 0.0, args.cutoff, np)
    if below:
        print(f"{'below cutoff (check)':<22}{below['a_rms_db']:>9.1f}dB"
              f"{below['b_rms_db']:>9.1f}dB{below['lsd']:>8.2f}")
    above = band_stats(Sa, Sb, freqs, args.cutoff, nyq, np)
    if above:
        print(f"{'ABOVE cutoff (scored)':<22}{above['a_rms_db']:>9.1f}dB"
              f"{above['b_rms_db']:>9.1f}dB{above['lsd']:>8.2f}")

    # Per-octave detail above the cutoff: a single number hides whether the model
    # tracks the original's roll-off or flattens it out.
    print(f"\n{'sub-band':<22}{'original':>10}{'restored':>10}{'LSD':>9}")
    print("-" * 51)
    lo = args.cutoff
    while lo < nyq:
        hi = min(lo + 2000.0, nyq)
        st = band_stats(Sa, Sb, freqs, lo, hi, np)
        if st:
            print(f"{f'{lo/1000:.0f}-{hi/1000:.0f} kHz':<22}{st['a_rms_db']:>9.1f}dB"
                  f"{st['b_rms_db']:>9.1f}dB{st['lsd']:>8.2f}")
        lo = hi

    if below and below["lsd"] > 6.0:
        print("\nWARNING: below-cutoff LSD is high. That band is passed through, "
              "so a large value means the files are misaligned (different trims "
              "or offsets) rather than differently restored. Some difference from "
              "resampling is normal; a big one makes the scored band untrustworthy.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
