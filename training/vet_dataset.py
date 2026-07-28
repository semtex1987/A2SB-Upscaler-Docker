#!/usr/bin/env python3
"""
Vet a container folder of audio sub-directories before adding them to the A2SB
fine-tuning dataset.

The script walks the given root recursively.  For every sub-directory that
directly contains at least one audio file it:
  1. Analyses each audio file in that directory.
  2. Writes a report.csv (name configurable via --report-name) into that
     directory so the results stay with the material.
  3. Prints a per-folder summary block to stdout.

At the end a grand-total summary is printed.

WHAT THIS GATES ON (default = authenticity mode)
------------------------------------------------
The goal is NOT "loudest / brightest" -- it is "a genuine master, free of
artificial-upscaling artifacts".  A bandwidth-extension model is poisoned by
FAKE high-frequency content (lossy transcodes, CD upsampled to "hi-res", DSD
noise-shaping hash), so those are what we reject.  Genuine material is kept
whatever its natural bandwidth: an intimate folk record that honestly rolls off
at 17 kHz is authentic and useful; a metal track upsampled from a 320k MP3 is
not, even though it "reaches" 20 kHz.  How bright a keeper is (its hf_edge) is
reported so you can BALANCE set composition -- it is not a gate.

Per-file columns in each report.csv:
  native_sr   : the file's real sample rate.
  est_true_sr : the SAME value training/finetune.py::estimate_true_sr computes
                (2x the 95th-percentile spectral rolloff, capped at 44100).
                Files below 32000 are dropped by the apply_sr_loss_mask filter
                during training; trainer_gate reflects that.
  rolloff95   : the underlying 95th-percentile rolloff frequency (Hz).
  hf_edge     : highest frequency carrying real energy (Hz), measured from the
                95th-PERCENTILE-over-time spectrum (not the mean).  The
                percentile captures intermittent/transient HF -- pick attacks,
                brushes, cymbals -- that a mean spectrum averages away, so sparse
                acoustic material (bluegrass, fingerpicked folk) is judged
                fairly instead of being wrongly called band-limited.
  shelf       : "Y" if there is a brickwall cliff at hf_edge (a drop of >40 dB
                over <1 kHz).  A genuine recording fades; a cliff is the
                signature of an artificial lowpass.
  trainer_gate: "keep" / "DROP" -- what the trainer's own 16 kHz gate will do.
  verdict     : AUTHENTIC -> genuine master, no artificial-upscaling artifact.
                            Keep.  (Check hf_edge to balance bright vs mellow.)
                ARTIFACT  -> brickwall cliff with a dead band below Nyquist:
                            lossy transcode or CD-to-hi-res upsample.  Discard.
                CHECK     -> ambiguous: e.g. a very high sample rate that may be
                            DSD-sourced (downsample and re-vet), or a borderline
                            case worth a spectrogram eyeball.
  note        : short human-readable reason for the verdict.

Pass --strict to restore the older bandwidth-threshold verdict instead
(PASS >=20.5 kHz / CHECK 17-20.5 kHz / REJECT <17 kHz), applied to the improved
hf_edge.  Useful when you deliberately want only the brightest material.

NOTE: numpy + librosa must be installed where you RUN this. It is meant for your
audio-staging machine. The trainer container already has these deps, so you can
also run it there against the mounted data dir, e.g.:

    docker compose -f training/docker-compose.train.yml run trainer \
        python /app/training/vet_dataset.py /data/training_data
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

GATE_HZ = 32000        # matches apply_sr_loss_mask exclusion in finetune.py
EST_LOAD_SEC = 60.0    # matches estimate_true_sr() window in finetune.py
ANALYSIS_SEC = 180.0   # window for the hf-edge spectral scan

# --- authenticity-mode thresholds -----------------------------------------
SHELF_DROP_DB = 30.0   # a drop steeper than this over ~1 kHz counts as a cliff
DEAD_PLATEAU_DB = 25.0 # how far the whole band above a cliff must sit below it
SMOOTH_HZ = 430.0      # median window for the spectral envelope
CONTENT_RANGE_DB = 60.0  # with no cliff, content counts down to this below peak
DEAD_BAND_HZ = 2000    # a cliff this far below Nyquist means an artificial LPF
HIRATE_HZ = 96000      # above this, ultrasonic band may be DSD hash -> CHECK
UPSAMPLE_CUTOFF_HZ = 24000  # a cliff at/under this in a >48 kHz file = upsample

# --- strict-mode (legacy) thresholds --------------------------------------
REJECT_HZ = 17000      # below this -> REJECT
PASS_HZ = 20500        # at/above this -> PASS


def find_audio_dirs(root: Path) -> list[Path]:
    """Return every directory under *root* that directly contains audio files."""
    dirs: set[Path] = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            dirs.add(p.parent)
    return sorted(dirs)


def find_audio_in_dir(folder: Path) -> list[Path]:
    """Return audio files that live directly inside *folder* (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def estimate_true_sr(y, sr, np, librosa) -> tuple[int, float]:
    """Identical logic to training/finetune.py::estimate_true_sr (first 60 s,
    95th percentile of per-frame 0.99 rolloff, doubled, capped at 44100)."""
    seg = y[: int(EST_LOAD_SEC * sr)] if len(y) > int(EST_LOAD_SEC * sr) else y
    # ⚡ Bolt: Increase hop_length/n_fft to avoid default 75% overlap overhead
    rolloff_frames = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.99, n_fft=2048, hop_length=2048)
    rolloff = float(np.percentile(rolloff_frames, 95))
    return int(min(2 * rolloff, 44100)), rolloff


def median_smooth(values, np, window):
    """Running median with edge padding.  Kept in sync with server/analysis.py."""
    if window <= 1 or values.size < window:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def spectral_scan(y, sr, np, librosa) -> tuple[float, bool]:
    """Highest frequency carrying real energy, and whether the spectrum cliffs
    there (brickwall = artificial-lowpass signature).

    The scan looks for a cliff first and only falls back to a level threshold
    when there is none.  Thresholding first does not work: a level-based floor
    assumes the dead band is the quietest part of the spectrum, which holds for
    a transcode but not for ordinary material whose spectrum already slopes
    60 dB from bass to Nyquist.  Against real music that approach places the
    "edge" somewhere in the mid-band.

    The spectrum is the 95th-PERCENTILE-over-time rather than the mean:
    transient HF (pick attacks, brushes, cymbals) is intermittent and a mean
    spectrum averages it below the noise floor, which unfairly rejects sparse
    acoustic material.  The percentile keeps that energy visible.
    """
    n_fft = 4096
    # ⚡ Bolt: 50% overlap for macroscopic aggregate (hop=2048) rather than default 75%
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=2048))
    p95 = np.percentile(spec, 95, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak = float(p95.max())
    if peak <= 0:
        return 0.0, False
    p95_db = 20.0 * np.log10(np.maximum(p95 / peak, 1e-12))

    bin_width = sr / n_fft
    # Median rather than mean: a dead band is not silent, it is dither and
    # coding noise whose per-bin level swings +/-10 dB.  A mean smooths that
    # into a slope; a median flattens it into the plateau the cliff test needs.
    curve = median_smooth(p95_db, np, max(3, int(round(SMOOTH_HZ / bin_width)) | 1))

    look = max(1, int(round(500.0 / bin_width)))  # ~500 Hz each side
    start_bin = int(3000.0 / bin_width)
    if curve.size > 2 * look + 1 and start_bin < curve.size - 2 * look:
        drops = curve[: -2 * look] - curve[2 * look :]
        offset = int(np.argmax(drops[start_bin:])) + start_bin
        cliff_bin, cliff_drop = offset + look, float(drops[offset])

        if cliff_drop > SHELF_DROP_DB:
            pre = float(np.median(curve[max(0, cliff_bin - 3 * look) : cliff_bin - look + 1]))
            post = float(np.median(curve[min(curve.size - 1, cliff_bin + look) :]))
            # Everything above the cliff must stay down, so that a local dip
            # (a notch, a crossover null) does not read as a brickwall.
            if pre - post > DEAD_PLATEAU_DB:
                shoulder = np.where(curve[: cliff_bin + 1] >= pre - 6.0)[0]
                edge_bin = int(shoulder.max()) if len(shoulder) else cliff_bin
                return float(freqs[edge_bin]), True

    # No cliff: report where content fades out relative to the loudest band.
    above = np.where(curve > curve.max() - CONTENT_RANGE_DB)[0]
    if len(above) == 0:
        return 0.0, False
    return float(freqs[int(above.max())]), False


def classify(sr, est_sr, edge_hz, shelf, strict) -> tuple[str, str]:
    """Return (verdict, note).

    strict=True   -> legacy bandwidth thresholds on hf_edge.
    strict=False  -> authenticity mode: reject artificial-upscaling artifacts,
                     keep genuine masters at whatever bandwidth the genre has.
    """
    if strict:
        if edge_hz >= PASS_HZ:
            return "PASS", ""
        if edge_hz >= REJECT_HZ:
            return "CHECK", ""
        return "REJECT", ""

    nyq = sr / 2.0

    # 1) Very high sample rate: the ultrasonic band may be DSD noise-shaping hash
    #    masquerading as content. Can't be trusted until decimated to PCM.
    if sr > HIRATE_HZ:
        return "CHECK", f"high-rate {int(sr/1000)}k: downsample (DSD-hash risk) & re-vet"

    # 2) Brickwall cliff with a real dead band below Nyquist = artificial lowpass
    #    (lossy transcode, or a lower-rate master upsampled to a higher rate).
    if shelf and (nyq - edge_hz) > DEAD_BAND_HZ:
        if sr > 48000 and edge_hz <= UPSAMPLE_CUTOFF_HZ:
            kind = "upsample"
        else:
            kind = "transcode"
        return "ARTIFACT", (
            f"{kind}-cliff@{edge_hz/1000:.1f}k (dead band {(nyq - edge_hz)/1000:.0f}k)"
        )

    # 3) Genuine gradual fade -> authentic to the genre, keep it.
    note = f"clean fade to {edge_hz/1000:.1f}k"
    if est_sr < GATE_HZ:
        note += "; trainer-drops"
    return "AUTHENTIC", note


def probe_sr(path: Path):
    """Read the sample rate from the header without decoding audio.
    Returns an int, or None if it can't be determined cheaply."""
    try:
        import soundfile as sf
        return int(sf.info(str(path)).samplerate)
    except Exception:  # noqa: BLE001 - fall back to the full decode path
        return None


def analyze(path: Path, np, librosa, strict: bool) -> dict:
    # Short-circuit DSD-rate material: at these rates the ultrasonic band is
    # almost certainly noise-shaping hash, the verdict is CHECK regardless, and a
    # full-length STFT at 352.8 kHz is painfully slow -- so skip the decode.
    if not strict:
        sr0 = probe_sr(path)
        if sr0 is not None and sr0 > HIRATE_HZ:
            return {
                "file": str(path), "native_sr": sr0, "est_true_sr": "",
                "rolloff95": "", "hf_edge": "", "shelf": "", "trainer_gate": "",
                "verdict": "CHECK",
                "note": f"high-rate {int(sr0/1000)}k: downsample (DSD-hash risk) & re-vet",
            }

    y, sr = librosa.load(str(path), sr=None, mono=True, duration=ANALYSIS_SEC)
    if y.size == 0:
        raise ValueError("empty / unreadable audio")
    est_sr, rolloff = estimate_true_sr(y, sr, np, librosa)
    edge_hz, shelf = spectral_scan(y, sr, np, librosa)
    v, note = classify(sr, est_sr, edge_hz, shelf, strict)
    return {
        "file": str(path),
        "native_sr": sr,
        "est_true_sr": est_sr,
        "rolloff95": int(rolloff),
        "hf_edge": int(edge_hz),
        "shelf": "Y" if shelf else "N",
        "trainer_gate": "keep" if est_sr >= GATE_HZ else "DROP",
        "verdict": v,
        "note": note,
    }


CSV_FIELDS = [
    "file", "native_sr", "est_true_sr", "rolloff95",
    "hf_edge", "shelf", "trainer_gate", "verdict", "note",
]


def process_folder(
    folder: Path,
    report_name: str,
    strict: bool,
    np,
    librosa,
) -> Counter:
    """Analyse all audio files directly inside *folder*, write a report CSV there,
    and return a Counter of verdicts (plus 'total') for grand-total accumulation."""
    counts: Counter = Counter()
    files = find_audio_in_dir(folder)
    if not files:
        return counts

    name_w = min(50, max(len(p.name) for p in files))
    header = (f"{'file':<{name_w}}  {'edge':>6}  {'roll95':>6}  "
              f"{'est_sr':>6}  {'shelf':>5}  {'gate':>4}  {'verdict':<9}  note")

    print(f"\n{'='*len(header)}")
    print(f"  {folder}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for p in files:
        try:
            r = analyze(p, np, librosa, strict)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the scan
            counts["ERROR"] += 1
            print(f"{p.name[:name_w]:<{name_w}}  {'--':>6}  {'--':>6}  "
                  f"{'--':>6}  {'--':>5}  {'--':>4}  {'ERROR':<9}  {e}")
            rows.append({
                "file": p.name, "native_sr": "", "est_true_sr": "",
                "rolloff95": "", "hf_edge": "", "shelf": "",
                "trainer_gate": "", "verdict": "ERROR", "note": str(e),
            })
            continue
        counts[r["verdict"]] += 1
        print(f"{p.name[:name_w]:<{name_w}}  {r['hf_edge']:>6}  {r['rolloff95']:>6}  "
              f"{r['est_true_sr']:>6}  {r['shelf']:>5}  {r['trainer_gate']:>4}  "
              f"{r['verdict']:<9}  {r['note'][:34]}")
        rows.append({**r, "file": p.name})

    print("-" * len(header))
    print("  |  ".join(f"{k} {counts[k]}" for k in _verdict_order(counts))
          + f"  (of {len(files)} files)")

    report_path = folder / report_name
    try:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  -> wrote {report_path}")
    except OSError as e:
        # A network share can blip mid-run; don't lose the whole scan over one
        # folder we couldn't write. Report it and carry on.
        print(f"  !! could not write {report_path}: {e}", file=sys.stderr)

    counts["total"] = len(files)
    return counts


def _verdict_order(counts: Counter) -> list[str]:
    """Stable, human-friendly ordering of whatever verdict labels are present."""
    preferred = ["AUTHENTIC", "PASS", "CHECK", "ARTIFACT", "REJECT", "ERROR"]
    present = [k for k in preferred if k in counts and k != "total"]
    extra = [k for k in counts if k not in preferred and k != "total"]
    return present + sorted(extra)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recursively vet audio sub-directories for authentic, "
            "artifact-free content, leaving a report CSV in each folder that "
            "contains audio files."
        )
    )
    ap.add_argument(
        "root",
        type=Path,
        help="Container directory to walk. A report.csv is written into every "
             "sub-directory (at any depth) that directly contains audio files.",
    )
    ap.add_argument(
        "--report-name",
        default="report.csv",
        metavar="FILENAME",
        help="Name of the CSV report written into each folder (default: report.csv).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Use the legacy bandwidth-threshold verdict (PASS >=20.5 kHz / "
             "CHECK 17-20.5 kHz / REJECT <17 kHz) instead of authenticity mode. "
             "Keeps only the brightest material.",
    )
    args = ap.parse_args()

    try:
        import numpy as np
        import librosa
    except ImportError as e:
        print(f"ERROR: this script needs numpy + librosa ({e}).", file=sys.stderr)
        return 2

    if not args.root.is_dir():
        print(f"ERROR: not a directory: {args.root}", file=sys.stderr)
        return 2

    audio_dirs = find_audio_dirs(args.root)
    if not audio_dirs:
        print(f"No audio files found under {args.root}", file=sys.stderr)
        return 1

    mode = "strict bandwidth" if args.strict else "authenticity"
    print(f"Found {len(audio_dirs)} folder(s) with audio files under {args.root}")
    print(f"Mode: {mode}")

    grand: Counter = Counter()
    for folder in audio_dirs:
        counts = process_folder(folder, args.report_name, args.strict, np, librosa)
        grand.update(counts)

    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    print(f"Folders scanned : {len(audio_dirs)}")
    print(f"Files analysed  : {grand.get('total', 0)}")
    for k in _verdict_order(grand):
        print(f"  {k:<10} {grand[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
