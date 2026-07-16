#!/usr/bin/env python3
"""
Vet a container folder of audio sub-directories for genuine full-spectrum
content before adding them to the A2SB fine-tuning dataset.

The script walks the given root recursively.  For every sub-directory that
directly contains at least one audio file it:
  1. Analyses each audio file in that directory.
  2. Writes a report.csv (name configurable via --report-name) into that
     directory so the results stay with the material.
  3. Prints a per-folder summary block to stdout.

At the end a grand-total summary is printed.

Per-file columns in each report.csv:
  est_true_sr : the SAME value training/finetune.py::estimate_true_sr computes
                (2x the 95th-percentile spectral rolloff, capped at 44100).
                Files below 32000 are dropped by the apply_sr_loss_mask filter
                during training -- this column tells you what the trainer will do.
  rolloff95   : the underlying 95th-percentile rolloff frequency (Hz).
  hf_edge     : highest frequency carrying real energy (Hz), measured against
                each file's own noise floor. This is what exposes lossy
                transcodes wearing a .flac extension: a true master fades
                toward ~21-22 kHz, a transcode shows a hard edge at 16/19/20 kHz.
  shelf       : "Y" if there is a steep cliff at hf_edge (a drop of >40 dB over
                <1 kHz) -- the signature of a brickwall lowpass, i.e. a transcode
                or aggressively filtered master. A genuine recording fades.
  verdict     : PASS  -> real energy to >=20.5 kHz; keep it.
                CHECK -> edge between 17 and 20.5 kHz; eyeball the spectrogram.
                         (gentle LPF on a genuine master vs a 320k transcode
                         both land here -- the shelf flag helps you decide.)
                REJECT-> edge below 17 kHz; band-limited, low value for training.

The verdict bar is intentionally STRICTER than the trainer's 16 kHz gate: the
goal here is to keep only genuinely full-spectrum material, not merely whatever
survives the loss mask.

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
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

GATE_HZ = 32000        # matches apply_sr_loss_mask exclusion in finetune.py
EST_LOAD_SEC = 60.0    # matches estimate_true_sr() window in finetune.py
ANALYSIS_SEC = 180.0   # window for the hf-edge spectral scan
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


def hf_edge(y, sr, np, librosa) -> tuple[float, bool]:
    """Highest frequency above the file's own noise floor, and whether the
    spectrum cliffs there (brickwall = transcode signature)."""
    n_fft = 4096
    # ⚡ Bolt: Use 50% overlap (hop_length=n_fft//2) instead of default 75% (n_fft//4) for heuristic aggregation to double speed without introducing blind spots.
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=2048))
    mean_mag = spec.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak = float(mean_mag.max())
    if peak <= 0:
        return 0.0, False
    mean_db = 20.0 * np.log10(np.maximum(mean_mag / peak, 1e-12))

    # Adaptive floor: the file's own quietest bins. For a transcode the dead
    # region above the cutoff sits at this floor; for a genuine file, HF energy
    # stays above it up toward Nyquist. This avoids wrongly rejecting mellow-but-
    # full-bandwidth recordings whose HF is quiet but present.
    floor = float(np.percentile(mean_db, 10))
    above = np.where(mean_db > floor + 6.0)[0]
    if len(above) == 0:
        return 0.0, False
    edge_bin = int(above.max())
    edge_hz = float(freqs[edge_bin])

    # Steep-cliff (brickwall) check around the edge.
    bin_width = sr / n_fft
    look = max(1, int(round(500.0 / bin_width)))  # ~500 Hz each side
    lo = max(0, edge_bin - look)
    hi = min(len(mean_db) - 1, edge_bin + look)
    drop = float(mean_db[lo] - mean_db[hi])
    return edge_hz, drop > 40.0


def verdict(edge_hz: float) -> str:
    if edge_hz < REJECT_HZ:
        return "REJECT"
    if edge_hz >= PASS_HZ:
        return "PASS"
    return "CHECK"


def analyze(path: Path, np, librosa) -> dict:
    y, sr = librosa.load(str(path), sr=None, mono=True, duration=ANALYSIS_SEC)
    if y.size == 0:
        raise ValueError("empty / unreadable audio")
    est_sr, rolloff = estimate_true_sr(y, sr, np, librosa)
    edge_hz, shelf = hf_edge(y, sr, np, librosa)
    return {
        "file": str(path),
        "native_sr": sr,
        "est_true_sr": est_sr,
        "rolloff95": int(rolloff),
        "hf_edge": int(edge_hz),
        "shelf": "Y" if shelf else "N",
        "trainer_gate": "keep" if est_sr >= GATE_HZ else "DROP",
        "verdict": verdict(edge_hz),
    }


CSV_FIELDS = [
    "file", "native_sr", "est_true_sr", "rolloff95",
    "hf_edge", "shelf", "trainer_gate", "verdict",
]


def process_folder(
    folder: Path,
    report_name: str,
    np,
    librosa,
) -> dict[str, int]:
    """Analyse all audio files directly inside *folder*, write a report CSV there,
    and return a counts dict for grand-total accumulation."""
    files = find_audio_in_dir(folder)
    if not files:
        return {"PASS": 0, "CHECK": 0, "REJECT": 0, "ERROR": 0, "total": 0}

    name_w = min(60, max(len(p.name) for p in files))
    header = (f"{'file':<{name_w}}  {'edge':>6}  {'roll95':>6}  "
              f"{'est_sr':>6}  {'shelf':>5}  {'gate':>4}  verdict")

    print(f"\n{'='*len(header)}")
    print(f"  {folder}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    counts: dict[str, int] = {"PASS": 0, "CHECK": 0, "REJECT": 0, "ERROR": 0}

    for p in files:
        try:
            r = analyze(p, np, librosa)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the scan
            counts["ERROR"] += 1
            print(f"{p.name[:name_w]:<{name_w}}  {'--':>6}  {'--':>6}  "
                  f"{'--':>6}  {'--':>5}  {'--':>4}  ERROR: {e}")
            rows.append({
                "file": p.name, "native_sr": "", "est_true_sr": "",
                "rolloff95": "", "hf_edge": "", "shelf": "",
                "trainer_gate": "", "verdict": f"ERROR: {e}",
            })
            continue
        counts[r["verdict"]] += 1
        print(f"{p.name[:name_w]:<{name_w}}  {r['hf_edge']:>6}  {r['rolloff95']:>6}  "
              f"{r['est_true_sr']:>6}  {r['shelf']:>5}  {r['trainer_gate']:>4}  "
              f"{r['verdict']}")
        rows.append({**r, "file": p.name})

    print("-" * len(header))
    print(f"PASS {counts['PASS']}  |  CHECK {counts['CHECK']}  |  "
          f"REJECT {counts['REJECT']}  |  ERROR {counts['ERROR']}  "
          f"(of {len(files)} files)")

    report_path = folder / report_name
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> wrote {report_path}")

    counts["total"] = len(files)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recursively vet audio sub-directories for genuine full-spectrum "
            "content, leaving a report CSV in each folder that contains audio files."
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

    print(f"Found {len(audio_dirs)} folder(s) with audio files under {args.root}")

    grand: dict[str, int] = {"PASS": 0, "CHECK": 0, "REJECT": 0, "ERROR": 0, "total": 0}
    for folder in audio_dirs:
        counts = process_folder(folder, args.report_name, np, librosa)
        for key in grand:
            grand[key] += counts.get(key, 0)

    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    print(f"Folders scanned : {len(audio_dirs)}")
    print(f"Files analysed  : {grand['total']}")
    print(f"  PASS   {grand['PASS']}")
    print(f"  CHECK  {grand['CHECK']}")
    print(f"  REJECT {grand['REJECT']}")
    print(f"  ERROR  {grand['ERROR']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
