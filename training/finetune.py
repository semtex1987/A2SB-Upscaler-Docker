#!/usr/bin/env python3
"""
A2SB fine-tuning orchestration: scan a directory for audio, build a training
manifest, and run fine-tuning for one or both time-split checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# Segment length and sample rate must match the dataset config (130560 @ 44100 ≈ 2.96s)
SEGMENT_LENGTH = 130560
SAMPLING_RATE = 44100
MIN_DURATION_SEC = SEGMENT_LENGTH / SAMPLING_RATE

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# NVIDIA release checkpoints (inside container)
CKPT_SPLIT_1 = "/app/ckpts/A2SB_twosplit_0.0_0.5_release.ckpt"
CKPT_SPLIT_2 = "/app/ckpts/A2SB_twosplit_0.5_1.0_release.ckpt"

# Config and script paths inside container
APP_ROOT = Path("/app")
TRAINING_DIR = APP_ROOT / "training"
CONFIG_SPLIT_1 = TRAINING_DIR / "configs" / "finetune_split1.yaml"
CONFIG_SPLIT_2 = TRAINING_DIR / "configs" / "finetune_split2.yaml"
MAIN_PY = APP_ROOT / "main.py"


def get_duration(path: str) -> float | None:
    # ⚡ Bolt Optimization: Use soundfile for O(1) duration lookup instead of librosa's O(N) decoding
    """
    Obtain the duration of an audio file in seconds.
    
    Returns:
        float: Duration in seconds if the file's duration can be determined, `None` otherwise.
    """
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        # Fallback to librosa (which uses audioread for mp3/m4a if soundfile fails)
        try:
            import librosa
            return float(librosa.get_duration(path=path))
        except Exception:
            return None


def find_audio_files(data_dir: Path) -> list[Path]:
    """
    Recursively locate audio files under the given directory that match the configured audio extensions.
    
    Parameters:
        data_dir (Path): Root directory to search. If `data_dir` is not a directory, an empty list is returned.
    
    Returns:
        list[Path]: Sorted list of file paths matching the known audio extensions (AUDIO_EXTENSIONS).
    """
    out: list[Path] = []
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        return out
    for f in data_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            out.append(f)
    return sorted(out)


def build_manifest(
    data_dir: Path,
    output_dir: Path,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Path:
    """Scan data_dir for audio, compute durations, write manifest CSV. Returns path to manifest."""
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(data_dir)
    if not files:
        raise SystemExit(f"No audio files found under {data_dir} (extensions: {AUDIO_EXTENSIONS})")

    # (path, duration); skip files too short to yield at least one segment
    rows: list[tuple[str, float]] = []
    for f in files:
        d = get_duration(str(f))
        if d is None:
            print(f"  skip (unreadable): {f.name}", file=sys.stderr)
            continue
        if d < MIN_DURATION_SEC:
            print(f"  skip (too short {d:.1f}s < {MIN_DURATION_SEC:.1f}s): {f.name}", file=sys.stderr)
            continue
        rows.append((str(f), d))

    if not rows:
        raise SystemExit("No valid audio files (readable and long enough).")

    random.seed(seed)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    n_train = len(rows) - n_val
    train_rows = rows[:n_train]
    val_rows = rows[n_train:]

    manifest_path = output_dir / "finetune_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"')
        w.writerow(["split", "filepath", "duration"])
        for path, dur in train_rows:
            w.writerow(["train", path, f"{dur:.4f}"])
        for path, dur in val_rows:
            w.writerow(["validation", path, f"{dur:.4f}"])

    print(f"Manifest: {len(train_rows)} train, {len(val_rows)} validation -> {manifest_path}")
    return manifest_path


def run_fit(
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float | None,
    extra_args: list[str],
) -> None:
    """Run main.py fit with the given config and checkpoint."""
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "fit",
        "-c", str(config_path),
        "--ckpt_path", str(ckpt_path),
        "--trainer.max_steps", str(max_steps),
        "--trainer.default_root_dir", str(output_dir),
        "--data.batch_size", str(batch_size),
    ]
    if learning_rate is not None:
        cmd += ["--model.learning_rate", str(learning_rate)]
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(APP_ROOT), check=True)


def latest_ckpt_in_dir(d: Path) -> Path | None:
    """Return path to the latest checkpoint in d (by mtime), or None."""
    if not d.is_dir():
        return None
    ckpts = list(d.glob("*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def copy_final_checkpoints(
    split_output_dir: Path,
    dest_dir: Path,
    name: str,
) -> None:
    """Copy the latest checkpoint from split_output_dir to dest_dir with a clear name."""
    latest = latest_ckpt_in_dir(split_output_dir)
    if latest is None:
        print(f"  No checkpoint found in {split_output_dir}", file=sys.stderr)
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.copy2(latest, dest)
    print(f"  Copied -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build manifest from audio dir and fine-tune A2SB split(s).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/training_data"),
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/training_output"),
        help="Directory for manifest and checkpoint outputs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Max training steps per split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--splits",
        choices=("both", "0.0-0.5", "0.5-1.0"),
        default="both",
        help="Which split(s) to fine-tune",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of files to use for validation (0–1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra args passed to Lightning (e.g. --trainer.precision bf16-mixed)",
    )
    args = parser.parse_args()

    # 1) Build manifest
    manifest_path = build_manifest(
        args.data_dir,
        args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # So the datamodule can find the manifest, we pass root_folder and filename.
    # Configs reference a placeholder; we override via CLI.
    root_folder = str(args.output_dir.resolve())
    manifest_filename = manifest_path.name

    common_override = [
        "--data.mix_dataset_config.CURATED.root_folder", root_folder,
        "--data.mix_dataset_config.CURATED.filename", manifest_filename,
        "--model.learning_rate", str(args.learning_rate),
    ]

    # 2) Fine-tune split(s)
    if args.splits in ("both", "0.0-0.5"):
        out1 = args.output_dir / "split_0.0_0.5"
        run_fit(
            CONFIG_SPLIT_1,
            Path(CKPT_SPLIT_1),
            out1,
            max_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            extra_args=common_override + ["--checkpoint_callback.dirpath", str(out1)] + args.extra,
        )

    if args.splits in ("both", "0.5-1.0"):
        out2 = args.output_dir / "split_0.5_1.0"
        run_fit(
            CONFIG_SPLIT_2,
            Path(CKPT_SPLIT_2),
            out2,
            max_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            extra_args=common_override + ["--checkpoint_callback.dirpath", str(out2)] + args.extra,
        )

    # 3) Copy latest checkpoints to a single folder for inference
    ckpt_dest = args.output_dir / "checkpoints"
    if args.splits in ("both", "0.0-0.5"):
        copy_final_checkpoints(
            args.output_dir / "split_0.0_0.5",
            ckpt_dest,
            "A2SB_twosplit_0.0_0.5_finetuned.ckpt",
        )
    if args.splits in ("both", "0.5-1.0"):
        copy_final_checkpoints(
            args.output_dir / "split_0.5_1.0",
            ckpt_dest,
            "A2SB_twosplit_0.5_1.0_finetuned.ckpt",
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
