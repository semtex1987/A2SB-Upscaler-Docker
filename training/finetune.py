#!/usr/bin/env python3
"""
A2SB fine-tuning orchestration: scan a directory for audio, build a training
manifest, and run fine-tuning for one or both time-split checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import yaml

# Configs travel with this script. Repo root is on sys.path so
# `python training/finetune.py` and `from training.finetune import ...` both
# resolve the shared eligibility module.
TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = TRAINING_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.eligibility import (
    LOADER_MIN_TRUE_SR,
    estimate_true_sr,
    is_loader_eligible,
    n_loader_segments,
)

# Segment length and sample rate must match the dataset config (130560 @ 44100 ≈ 2.96s)
SEGMENT_LENGTH = 130560
SAMPLING_RATE = 44100
MIN_DURATION_SEC = SEGMENT_LENGTH / SAMPLING_RATE

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# The heavy, rarely-changing parts (A2SB framework + release checkpoints) live in
# the container image; the code you iterate on does not have to. APP_ROOT points
# at the baked-in install and is overridable via A2SB_APP_ROOT, while the configs
# are resolved next to THIS script -- so you can `git clone`/`git pull` the repo
# onto a pod and run that copy directly, with no image rebuild.
APP_ROOT = Path(os.environ.get("A2SB_APP_ROOT", "/app"))
MAIN_PY = APP_ROOT / "main.py"
CKPT_DIR = Path(os.environ.get("A2SB_CKPT_DIR", str(APP_ROOT / "ckpts")))

# NVIDIA release checkpoints
CKPT_SPLIT_1 = str(CKPT_DIR / "A2SB_twosplit_0.0_0.5_release.ckpt")
CKPT_SPLIT_2 = str(CKPT_DIR / "A2SB_twosplit_0.5_1.0_release.ckpt")

CONFIG_SPLIT_1 = TRAINING_DIR / "configs" / "finetune_split1.yaml"
CONFIG_SPLIT_2 = TRAINING_DIR / "configs" / "finetune_split2.yaml"


def get_duration(path: str) -> float:
    """Duration in seconds. Raises if the file can't be read -- the caller
    reports the reason, so a missing decoder is distinguishable from bad audio."""
    return float(librosa.get_duration(path=path))


def find_audio_files(data_dir: Path) -> list[Path]:
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
) -> tuple[Path, int]:
    """Scan data_dir for audio, keep only loader-eligible files, then split.

    Eligibility is the same gate MixAudioDataset applies (estimated true
    sample rate >= 32 kHz). Files that would contribute zero samples are
    excluded before the train/validation split so the printed counts match
    the actual loaders.
    """
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(data_dir)
    if not files:
        raise SystemExit(f"No audio files found under {data_dir} (extensions: {AUDIO_EXTENSIONS})")

    eligible: list[tuple[str, float, int]] = []
    for f in files:
        try:
            d = get_duration(str(f))
        except Exception as e:  # noqa: BLE001 - report why, then keep scanning
            print(f"  skip (unreadable: {type(e).__name__}: {e}): {f.name}",
                  file=sys.stderr)
            continue
        if d < MIN_DURATION_SEC:
            print(f"  skip (too short {d:.1f}s < {MIN_DURATION_SEC:.1f}s): {f.name}", file=sys.stderr)
            continue
        true_sr = estimate_true_sr(str(f))
        if not is_loader_eligible(true_sr):
            print(
                f"  skip (trainer gate: estimated {true_sr} Hz < {LOADER_MIN_TRUE_SR} Hz): {f.name}",
                file=sys.stderr,
            )
            continue
        eligible.append((str(f), d, true_sr))

    if not eligible:
        raise SystemExit(
            "No training-eligible audio files (readable, long enough, and "
            f"estimated bandwidth >= {LOADER_MIN_TRUE_SR // 2} Hz)."
        )

    random.seed(seed)
    random.shuffle(eligible)
    n_val = max(1, int(len(eligible) * val_frac)) if len(eligible) > 1 else 0
    if n_val >= len(eligible):
        n_val = 1 if len(eligible) > 1 else 0
    n_train = len(eligible) - n_val
    train_rows = eligible[:n_train]
    val_rows = eligible[n_train:]

    if not train_rows:
        raise SystemExit(
            "No training samples remain after the train/validation split. "
            "Lower --val-frac or add more eligible audio files."
        )
    if not val_rows:
        raise SystemExit(
            "No validation samples remain after eligibility filtering. "
            "Add at least two eligible files so one can be held out."
        )

    manifest_path = output_dir / "finetune_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"')
        w.writerow(["split", "filepath", "duration", "estimated_true_sr"])
        for path, dur, true_sr in train_rows:
            w.writerow(["train", path, f"{dur:.4f}", str(true_sr)])
        for path, dur, true_sr in val_rows:
            w.writerow(["validation", path, f"{dur:.4f}", str(true_sr)])

    n_train_segments = sum(
        n_loader_segments(dur, SEGMENT_LENGTH, SAMPLING_RATE) for _, dur, _ in train_rows
    )
    print(
        f"Manifest: {len(train_rows)} train, {len(val_rows)} validation "
        f"({n_train_segments} train segments) -> {manifest_path}"
    )
    return manifest_path, n_train_segments


def write_run_override(
    config_path: Path,
    dest: Path,
    root_folder: str,
    manifest_filename: str,
    wandb_init: dict | None = None,
) -> Path:
    """Write a small config that points the datamodule at our manifest, and
    optionally swaps in a Weights & Biases logger.

    mix_dataset_config is an un-annotated dict parameter, and overriding a nested
    key through CLI dot-notation (--data.mix_dataset_config.CURATED.root_folder)
    makes jsonargparse hand the datamodule tuples instead of dicts:

        TypeError: tuple indices must be integers or slices, not str

    Supplying the whole mapping via an extra -c config uses the same code path
    that already parses the base config correctly, so the structure survives.
    Each dataset entry is copied from the base config with only the manifest
    location patched, preserving flags like apply_sr_loss_mask.
    """
    import yaml

    base = yaml.safe_load(open(config_path, encoding="utf-8"))
    mdc = (base.get("data") or {}).get("mix_dataset_config") or {}
    patched = {}
    for name, entry in mdc.items():
        entry = dict(entry or {})
        entry["root_folder"] = root_folder
        entry["filename"] = manifest_filename
        patched[name] = entry
    if not patched:
        raise SystemExit(f"No data.mix_dataset_config found in {config_path}")

    override: dict = {"data": {"mix_dataset_config": patched}}
    if wandb_init is not None:
        # The base config leaves trainer.logger null, so Lightning falls back to
        # CSVLogger. Point it at W&B instead; going through this config keeps the
        # nested class_path/init_args structure intact.
        override["trainer"] = {
            "logger": {
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": wandb_init,
            }
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(override, f, sort_keys=False)
    return dest


def run_fit(
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float | None,
    extra_args: list[str],
    data_override: Path | None = None,
) -> None:
    """Run main.py fit with the given config and checkpoint."""
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "fit",
        "-c", str(config_path),
    ]
    # Merged after the base config, so it wins.
    if data_override is not None:
        cmd += ["-c", str(data_override)]
    cmd += [
        "--ckpt_path", str(ckpt_path),
        "--trainer.max_steps", str(max_steps),
        # main.py does link_arguments("checkpoint_callback.dirpath",
        # "trainer.default_root_dir"), so default_root_dir is DERIVED and must
        # never be passed directly -- jsonargparse rejects setting a link target.
        # Setting dirpath drives both.
        "--checkpoint_callback.dirpath", str(output_dir),
        "--data.batch_size", str(batch_size),
    ]
    if learning_rate is not None:
        cmd += ["--model.learning_rate", str(learning_rate)]
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(APP_ROOT), check=True)


INIT_CHECKPOINT_NAME = "finetune_start.ckpt"


def is_trained_checkpoint(path: Path) -> bool:
    """True if *path* is a real training artifact, not an initialization stub.

    ``prepare_finetune_checkpoint`` writes empty optimizer_states so Lightning
    will load it. That file must never be exported as a fine-tuned weight or
    selected for resume as if training had run.
    """
    if path.name == INIT_CHECKPOINT_NAME:
        return False
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    opts = ckpt.get("optimizer_states")
    sched = ckpt.get("lr_schedulers")
    if not opts or not sched:
        return False
    return int(ckpt.get("global_step", 0)) > 0


def latest_ckpt_in_dir(d: Path) -> Path | None:
    """Latest *trained* checkpoint in d (by mtime), or None.

    Initialization stubs and unreadable files are skipped so export cannot
    label ``finetune_start.ckpt`` as a fine-tuned weight.
    """
    if not d.is_dir():
        return None
    last = d / "last.ckpt"
    if last.is_file() and is_trained_checkpoint(last):
        return last
    ckpts = [p for p in d.glob("*.ckpt") if is_trained_checkpoint(p)]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def checkpoint_global_step(ckpt_path: Path) -> int:
    """global_step recorded in a checkpoint. fit(ckpt_path=...) restores it, so
    trainer.max_steps must be offset by it or training stops before a single
    step runs."""
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return int(ckpt.get("global_step", 0))
    except Exception as e:
        print(f"  WARNING: could not read global_step from {ckpt_path}: {e}",
              file=sys.stderr)
        return 0


def find_resume_checkpoint(split_dir: Path) -> Path | None:
    """Newest usable training checkpoint in *split_dir*, or None.

    Skips our own start checkpoint, and skips anything that fails to load: a
    checkpoint written while the disk was filling up is truncated, and Lightning
    would rather resume from an older good one than die on a corrupt newest one.
    """
    if not split_dir.is_dir():
        return None

    candidates = sorted(
        (p for p in split_dir.glob("*.ckpt") if is_trained_checkpoint(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def prepare_finetune_checkpoint(release_ckpt: Path, dest: Path) -> Path:
    """Make a checkpoint that `fit --ckpt_path` will accept for FINE-TUNING.

    The NVIDIA release checkpoints carry only the model:

        KeyError: Trying to restore optimizer state but checkpoint contains only
        the model. This is probably due to `ModelCheckpoint.save_weights_only`
        being set to `True`.

    Lightning's --ckpt_path means *resume*, so it demands optimizer_states and
    lr_schedulers. Fine-tuning wants the opposite of a resume: pretrained
    weights, but a fresh optimizer at the new learning rate. Writing empty
    optimizer_states/lr_schedulers satisfies the restore path while leaving the
    freshly-built optimizer and scheduler untouched (both restore loops zip over
    the empty lists and do nothing). Loop counters are cleared too, so --steps
    counts from zero rather than from the release checkpoint's global_step.

    A checkpoint that already carries training state is returned untouched, so
    resuming an interrupted fine-tune still works.
    """
    import torch

    ckpt = torch.load(str(release_ckpt), map_location="cpu", weights_only=False)
    if "optimizer_states" in ckpt and "lr_schedulers" in ckpt:
        return release_ckpt

    ckpt["optimizer_states"] = []
    ckpt["lr_schedulers"] = []
    ckpt.pop("loops", None)
    ckpt["global_step"] = 0
    ckpt["epoch"] = 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(dest))
    print(f"  prepared fine-tune start checkpoint -> {dest}")
    return dest


def planned_max_steps(
    resumed_step: int,
    additional_steps: int,
    until_step: int | None,
) -> int:
    """Lightning ``trainer.max_steps`` is an absolute global_step target.

    ``--steps`` is additional work from the starting checkpoint. ``--until-step``
    is the absolute target and wins when both are supplied.
    """
    if until_step is not None:
        return int(until_step)
    return int(resumed_step) + int(additional_steps)


def write_run_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def per_rank_epoch_batches(n_train_segments: int, batch_size: int, devices: int | None) -> int:
    """Train batches one rank sees per epoch after DistributedSampler sharding."""
    world = max(1, int(devices or 1))
    return max(1, n_train_segments // (batch_size * world))


def merge_run_record(path: Path, updates: dict) -> None:
    current: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            current = loaded
    current.update(updates)
    write_run_record(path, current)


def copy_final_checkpoints(
    split_output_dir: Path,
    dest_dir: Path,
    name: str,
) -> bool:
    """Copy a trained checkpoint to dest_dir under the inference filename.

    Initialization stubs (``finetune_start.ckpt``) and files with no completed
    optimizer step are not exportable. Prefer Lightning's ``last.ckpt`` when
    it is a real terminal training state.
    """
    latest = latest_ckpt_in_dir(split_output_dir)
    if latest is None:
        print(f"  No trained checkpoint found in {split_output_dir}", file=sys.stderr)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.copy2(latest, dest)
    print(f"  Copied -> {dest}")
    return True


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
        help="Additional optimizer steps to take from the starting checkpoint "
             "(global_step + N). For an absolute target, use --until-step.",
    )
    parser.add_argument(
        "--until-step",
        type=int,
        default=None,
        help="Absolute Lightning trainer.max_steps (global_step target). "
             "When set, this overrides --steps.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for inference-named checkpoint copies. "
             "Default: <output-dir>/checkpoints.",
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
        "--devices",
        type=int,
        default=None,
        metavar="N",
        help="GPUs to train on (default: whatever the config says, 1). With more "
             "than one, DDP is used and the effective batch becomes "
             "devices x --batch-size x accumulate_grad_batches, so consider "
             "raising --learning-rate to match. Saturate a single GPU with "
             "--batch-size before adding a second one.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases instead of the default CSV logger. "
             "Needs the wandb package and a WANDB_API_KEY (or a prior "
             "'wandb login'). Each split is logged as its own run.",
    )
    parser.add_argument(
        "--wandb-project",
        default="a2sb-finetune",
        metavar="NAME",
        help="W&B project name (default: a2sb-finetune).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        metavar="NAME",
        help="W&B run name prefix. The split is appended, so one invocation of "
             "--splits both yields two clearly-named runs. Default: the output "
             "directory name.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=None,
        metavar="N",
        help="Validate every N training batches (default: min(1000, "
             "batches_per_epoch)). Validation is far more expensive than a "
             "training step -- see --val-samples -- so raise this if validation "
             "dominates wall-clock.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap validation samples (data.val_max_samples). Each one runs a "
             "25-step diffusion sample, vocoding and audio metrics, so the "
             "config's 256 can take ~an hour per cycle; 16-32 gives the same "
             "signal in minutes. Checkpoint selection monitors global_step, not "
             "a validation metric, so this does not change which are kept.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore checkpoints already in this output dir and start from "
             "the release checkpoint. New GUI jobs use a fresh directory, so "
             "resume only happens when you point --output-dir at an existing run.",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra args passed through to Lightning. These are positional, so "
             "separate them with a bare '--' or argparse treats them as unknown "
             "options: finetune.py --steps 5000 -- --trainer.precision bf16-mixed",
    )
    args = parser.parse_args()

    # 0) Preflight: fail fast with an actionable message rather than letting a
    # missing framework/checkpoint surface later as an opaque Lightning error.
    problems: list[str] = []
    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as e:  # noqa: BLE001 - also catches libsndfile load failures
        problems.append(
            f"audio stack unusable ({type(e).__name__}: {e})\n"
            f"    Every file would be reported as unreadable. Install it with:\n"
            f"      apt-get update && apt-get install -y ffmpeg libsndfile1\n"
            f"      pip install librosa soundfile"
        )
    if args.wandb:
        try:
            import wandb  # noqa: F401
        except Exception as e:  # noqa: BLE001
            problems.append(
                f"--wandb given but wandb is unusable ({type(e).__name__}: {e})\n"
                f"    pip install wandb"
            )
        else:
            if not (os.environ.get("WANDB_API_KEY") or
                    (Path.home() / ".netrc").exists()):
                print("  NOTE: no WANDB_API_KEY and no ~/.netrc; wandb may prompt "
                      "or fall back to offline mode. Run 'wandb login' first.",
                      file=sys.stderr)
    if not MAIN_PY.is_file():
        problems.append(
            f"A2SB main.py not found at {MAIN_PY}\n"
            f"    Set A2SB_APP_ROOT to the directory containing main.py "
            f"(the contents of nvidia-a2sb-original-repo/)."
        )
    needed = []
    if args.splits in ("both", "0.0-0.5"):
        needed.append(CKPT_SPLIT_1)
    if args.splits in ("both", "0.5-1.0"):
        needed.append(CKPT_SPLIT_2)
    missing = [c for c in needed if not Path(c).is_file()]
    if missing:
        problems.append(
            "release checkpoint(s) not found:\n"
            + "".join(f"      {c}\n" for c in missing)
            + f"    Set A2SB_CKPT_DIR (currently {CKPT_DIR}) or download them, e.g.:\n"
            f"      mkdir -p {CKPT_DIR} && wget -P {CKPT_DIR} \\\n"
            f"        https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/"
            f"resolve/main/ckpt/A2SB_twosplit_0.0_0.5_release.ckpt \\\n"
            f"        https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/"
            f"resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt"
        )
    if problems:
        raise SystemExit(
            "ERROR: cannot start fine-tuning:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    # 1) Build manifest
    manifest_path, n_train_segments = build_manifest(
        args.data_dir,
        args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    write_run_record(
        args.output_dir / "run.json",
        {
            "data_dir": str(args.data_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "manifest": manifest_path.name,
            "splits": args.splits,
            "additional_steps": args.steps,
            "until_step": args.until_step,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "restart": bool(args.restart),
            "val_frac": args.val_frac,
            "val_every": args.val_every,
            "val_samples": args.val_samples,
        },
    )

    # So the datamodule can find the manifest, we pass root_folder and filename.
    # Configs reference a placeholder; we override via CLI.
    root_folder = str(args.output_dir.resolve())
    manifest_filename = manifest_path.name

    # Clamp val_check_interval so Lightning never raises on small datasets.
    # With DDP, each rank sees about 1/world_size of the segments.
    batches_per_epoch = per_rank_epoch_batches(
        n_train_segments, args.batch_size, args.devices
    )
    # val_check_interval counts BATCHES, and Lightning rejects a value larger
    # than one epoch's worth. batches_per_epoch shrinks as --batch-size grows,
    # so a --val-every that was fine at batch 2 becomes invalid at batch 16.
    val_interval = min(args.val_every or 1000, batches_per_epoch)
    if args.val_every and val_interval != args.val_every:
        print(f"  NOTE: --val-every {args.val_every} exceeds one epoch "
              f"({batches_per_epoch} batches at --batch-size {args.batch_size}); "
              f"clamped to {val_interval}.", file=sys.stderr)

    # learning_rate is passed by run_fit itself; don't duplicate it here.
    common_override = [
        "--trainer.val_check_interval", str(val_interval),
    ]
    if args.devices is not None:
        common_override += ["--trainer.devices", str(args.devices)]
        # 'auto' picks a single-device strategy; multi-GPU needs DDP named.
        if args.devices > 1:
            common_override += [
                "--trainer.strategy", "ddp",
                "--trainer.use_distributed_sampler", "true",
            ]
    if args.val_samples is not None:
        common_override += ["--data.val_max_samples", str(args.val_samples)]

    def wandb_init_for(split_tag: str) -> dict | None:
        if not args.wandb:
            return None
        prefix = args.wandb_run_name or args.output_dir.resolve().name
        return {
            "project": args.wandb_project,
            "name": f"{prefix}-{split_tag}",
            "save_dir": str(args.output_dir.resolve()),
        }

    # 2) Fine-tune split(s)
    if args.splits in ("both", "0.0-0.5"):
        out1 = args.output_dir / "split_0.0_0.5"
        start1 = None if args.restart else find_resume_checkpoint(out1)
        if start1 is not None:
            print(f"Split 0.0-0.5 resuming from {start1}")
        else:
            start1 = prepare_finetune_checkpoint(Path(CKPT_SPLIT_1), out1 / "finetune_start.ckpt")
        resumed = checkpoint_global_step(start1)
        effective_max = planned_max_steps(resumed, args.steps, args.until_step)
        print(
            f"Split 0.0-0.5 starts at global_step={resumed}; "
            f"training until {effective_max} "
            f"({'absolute --until-step' if args.until_step is not None else f'+{args.steps} additional'})"
        )
        if resumed >= effective_max:
            print(f"  NOTE: already at/past the target ({resumed} >= {effective_max}); "
                  f"raise --steps or --until-step to train further.", file=sys.stderr)
        merge_run_record(
            args.output_dir / "run.json",
            {
                "split_0.0-0.5": {
                    "start_checkpoint": str(start1),
                    "resumed_step": resumed,
                    "max_steps": effective_max,
                },
            },
        )
        run_fit(
            CONFIG_SPLIT_1,
            start1,
            out1,
            max_steps=effective_max,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            extra_args=common_override + args.extra,
            data_override=write_run_override(
                CONFIG_SPLIT_1, out1 / "data_override.yaml",
                root_folder, manifest_filename,
                wandb_init=wandb_init_for("split_0.0_0.5"),
            ),
        )

    if args.splits in ("both", "0.5-1.0"):
        out2 = args.output_dir / "split_0.5_1.0"
        start2 = None if args.restart else find_resume_checkpoint(out2)
        if start2 is not None:
            print(f"Split 0.5-1.0 resuming from {start2}")
        else:
            start2 = prepare_finetune_checkpoint(Path(CKPT_SPLIT_2), out2 / "finetune_start.ckpt")
        resumed = checkpoint_global_step(start2)
        effective_max = planned_max_steps(resumed, args.steps, args.until_step)
        print(
            f"Split 0.5-1.0 starts at global_step={resumed}; "
            f"training until {effective_max} "
            f"({'absolute --until-step' if args.until_step is not None else f'+{args.steps} additional'})"
        )
        if resumed >= effective_max:
            print(f"  NOTE: already at/past the target ({resumed} >= {effective_max}); "
                  f"raise --steps or --until-step to train further.", file=sys.stderr)
        merge_run_record(
            args.output_dir / "run.json",
            {
                "split_0.5-1.0": {
                    "start_checkpoint": str(start2),
                    "resumed_step": resumed,
                    "max_steps": effective_max,
                },
            },
        )
        run_fit(
            CONFIG_SPLIT_2,
            start2,
            out2,
            max_steps=effective_max,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            extra_args=common_override + args.extra,
            data_override=write_run_override(
                CONFIG_SPLIT_2, out2 / "data_override.yaml",
                root_folder, manifest_filename,
                log_dir=out2,
                tensorboard=args.tensorboard,
                wandb_init=wandb_init_for("split_0.5_1.0"),
            ),
        )

    # 3) Copy latest checkpoints to a single folder for inference
    ckpt_dest = Path(args.export_dir) if args.export_dir is not None else args.output_dir / "checkpoints"
    any_missing = False
    if args.splits in ("both", "0.0-0.5"):
        ok = copy_final_checkpoints(
            args.output_dir / "split_0.0_0.5",
            ckpt_dest,
            "A2SB_twosplit_0.0_0.5_finetuned.ckpt",
        )
        if not ok:
            any_missing = True
    if args.splits in ("both", "0.5-1.0"):
        ok = copy_final_checkpoints(
            args.output_dir / "split_0.5_1.0",
            ckpt_dest,
            "A2SB_twosplit_0.5_1.0_finetuned.ckpt",
        )
        if not ok:
            any_missing = True

    if any_missing:
        print(
            "ERROR: one or more splits produced no checkpoint. "
            "Check the training logs above for details.",
            file=sys.stderr,
        )
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
