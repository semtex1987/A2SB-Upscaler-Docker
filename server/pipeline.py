"""Per-file restoration: filter, run both channels through A2SB, recombine, measure."""
from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment

from server.audio import (
    apply_lowpass_to_segment,
    ensure_a2sb_input_format,
    high_band_rms_db,
)
from server.config import WORK_DIR
from server.inference import InferenceCancelled, InferenceError, run_a2sb_inference
from server.serialization import camelize

#: Share of a file's progress spent before and after the diffusion passes.
#: Deliberately small: loading, filtering and recombining take seconds while
#: inference takes minutes, and a progress bar that claims otherwise reads as
#: a hung process.
PREPARE_SHARE = 0.02
FINALIZE_SHARE = 0.06
INFERENCE_SHARE = 1.0 - PREPARE_SHARE - FINALIZE_SHARE
#: Within one channel, how much of its span the lowpass accounts for.
FILTER_SHARE_OF_CHANNEL = 0.02


@dataclass
class FileProgress:
    stage: str
    #: None while the inference subprocess has not reported a step yet. The UI
    #: shows elapsed time and an indeterminate bar rather than inventing a number.
    fraction: Optional[float] = None
    eta_sec: Optional[float] = None


@dataclass
class FileResult:
    name: str
    source_path: str
    restored_path: str
    filtered_path: str
    channels: int
    duration_sec: float
    cutoff_hz: int
    steps: int
    batch_size: int
    high_band_in_db: float
    high_band_out_db: float
    high_band_delta_db: float
    elapsed_sec: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return camelize(asdict(self))


ProgressCallback = Callable[[FileProgress], None]
LogCallback = Callable[[str], None]


class PipelineError(RuntimeError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def restore_file(
    source_path: str,
    run_dir: Path,
    steps: int,
    cutoff_hz: int,
    batch_size: int,
    on_progress: ProgressCallback,
    on_log: LogCallback,
    cancel_event: threading.Event,
) -> FileResult:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)

    on_progress(FileProgress(stage="Loading source", fraction=0.0))
    try:
        audio = ensure_a2sb_input_format(AudioSegment.from_file(source_path))
    except Exception as exc:
        raise PipelineError(f"Could not decode {os.path.basename(source_path)}.", str(exc)) from exc

    if audio.channels not in (1, 2):
        raise PipelineError(
            f"{os.path.basename(source_path)} has {audio.channels} channels; only mono and stereo are supported."
        )

    stem = Path(source_path).stem.replace(" ", "_") or "audio"
    restored_path = run_dir / f"{stem}_restored.wav"
    filtered_path = run_dir / f"{stem}_filtered_input.wav"

    channel_names = ["Mono"] if audio.channels == 1 else ["Left", "Right"]
    channel_segments = [audio] if audio.channels == 1 else audio.split_to_mono()
    channel_span = INFERENCE_SHARE / len(channel_segments)

    restored_channels: list[AudioSegment] = []
    filtered_channels: list[AudioSegment] = []

    for index, (label, segment) in enumerate(zip(channel_names, channel_segments)):
        _raise_if_cancelled(cancel_event)
        base = PREPARE_SHARE + (index * channel_span)

        on_progress(
            FileProgress(stage=f"{label}: applying {cutoff_hz} Hz lowpass", fraction=base)
        )
        filtered = apply_lowpass_to_segment(segment, cutoff_hz)
        filtered_channels.append(filtered)

        channel_in = Path(WORK_DIR) / f"{stem}_{label.lower()}_input.wav"
        channel_out = run_dir / f".{stem}_{label.lower()}_restored.wav"
        if channel_out.exists():
            channel_out.unlink()
        filtered.export(channel_in, format="wav")

        inference_base = base + (channel_span * FILTER_SHARE_OF_CHANNEL)
        inference_span = channel_span * (1.0 - FILTER_SHARE_OF_CHANNEL)

        def report(fraction: Optional[float], eta: Optional[float], _label=label, _base=inference_base, _span=inference_span) -> None:
            overall = None if fraction is None else _base + (_span * fraction)
            on_progress(FileProgress(stage=f"{_label}: diffusion", fraction=overall, eta_sec=eta))

        report(None, None)
        try:
            run_a2sb_inference(
                input_path=str(channel_in),
                output_path=str(channel_out),
                steps=steps,
                cutoff_hz=cutoff_hz,
                batch_size=batch_size,
                on_log=on_log,
                on_progress=report,
                cancel_event=cancel_event,
            )
        except InferenceError as exc:
            raise PipelineError(str(exc), exc.tail) from exc
        finally:
            channel_in.unlink(missing_ok=True)

        restored_channels.append(AudioSegment.from_file(channel_out))
        channel_out.unlink(missing_ok=True)

    _raise_if_cancelled(cancel_event)
    finalize_base = PREPARE_SHARE + INFERENCE_SHARE
    on_progress(FileProgress(stage="Recombining", fraction=finalize_base))

    if len(restored_channels) == 1:
        restored_channels[0].export(restored_path, format="wav")
        filtered_channels[0].export(filtered_path, format="wav")
    else:
        AudioSegment.from_mono_audiosegments(*restored_channels).export(restored_path, format="wav")
        AudioSegment.from_mono_audiosegments(*filtered_channels).export(filtered_path, format="wav")

    on_progress(FileProgress(stage="Measuring high-band energy", fraction=finalize_base + FINALIZE_SHARE * 0.4))
    hf_in = high_band_rms_db(str(filtered_path), cutoff_hz)
    hf_out = high_band_rms_db(str(restored_path), cutoff_hz)
    delta = hf_out - hf_in

    warnings: list[str] = []
    if delta < 1.0:
        warnings.append(
            f"The model added {delta:+.1f} dB above {cutoff_hz} Hz. The release "
            f"checkpoints are weak above ~12 kHz; try a lower cutoff or fine-tuned weights."
        )

    on_progress(FileProgress(stage="Done", fraction=1.0))

    return FileResult(
        name=os.path.basename(source_path),
        source_path=source_path,
        restored_path=str(restored_path),
        filtered_path=str(filtered_path),
        channels=audio.channels,
        duration_sec=round(len(audio) / 1000.0, 3),
        cutoff_hz=cutoff_hz,
        steps=steps,
        batch_size=batch_size,
        high_band_in_db=round(hf_in, 2),
        high_band_out_db=round(hf_out, 2),
        high_band_delta_db=round(delta, 2),
        elapsed_sec=round(time.monotonic() - started, 1),
        warnings=warnings,
    )


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise InferenceCancelled()


def cleanup_run_dir(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)
