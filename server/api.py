"""HTTP surface: staging, submission, live progress, result media, and training."""
from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.analysis import analyze_source, peak_envelope, spectrogram_payload
from server.config import (
    AUDIO_EXTENSIONS,
    BATCH_DEFAULT,
    BATCH_MAX,
    BATCH_MIN,
    CUTOFF_DEFAULT_HZ,
    CUTOFF_MAX_HZ,
    CUTOFF_MIN_HZ,
    INPUT_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    STEPS_DEFAULT,
    STEPS_MAX,
    STEPS_MIN,
    TRAIN_BATCH_DEFAULT,
    TRAIN_BATCH_MAX,
    TRAIN_BATCH_MIN,
    TRAIN_LR_DEFAULT,
    TRAIN_STEPS_DEFAULT,
    TRAIN_STEPS_MAX,
    TRAIN_STEPS_MIN,
    TRAINING_DATA_DIR,
    TRAINING_OUTPUT_DIR,
    WORK_DIR,
)
from server.jobs import JOB_KIND_TRAIN, TrainParams, store
from server.serialization import camelize
from server.training import (
    SPLIT_BOTH,
    SPLIT_FIRST,
    SPLIT_SECOND,
    activate_checkpoints,
    checkpoint_status,
    preflight,
    read_training_metrics,
    revert_to_release,
    vet_dataset,
)

router = APIRouter(prefix="/api")

UPLOAD_DIR = INPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_ROOTS = [
    p.resolve()
    for p in (INPUT_DIR, OUTPUT_DIR, RUNS_DIR, WORK_DIR, TRAINING_OUTPUT_DIR)
    if p.exists() or True  # ensure resolver accepts paths even before they exist
]
# Include training data root unconditionally so scan paths are accepted.
_TRAINING_DATA_ROOTS = [TRAINING_DATA_DIR.resolve()]


def _describe(exc: Exception) -> str:
    """Decoder errors are often raised bare, so fall back to the type name."""
    message = str(exc).strip()
    return message or f"Could not read this file ({type(exc).__name__})."


def _resolve_media_path(raw: str) -> Path:
    """Confine every path-taking endpoint to the staging and output trees."""
    if not raw:
        raise HTTPException(status_code=400, detail="A path is required.")
    candidate = Path(raw).resolve()
    for root in _ALLOWED_ROOTS:
        if candidate == root or root in candidate.parents:
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"No such file: {raw}")
            return candidate
    raise HTTPException(status_code=403, detail="Path is outside the permitted directories.")


def _resolve_training_path(raw: str) -> Path:
    """Confine training scan endpoints to the training_data root."""
    if not raw:
        raise HTTPException(status_code=400, detail="A path is required.")
    candidate = Path(raw).resolve()
    # Also accept paths inside the general allowed roots (e.g. pod-staged audio
    # that the user wants to vet without copying to training_data/).
    all_roots = _ALLOWED_ROOTS + _TRAINING_DATA_ROOTS
    for root in all_roots:
        if candidate == root or root in candidate.parents:
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"No such path: {raw}")
            return candidate
    raise HTTPException(status_code=403, detail="Path is outside the permitted directories.")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@router.get("/config")
def get_config() -> dict:
    return {
        "steps": {"min": STEPS_MIN, "max": STEPS_MAX, "default": STEPS_DEFAULT},
        "batchSize": {"min": BATCH_MIN, "max": BATCH_MAX, "default": BATCH_DEFAULT},
        "cutoffHz": {"min": CUTOFF_MIN_HZ, "max": CUTOFF_MAX_HZ, "default": CUTOFF_DEFAULT_HZ},
        "inputDir": str(INPUT_DIR),
        "outputDir": str(OUTPUT_DIR),
        "audioExtensions": sorted(AUDIO_EXTENSIONS),
        "training": {
            "steps": {"min": TRAIN_STEPS_MIN, "max": TRAIN_STEPS_MAX, "default": TRAIN_STEPS_DEFAULT},
            "batchSize": {"min": TRAIN_BATCH_MIN, "max": TRAIN_BATCH_MAX, "default": TRAIN_BATCH_DEFAULT},
            "learningRateDefault": TRAIN_LR_DEFAULT,
            "dataDir": str(TRAINING_DATA_DIR),
            "outputDir": str(TRAINING_OUTPUT_DIR),
            "splits": [SPLIT_BOTH, SPLIT_FIRST, SPLIT_SECOND],
        },
    }


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


@router.post("/uploads")
def upload_files(files: list[UploadFile]) -> dict:
    """Persist uploads and measure each one so the UI can suggest a cutoff."""
    batch_dir = UPLOAD_DIR / uuid.uuid4().hex[:8]
    batch_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    errors = []
    for upload in files:
        name = os.path.basename(upload.filename or "audio")
        if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
            upload.file.close()
            errors.append(
                {
                    "name": name,
                    "error": f"Not an audio file. Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}.",
                }
            )
            continue

        destination = batch_dir / name
        try:
            with destination.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle, length=1024 * 1024)
        finally:
            upload.file.close()

        try:
            analyses.append(analyze_source(str(destination)).to_dict())
        except Exception as exc:  # noqa: BLE001 - one bad file must not fail the batch
            destination.unlink(missing_ok=True)
            errors.append({"name": name, "error": _describe(exc)})

    return {"files": analyses, "errors": errors}


class AnalyzeRequest(BaseModel):
    paths: list[str]


@router.post("/analyze")
def analyze_paths(request: AnalyzeRequest) -> dict:
    analyses = []
    errors = []
    for raw in request.paths:
        try:
            resolved = _resolve_media_path(raw)
            analyses.append(analyze_source(str(resolved)).to_dict())
        except HTTPException as exc:
            errors.append({"name": os.path.basename(raw), "error": exc.detail})
        except Exception as exc:  # noqa: BLE001
            errors.append({"name": os.path.basename(raw), "error": _describe(exc)})
    return {"files": analyses, "errors": errors}


@router.get("/browse")
def browse(pattern: str = Query(..., description="Glob or literal path under the staging directories")) -> dict:
    """Resolve pod-staged paths without analysing them, so the list appears instantly."""
    matches: list[str] = []
    for line in (p.strip() for p in pattern.splitlines()):
        if not line:
            continue
        found = sorted(glob.glob(line)) or ([line] if os.path.exists(line) else [])
        matches.extend(found)

    entries = []
    seen: set[str] = set()
    for match in matches:
        try:
            resolved = _resolve_media_path(match)
        except HTTPException:
            continue
        for path in _expand_audio_files(resolved):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"path": key, "name": path.name, "sizeBytes": path.stat().st_size})
    return {"entries": entries}


def _expand_audio_files(path: Path) -> list[Path]:
    """A file yields itself; a directory yields the audio it contains, recursively."""
    if path.is_file():
        return [path] if path.suffix.lower() in AUDIO_EXTENSIONS else []
    if not path.is_dir():
        return []
    found = [
        child
        for child in path.rglob("*")
        # Scratch and metadata directories (.work, run temp files) are not user content.
        if not any(part.startswith(".") for part in child.relative_to(path).parts)
        and child.is_file()
        and child.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(found)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class JobFileRequest(BaseModel):
    name: str
    source_path: str = Field(alias="sourcePath")
    cutoff_hz: int = Field(alias="cutoffHz", ge=CUTOFF_MIN_HZ, le=CUTOFF_MAX_HZ)

    model_config = {"populate_by_name": True}


class JobRequest(BaseModel):
    files: list[JobFileRequest]
    steps: int = Field(default=STEPS_DEFAULT, ge=STEPS_MIN, le=STEPS_MAX)
    batch_size: int = Field(default=BATCH_DEFAULT, alias="batchSize", ge=BATCH_MIN, le=BATCH_MAX)

    model_config = {"populate_by_name": True}


@router.post("/jobs", status_code=202)
def create_job(request: JobRequest) -> dict:
    if not request.files:
        raise HTTPException(status_code=400, detail="Select at least one file.")

    entries = []
    for item in request.files:
        resolved = _resolve_media_path(item.source_path)
        entries.append(
            {"name": item.name, "source_path": str(resolved), "cutoff_hz": item.cutoff_hz}
        )

    job = store.submit(entries, steps=request.steps, batch_size=request.batch_size)
    return job.to_dict()


@router.get("/jobs")
def list_jobs() -> dict:
    return {
        "jobs": [job.to_dict() for job in store.list_jobs()],
        "activeJobId": store.active_job_id(),
        "queueDepth": store.queue_depth(),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return {"job": job.to_dict(), "log": store.get_log(job_id)}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not store.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job is not cancellable.")
    return {"ok": True}


@router.get("/events")
async def events(request: Request) -> StreamingResponse:
    """Server-sent job and log events.

    Opens with a snapshot so a reconnecting tab is immediately consistent
    without a separate fetch.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    store.broker.subscribe(loop, queue)

    async def stream():
        try:
            snapshot = {
                "type": "snapshot",
                "jobs": [job.to_dict() for job in store.list_jobs()],
                "activeJobId": store.active_job_id(),
            }
            yield f"data: {json.dumps(snapshot)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keeps intermediary proxies (RunPod, nginx) from closing idle streams.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            store.broker.unsubscribe(loop, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------


@router.get("/audio")
def get_audio(path: str) -> FileResponse:
    resolved = _resolve_media_path(path)
    # Range support matters here: the transport seeks around a multi-minute WAV.
    return FileResponse(resolved, media_type="audio/wav", filename=resolved.name)


@router.get("/download")
def download(path: str) -> FileResponse:
    resolved = _resolve_media_path(path)
    return FileResponse(
        resolved,
        media_type="application/octet-stream",
        filename=resolved.name,
        content_disposition_type="attachment",
    )


@router.get("/spectrogram")
def get_spectrogram(path: str, maxSeconds: Optional[float] = None) -> dict:
    resolved = _resolve_media_path(path)
    return spectrogram_payload(str(resolved), max_seconds=maxSeconds)


@router.get("/waveform")
def get_waveform(path: str, buckets: int = 1600) -> dict:
    resolved = _resolve_media_path(path)
    return peak_envelope(str(resolved), buckets=max(200, min(buckets, 4000)))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class VetRequest(BaseModel):
    paths: list[str]


@router.post("/training/vet")
def training_vet(request: VetRequest) -> dict:
    """Vet a list of audio file paths for training suitability."""
    resolved_paths = []
    errors = []
    for raw in request.paths:
        try:
            resolved_paths.append(str(_resolve_training_path(raw)))
        except HTTPException as exc:
            errors.append({"path": raw, "error": exc.detail})
    results = vet_dataset(resolved_paths)
    return {
        "files": [camelize(r.__dict__) for r in results],
        "errors": errors,
    }


@router.get("/training/browse")
def training_browse(
    pattern: str = Query(..., description="Directory or glob path for audio files"),
) -> dict:
    """List audio files under the training_data directory."""
    entries = []
    seen: set[str] = set()
    for line in (p.strip() for p in pattern.splitlines()):
        if not line:
            continue
        found = sorted(glob.glob(line)) or ([line] if os.path.exists(line) else [])
        for match in found:
            try:
                resolved = _resolve_training_path(match)
            except HTTPException:
                continue
            for path in _expand_audio_files(resolved):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                entries.append({"path": key, "name": path.name, "sizeBytes": path.stat().st_size})
    return {"entries": entries}


class TrainJobRequest(BaseModel):
    data_dir: str = Field(alias="dataDir")
    steps: int = Field(default=TRAIN_STEPS_DEFAULT, ge=TRAIN_STEPS_MIN, le=TRAIN_STEPS_MAX)
    batch_size: int = Field(default=TRAIN_BATCH_DEFAULT, alias="batchSize", ge=TRAIN_BATCH_MIN, le=TRAIN_BATCH_MAX)
    learning_rate: float = Field(default=TRAIN_LR_DEFAULT, alias="learningRate", gt=0)
    splits: str = Field(default=SPLIT_BOTH)
    val_frac: float = Field(default=0.1, alias="valFrac", ge=0.01, le=0.5)
    val_every: Optional[int] = Field(default=None, alias="valEvery", ge=1)
    val_samples: Optional[int] = Field(default=None, alias="valSamples", ge=1)
    restart: bool = Field(default=False)

    model_config = {"populate_by_name": True}


@router.post("/training/jobs", status_code=202)
def create_training_job(request: TrainJobRequest) -> dict:
    if request.splits not in (SPLIT_BOTH, SPLIT_FIRST, SPLIT_SECOND):
        raise HTTPException(
            status_code=400,
            detail=f"splits must be one of: {SPLIT_BOTH}, {SPLIT_FIRST}, {SPLIT_SECOND}.",
        )

    # Validate the data directory is accessible.
    try:
        data_path = _resolve_training_path(request.data_dir)
    except HTTPException:
        raise HTTPException(status_code=400, detail=f"dataDir is not accessible: {request.data_dir}")
    if not data_path.is_dir():
        raise HTTPException(status_code=400, detail=f"dataDir must be a directory: {request.data_dir}")

    problems = preflight(splits=request.splits, training_data_dir=str(data_path))
    if problems:
        raise HTTPException(status_code=422, detail={"problems": problems})

    params = TrainParams(
        data_dir=str(data_path),
        output_dir=str(TRAINING_OUTPUT_DIR),
        steps=request.steps,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        splits=request.splits,
        val_frac=request.val_frac,
        val_every=request.val_every,
        val_samples=request.val_samples,
        restart=request.restart,
    )
    job = store.submit_training(params)
    return job.to_dict()


def _checkpoint_status_to_dict(status) -> dict:
    """Serialize CheckpointStatus without camelizing the checkpoint filename keys."""
    return {
        "active": status.active,
        "finetunedPaths": status.finetuned_paths,
        "releasePaths": status.release_paths,
        "ensembleConfig": status.ensemble_config,
    }


@router.get("/training/checkpoints")
def get_checkpoints() -> dict:
    status = checkpoint_status()
    return _checkpoint_status_to_dict(status)


@router.post("/training/activate")
def activate() -> dict:
    try:
        n = activate_checkpoints()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    if n == 0:
        raise HTTPException(
            status_code=404,
            detail="No fine-tuned checkpoints found. Run a training job first.",
        )
    return {"activated": n, "checkpoints": _checkpoint_status_to_dict(checkpoint_status())}


@router.post("/training/revert")
def revert() -> dict:
    try:
        revert_to_release()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return {"checkpoints": _checkpoint_status_to_dict(checkpoint_status())}


@router.get("/training/metrics")
def get_metrics(splits: str = SPLIT_BOTH) -> dict:
    return read_training_metrics(str(TRAINING_OUTPUT_DIR), splits=splits)
