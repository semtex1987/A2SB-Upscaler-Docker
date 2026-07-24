# Running the first fine-tune on RunPod

The **code** travels through git; the **dataset** (~25 GB of FLAC) does not — git
can't carry it and `.gitignore` deliberately excludes `training_data/`. So the
flow is: pull the repo on the pod, ship the audio separately, point the trainer
at it.

## The first-run dataset

`training/first_run_dataset.csv` is the manifest of the curated first-run set:
**529 tracks across 44 albums**, selected by `training/vet_dataset.py` in its
default *authenticity* mode (genuine masters only — lossy transcodes, CD→hi-res
upsamples, and DSD-rate hash are rejected; see the header of `vet_dataset.py`).
Genre spread is broad (Rock, Electronic, Pop, Jazz, Vocal, Folk/Bluegrass,
Hip-hop, Classical, …) and no single artist dominates. The `hf_edge` column is
how bright each track genuinely is — informational, not a gate.

Only authentic material is included, so nothing here will surprise-fail the
trainer's own 16 kHz gate.

## 1. Pull the repo on the pod

```bash
git clone https://github.com/semtex1987/A2SB-Upscaler-Docker.git
cd A2SB-Upscaler-Docker
```

## 2. Get the dataset onto the pod

The audio lives in `A2SB_first_run_dataset/` on the staging machine (a sibling of
this repo). Move it to the pod by whichever route fits — e.g.:

- **runpodctl (peer-to-peer, good for a one-off):**
  ```bash
  # on the staging machine
  runpodctl send A2SB_first_run_dataset
  # on the pod (paste the code it prints)
  runpodctl receive <code>
  ```
- **Cloud bucket (repeatable):** `rclone`/`aws s3 sync` the folder up, then down
  onto the pod.

Land it at e.g. `/workspace/training_data` (preserving the per-album
sub-folders — the trainer scans recursively).

## 3. Launch fine-tuning

The training container (`training/Dockerfile.train`) is self-contained: it
installs the deps and downloads the three A2SB release checkpoints. On a RunPod
pod that already has the NVIDIA runtime you can either build/run that image, or
run `finetune.py` directly in a PyTorch pod after installing the deps listed in
`Dockerfile.train`.

```bash
python training/finetune.py \
    --data-dir /workspace/training_data \
    --output-dir /workspace/training_output \
    --splits both \
    --steps 5000 \
    --batch-size 2 \
    --learning-rate 0.00005
```

`finetune.py` builds its own manifest by scanning `--data-dir` (estimating each
file's true sample rate and dropping anything under 16 kHz), fine-tunes the two
A2SB split checkpoints, and writes checkpoints + manifest to `--output-dir`.

## 4. (Optional) re-vet on the pod

The deps in `Dockerfile.train` (numpy, librosa, soundfile) are all `vet_dataset.py`
needs, so you can re-run the vetting there against the mounted data dir:

```bash
python training/vet_dataset.py /workspace/training_data
```

Each folder already carries a `report.csv` from the staging-side vetting; re-run
only if you add material. For DSD/SACD sources, decimate to 44.1 kHz first
(`aresample=resampler=soxr` in ffmpeg) — the vetter flags raw high-rate files as
CHECK because their ultrasonic band may be noise-shaping hash, not music.
