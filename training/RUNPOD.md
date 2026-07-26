# Running the first fine-tune on RunPod

The **code** travels through git; the **dataset** (~25 GB of FLAC) does not — git
can't carry it and `.gitignore` deliberately excludes `training_data/`. So the
flow is: pull the repo on the pod, ship the audio separately, point the trainer
at it.

## Don't rebuild the image to change code

The training image is large because it bakes in the A2SB framework and the three
multi-GB release checkpoints — none of which change when you edit a script. So
treat the image as a **fixed runtime** and get the code from git instead:

- `finetune.py` resolves its configs **next to itself**, so a cloned checkout
  uses its own `training/configs/*.yaml`.
- It finds the baked-in framework and weights through `A2SB_APP_ROOT`
  (default `/app`) and `A2SB_CKPT_DIR` (default `/app/ckpts`).

Net effect: `git pull` on the pod is enough to pick up code changes. Rebuild the
image only when the *dependencies or checkpoints* change.

(The image still contains a copy of `training/` as a working default — the
cloned copy simply takes precedence when you run it directly.)

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

Run the **cloned** copy of the script inside the training container. It picks up
its own configs and uses the framework + checkpoints baked into the image:

```bash
python /workspace/A2SB-Upscaler-Docker/training/finetune.py \
    --data-dir /workspace/training_data \
    --output-dir /workspace/training_output \
    --splits both \
    --steps 5000 \
    --batch-size 2 \
    --learning-rate 0.00005
```

If the image's install lives somewhere other than `/app` (or you mounted the
weights on a volume to keep the image small), point at it:

```bash
export A2SB_APP_ROOT=/app          # dir holding main.py
export A2SB_CKPT_DIR=/workspace/ckpts   # dir holding the release .ckpt files
```

If you set `PYTORCH_CUDA_ALLOC_CONF` yourself, use `expandable_segments:True`
**alone**. Combining it with `max_split_size_mb` mixes two incompatible
allocation strategies (expandable segments cannot be split) and crashes a few
steps into training with:

```
RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED at
"../c10/cuda/CUDACachingAllocator.cpp", please report a bug to PyTorch.
```

To iterate: edit locally, push, then `git pull` in
`/workspace/A2SB-Upscaler-Docker` on the pod and re-run. No image rebuild.

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

## Tracking metrics with Weights & Biases

The base config leaves `trainer.logger` null, so Lightning falls back to
`CSVLogger` and metrics land in `<output-dir>/split_*/lightning_logs/version_*/metrics.csv`.
That survives fine, but not if the pod does -- W&B keeps the history off-pod,
which matters when pods churn.

```bash
pip install wandb          # already in the trainer image
wandb login                # or: export WANDB_API_KEY=...
```

Then add `--wandb` to the run:

```bash
python training/finetune.py \
    --data-dir /workspace/training_data \
    --output-dir /root/training_output \
    --splits both --steps 5000 --wandb --wandb-project a2sb-finetune
```

Each split is logged as its own run (`<prefix>-split_0.0_0.5`,
`<prefix>-split_0.5_1.0`), so `--splits both` gives two comparable curves.
`--wandb-run-name` sets the prefix; it defaults to the output directory name.

Note that the useful audio metrics (`val_lsd`, `val_sisdr`, the per-timestep
`val_loss_t=*`) only appear if validation actually runs -- see `--val-samples`
and `--val-every`, which exist because a full validation cycle is expensive.
