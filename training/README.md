# A2SB Fine-Tuning

Drop high-quality, genuinely full-bandwidth audio into `training_data/`, vet it, then run the trainer to fine-tune both A2SB splits.

## Quick start — via the Train tab (no separate container needed)

The web UI's **Train** tab is the easiest path and does not require a second Docker container.

1. Put audio files in `training_data/` (at repo root — bind-mounted to `/app/training_data`).
2. Open the Train tab, enter the directory path, and click **Scan & vet**.
3. Review the PASS/CHECK/REJECT breakdown, configure steps and batch size, then click **Start fine-tuning**.
4. Watch the loss curves in the **Loss curves** panel, or click **Launch TensorBoard** for the full dashboard. TensorBoard is proxied at `/tensorboard/` on the app's own port, so no extra port needs publishing.
5. When training completes, click **Activate fine-tuned** — no container restart needed.

Progress bars are relative to the splits you selected: a `0.5-1.0`-only run starts at 0% of that one split, not 100%. The displayed step interpolates from the split header's resume offset to its `until` target. Validation/sanity-check bars are logged but do not jump the overall percentage. Restoration jobs similarly read a `Sampling:` tqdm from the diffusion loop itself, not Lightning's one-item predict bar.

## Quick start — via CLI (separate training container)

1. Put audio files in `training_data/` (at repo root).
2. (Optional but recommended) Vet the dataset — see [Dataset vetting](#dataset-vetting).
3. Build and run fine-tuning (both splits, 5000 new steps each):

   ```bash
   docker compose -f training/docker-compose.train.yml run trainer \
     python /app/training/finetune.py --steps 5000
   ```

4. Use the Train tab's **Activate fine-tuned** button to activate the checkpoints in the running inference container, or restart:

   ```bash
   docker compose down && docker compose up -d
   ```

## Dataset vetting

Before spending GPU time on training, verify that your audio is genuinely full-bandwidth. **Lossy files in a lossless container** (MP3-sourced FLACs, re-encoded WAVs, etc.) have a hard spectral cutoff well below 22 kHz. Training on them teaches the model to output silence in the high band rather than regenerating it.

Run the vetting tool against your staging directory — it works on your machine if you have `numpy` and `librosa`, or inside the trainer container against the mounted data volume:

```bash
# On your audio-staging machine
python3 training/vet_dataset.py /path/to/your/flacs

# Or inside the trainer container (writes `report.csv` into each audio folder;
# override the filename with `--report-name`)
docker compose -f training/docker-compose.train.yml run trainer \
    python /app/training/vet_dataset.py /data/training_data
```

### Output columns

| Column | Meaning |
|--------|---------|
| `edge` | Highest frequency (Hz) carrying real energy above the file's own noise floor — where HF content actually stops |
| `roll95` | 95th-percentile of per-frame 0.99 spectral rolloff (Hz) — the underlying bandwidth estimate |
| `est_sr` | `2 × roll95`, capped at 44100 — same value `finetune.py` writes to the manifest |
| `shelf` | **Y** if a steep cliff (>40 dB drop over <1 kHz) is found at `edge` — the brickwall lowpass signature of a transcode |
| `gate` | **keep** / **DROP** — whether the file would survive the `apply_sr_loss_mask` filter during training |
| `verdict` | **PASS** (≥20.5 kHz), **CHECK** (17–20.5 kHz), **REJECT** (<17 kHz) |

**PASS** files are ready to use. **CHECK** files have edge content between 17 and 20.5 kHz — this can be either a genuine but dark master or a 320 kbps transcode; use the `shelf` flag and your ear to decide. **REJECT** files are band-limited; remove them.

### Recommended genres for a compact training set

The A2SB model regenerates frequencies above the cutoff. To teach it a wide palette of HF primitives with a minimal dataset:

| Genre | What it contributes |
|-------|-------------------|
| Orchestral classical | Harmonic overtone series extending above 14 kHz (strings, brass, woodwinds) |
| Acoustic jazz | Cymbal shimmer, transient attack (high temporal + spectral resolution) |
| Vocal / acoustic folk | Sibilance, breath noise, acoustic guitar harmonics |
| Electronic / synth | Synthetic HF textures, saw/square harmonics that reach Nyquist |
| Rock / metal | Saturated guitar harmonics, drum transients across the full spectrum |

Aim for 30–60 minutes of fully vetted, PASS-rated material per genre. This is enough to shift the model toward better HF generation without overfitting.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 5000 | **Additional** optimizer steps per split (`trainer.max_steps = global_step + N`). |
| `--until-step` | unset | Absolute Lightning `max_steps` target. Overrides `--steps` when set. |
| `--export-dir` | `<output-dir>/checkpoints` | Directory for inference-named checkpoint copies. The GUI points this at the shared activation registry. |
| `--data-dir` | `/data/training_data` | Directory containing audio files (scanned recursively). |
| `--output-dir` | `/data/training_output` | Destination for `run.json`, the manifest CSV, and per-split checkpoint subdirectories. GUI jobs use `<training-output>/runs/<job-id>/`. |
| `--batch-size` | 2 | Training batch size. Increase to 4 on 48 GB+ GPUs. |
| `--learning-rate` | 5e-5 | Learning rate. Enforced at the start of each run regardless of what the resumed checkpoint stored. |
| `--splits` | `both` | Which split(s) to train: `both`, `0.0-0.5`, or `0.5-1.0`. |
| `--val-frac` | 0.1 | Fraction of files held out for validation (minimum 1 file). |
| `--val-every` | config | Override `val_check_interval` (optimizer steps between validation). |
| `--val-samples` | 32 (YAML) | Cap on validation segments (`--data.val_max_samples`). The fine-tune configs default to 32. |
| `--seed` | 42 | Random seed for the train/validation split. |
| `--devices` | config (`1`) | GPU count. Values above 1 switch to DDP with Lightning's distributed sampler so each rank sees a shard of the dataset. |

Example — train only the first split for 10k steps with a larger batch:

```bash
docker compose -f training/docker-compose.train.yml run trainer \
  python /app/training/finetune.py --splits 0.0-0.5 --steps 10000 --batch-size 4
```

## How the training pipeline works

### Step 1 — Manifest generation

`finetune.py` scans `--data-dir` recursively for supported audio formats (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`), skips files shorter than ~3 seconds, and writes `finetune_manifest.csv` under `--output-dir`. The dataloader then reads **only the requested segment** (plus 50 ms of resampling context) rather than decoding the whole track on every step.

```
split, filepath, duration, estimated_true_sr
```

`estimated_true_sr` is `2 × 95th-percentile(per-frame 0.99 spectral rolloff)`, capped at 44100 Hz — the same estimator used by GUI vetting (`server/training.py`) and `training/vet_dataset.py`. Files estimated below 32000 Hz (16 kHz of real bandwidth) are **excluded before the train/validation split**. Manifest counts are therefore the files `MixAudioDataset` will keep when `apply_sr_loss_mask` is on. If either split would be empty, `finetune.py` exits instead of launching a loader with zero samples. `val_check_interval` is derived from those loader-eligible train segments.

### Step 2 — Fine-tuning

`finetune.py` runs `main.py fit` twice (once per split) using the two release checkpoints as starting points. Key details:

- **Step count**: `--steps N` means **N additional** optimizer steps from the starting checkpoint (`trainer.max_steps = global_step + N`). `--until-step N` is the absolute Lightning `max_steps` target and overrides `--steps` when set. Resuming a run that is already past the absolute target does no further work — raise `--steps` or `--until-step`.
- **Run directories**: GUI jobs write to `<training-output>/runs/<job-id>/` (never the shared training root). A new job does not auto-resume an earlier one. Resume is explicit: point `--output-dir` at an existing run, or choose **Resume from** in the Train tab. Each run writes `run.json` with dataset path, step plan, and hyperparameters. GUI exports still copy trained weights to the shared activation directory via `--export-dir`.
- **`val_check_interval`**: Automatically clamped to `min(1000, batches_per_epoch)` where `batches_per_epoch` is the per-rank loader length (`segments / (batch_size × devices)`). This prevents Lightning from raising a `ValueError` on small datasets, including multi-GPU runs where each rank sees a shard of the data.
- **Multi-GPU**: `--devices N` (N>1) uses DDP with Lightning's distributed sampler enabled so ranks do not traverse the full dataset. The fine-tune YAMLs set `use_distributed_sampler: true`. Saturate one GPU with `--batch-size` before adding devices.
- **Mask focus**: The fine-tune configs (`training/configs/finetune_split*.yaml`) concentrate 100% of training steps on the upsample mask task (filling 8–16 kHz from realistic cutoffs), matching what inference actually does. Inpainting is disabled during fine-tuning.
- **Checkpointing**: Each split saves a `last.ckpt` and periodic checkpoints under `training_output/split_0.0_0.5/` and `training_output/split_0.5_1.0/`. `ModelCheckpoint` **monitors `global_step`**, not validation loss — the kept files are the latest steps, not the best `val_loss`. Export copies a *trained* artifact (`last.ckpt` if it has a completed optimizer step) to `training_output/checkpoints/` with the names the inference container expects. The initialization stub `finetune_start.ckpt` is never exported or selected for resume.
- **Optimizer**: Training uses RAdam with AdamW-style (decoupled) weight decay. PyTorch 2.2 added a `decoupled_weight_decay` flag that the Docker-pinned 2.1 RAdam rejects; `training_optimizer.py` passes that flag when the installed torch supports it and otherwise applies the same decay before the adaptive step.
- **Gradient clipping**: The Trainer configs set `gradient_clip_val: 0.5`. Clipping is not applied inside `training_step` — that mutated leftover gradients during `accumulate_grad_batches`. The logged `train/gradient_norm` is the global norm at the optimizer-step boundary, after Lightning's clip.
- **CSV metrics**: Each split's Trainer is given an explicit `CSVLogger` with `init_args.save_dir` set to that split's output directory. Passing `--trainer.logger=CSVLogger` on the CLI is not used: Lightning 2.x requires `save_dir` on the constructor, and `default_root_dir` does not supply it.

### Step 3 — Checkpoint pickup

After fine-tuning, click **Activate fine-tuned** in the Train tab (no restart). The rewrite is atomic (temp file + rename). Exports land in `<output-dir>/checkpoints/` — the same directory status and activation read. Restoration jobs copy that YAML when they start, so Left and Right cannot diverge if you activate again while a file is running. The default GUI output is `/app/outputs/training`, already on the restored-audio volume. A CLI training container that writes to `./training_output` can either copy those files into `restored_audio/training/checkpoints/` or set `A2SB_FINETUNED_CKPT_DIR` to the mounted tree.

## Data preparation notes

- **Feed files as-is** — do not trim silences or concatenate tracks. Silent passages dilute step budget but don't corrupt training. Splicing across removed gaps risks introducing clicks at boundaries.
- **Sample rate** — files are loaded at their native sample rate and resampled by the dataloader (`librosa.resample`). The training image pins `soxr` so that downsampling is band-limited. 44.1 kHz originals are ideal; 48 kHz files also work.
- **Mono vs stereo** — the manifest builder loads mono for bandwidth estimation; the dataloader handles stereo internally.
- **Minimum length** — files shorter than one segment (~3 seconds at 44.1 kHz) are skipped by the manifest builder.

## Shell access

To get a shell inside the training container:

```bash
docker compose -f training/docker-compose.train.yml run trainer /bin/bash
```

Then run `python /app/training/finetune.py ...` or `python /app/training/vet_dataset.py ...` manually.
