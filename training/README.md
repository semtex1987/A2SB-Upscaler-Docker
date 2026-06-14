# A2SB Fine-Tuning

Drop high-quality, genuinely full-bandwidth audio into `training_data/`, vet it, then run the trainer to fine-tune both A2SB splits. Fine-tuned checkpoints are written to `training_output/checkpoints/` and are picked up by the inference container when present.

## Quick start

1. Put audio files in `training_data/` (at repo root).
2. (Optional but recommended) Vet the dataset — see [Dataset vetting](#dataset-vetting).
3. Build and run fine-tuning (both splits, 5000 new steps each):

   ```bash
   docker compose -f training/docker-compose.train.yml run trainer \
     python /app/training/finetune.py --steps 5000
   ```

4. Restart the inference container to use the new checkpoints:

   ```bash
   docker compose down && docker compose up -d
   ```

## Dataset vetting

Before spending GPU time on training, verify that your audio is genuinely full-bandwidth. **Lossy files in a lossless container** (MP3-sourced FLACs, re-encoded WAVs, etc.) have a hard spectral cutoff well below 22 kHz. Training on them teaches the model to output silence in the high band rather than regenerating it.

Run the vetting tool against your staging directory — it works on your machine if you have `numpy` and `librosa`, or inside the trainer container against the mounted data volume:

```bash
# On your audio-staging machine
python3 training/vet_dataset.py /path/to/your/flacs

# Or inside the trainer container
docker compose -f training/docker-compose.train.yml run trainer \
    python /app/training/vet_dataset.py /data/training_data --csv /data/vet_report.csv
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
| `--steps` | 5000 | New training steps per split. Added to the checkpoint's existing `global_step`; the requested count is always trained regardless of where the release checkpoint starts. |
| `--data-dir` | `/data/training_data` | Directory containing audio files (scanned recursively). |
| `--output-dir` | `/data/training_output` | Destination for the manifest CSV and checkpoint subdirectories. |
| `--batch-size` | 2 | Training batch size. Increase to 4 on 48 GB+ GPUs. |
| `--learning-rate` | 5e-5 | Learning rate. Enforced at the start of each run regardless of what the resumed checkpoint stored. |
| `--splits` | `both` | Which split(s) to train: `both`, `0.0-0.5`, or `0.5-1.0`. |
| `--val-frac` | 0.1 | Fraction of files held out for validation (minimum 1 file). |
| `--seed` | 42 | Random seed for the train/validation split. |

Example — train only the first split for 10k steps with a larger batch:

```bash
docker compose -f training/docker-compose.train.yml run trainer \
  python /app/training/finetune.py --splits 0.0-0.5 --steps 10000 --batch-size 4
```

## How the training pipeline works

### Step 1 — Manifest generation

`finetune.py` scans `--data-dir` recursively for supported audio formats (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`), skips files shorter than ~3 seconds, and writes `finetune_manifest.csv` under `--output-dir`. The manifest has four columns:

```
split, filepath, duration, estimated_true_sr
```

`estimated_true_sr` is `2 × 95th-percentile(per-frame 0.99 spectral rolloff)`, capped at 44100 Hz. Files estimated below 32000 Hz (16 kHz effective bandwidth) are flagged with a warning — they will be silently excluded during training by the `apply_sr_loss_mask` filter in the dataset config.

### Step 2 — Fine-tuning

`finetune.py` runs `main.py fit` twice (once per split) using the two release checkpoints as starting points. Key details:

- **Step count**: The release checkpoints are full Lightning checkpoints carrying a large `global_step`. The pipeline reads this value and sets `--trainer.max_steps = checkpoint_global_step + args.steps`, so exactly `--steps` new gradient steps are always taken.
- **`val_check_interval`**: Automatically clamped to `min(1000, batches_per_epoch)` where `batches_per_epoch` is computed from the manifest's train segment count. This prevents Lightning from raising a `ValueError` on small datasets (<100 minutes of audio at batch size 2).
- **Mask focus**: The fine-tune configs (`training/configs/finetune_split*.yaml`) concentrate 100% of training steps on the upsample mask task (filling 8–16 kHz from realistic cutoffs), matching what inference actually does. Inpainting is disabled during fine-tuning.
- **Checkpointing**: Each split saves a `last.ckpt` and the best checkpoint by validation loss under `training_output/split_0.0_0.5/` and `training_output/split_0.5_1.0/`. The latest checkpoint from each split is then copied to `training_output/checkpoints/` with the names the inference container expects.

### Step 3 — Checkpoint pickup

After fine-tuning, restart the inference container (see Quick start step 4). If the inference `docker-compose.yml` mounts `./training_output/checkpoints:/app/ckpts/finetuned:ro`, the container's startup script detects both finetuned checkpoints and updates the ensemble config to use them automatically.

## Data preparation notes

- **Feed files as-is** — do not trim silences or concatenate tracks. Silent passages dilute step budget but don't corrupt training. Splicing across removed gaps risks introducing clicks at boundaries.
- **Sample rate** — files are loaded at their native sample rate and resampled by the dataloader. 44.1 kHz originals are ideal; 48 kHz files also work.
- **Mono vs stereo** — the manifest builder loads mono for bandwidth estimation; the dataloader handles stereo internally.
- **Minimum length** — files shorter than one segment (~3 seconds at 44.1 kHz) are skipped by the manifest builder.

## Shell access

To get a shell inside the training container:

```bash
docker compose -f training/docker-compose.train.yml run trainer /bin/bash
```

Then run `python /app/training/finetune.py ...` or `python /app/training/vet_dataset.py ...` manually.
