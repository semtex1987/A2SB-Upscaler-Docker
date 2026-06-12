# A2SB Fine-Tuning

Drop high-quality audio (44.1 kHz WAV/FLAC, etc.) into `training_data/`, then run the trainer to fine-tune both A2SB splits. Fine-tuned checkpoints are written to `training_output/checkpoints/` and are picked up by the inference container when present.

## Quick start

1. Put audio files in `training_data/` (at repo root).
2. Build and run fine-tuning (both splits, 5000 steps each):

   ```bash
   docker compose -f training/docker-compose.train.yml run trainer \
     python /app/training/finetune.py --steps 5000
   ```

3. Restart the inference container to use the new checkpoints:

   ```bash
   docker compose down && docker compose up -d
   ```

## Options

- `--data-dir`, `--output-dir`: Override paths (defaults: `/data/training_data`, `/data/training_output`).
- `--steps`: Max steps per split (default 5000).
- `--batch-size`: Batch size (default 2; increase if GPU has 48GB+).
- `--learning-rate`: Default 5e-5.
- `--splits`: `both`, `0.0-0.5`, or `0.5-1.0` to train one split only.
- `--val-frac`: Validation fraction (default 0.1).

Example: train only the first split for 10k steps with a larger batch:

```bash
docker compose -f training/docker-compose.train.yml run trainer \
  python /app/training/finetune.py --splits 0.0-0.5 --steps 10000 --batch-size 4
```

## Data requirements

Training audio must be genuinely full-bandwidth (lossless or CD-quality sources, **not** MP3-derived). The manifest now records each file's estimated true bandwidth (2× the 99th-percentile spectral rolloff). Files whose estimated bandwidth falls below 16 kHz will be excluded by the `apply_sr_loss_mask` filter in the dataset config — band-limited training material would otherwise teach the model to output silence in the high band.

`--steps` is relative to the **resumed** global step of the release checkpoint. The previous behavior started counting from the checkpoint's existing step count and stopped immediately because `max_steps` was already exceeded. The updated `finetune.py` offsets `max_steps` by the checkpoint's `global_step` so the requested number of steps is always trained.

## Shell access

To get a shell inside the training container:

```bash
docker compose -f training/docker-compose.train.yml run trainer /bin/bash
```

Then run `python /app/training/finetune.py ...` manually.
