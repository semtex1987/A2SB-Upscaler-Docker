#!/bin/bash
set -e

# If fine-tuned checkpoints are mounted at /app/ckpts/finetuned, point the
# ensemble config at them instead of the release checkpoints.
python3 /app/update_ckpt_config.py || true

# Bind-mounted volumes inherit host ownership, which may not match appuser
# (UID 1000).  Fix at startup so the app can read/write both directories.
# /debug is Lightning's CSVLogger root_dir (set by ensembled_inference_api.py).
chown appuser:appuser /app/inputs /app/outputs /debug

exec runuser -u appuser -- "$@"
