#!/bin/bash
set -e

# If fine-tuned checkpoints are mounted at /app/ckpts/finetuned, point the
# ensemble config at them instead of the release checkpoints.
python3 /app/update_ckpt_config.py || true

# Bind-mounted volumes inherit host ownership, which may not match appuser
# (UID 1000).  Fix at startup so the app can read/write both directories.
# /debug is Lightning's CSVLogger root_dir (set by ensembled_inference_api.py).
if ! chown appuser:appuser /app/inputs /app/outputs /debug 2>/dev/null; then
  echo "[entrypoint] Warning: could not chown one or more runtime directories; checking writability."
fi

if runuser -u appuser -- test -w /app/inputs && \
   runuser -u appuser -- test -w /app/outputs && \
   runuser -u appuser -- test -w /debug; then
  exec runuser -u appuser -- "$@"
fi

echo "[entrypoint] Warning: runtime directories are not writable by appuser; continuing as $(id -un)."
exec "$@"
