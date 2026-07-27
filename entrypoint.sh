#!/bin/bash
set -e
# 5. Download Checkpoints
wget -O /app/ckpts/A2SB_twosplit_0.5_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt
wget -O /app/ckpts/A2SB_onesplit_0.0_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt
wget -O /app/ckpts/A2SB_twosplit_0.0_0.5_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.0_0.5_release.ckpt
# If fine-tuned checkpoints are mounted at /app/ckpts/finetuned, point the
# ensemble config at them instead of the release checkpoints.
python3 /app/update_ckpt_config.py || true

# Bind-mounted volumes inherit host ownership, which may not match appuser
# (UID 1000).  Fix at startup so the app can read/write all runtime directories.
# /debug is Lightning's CSVLogger root_dir (set by ensembled_inference_api.py).
# /app/training_data is the user-supplied audio for fine-tuning.
if ! chown appuser:appuser /app/inputs /app/outputs /app/training_data /debug 2>/dev/null; then
  echo "[entrypoint] Warning: could not chown one or more runtime directories; checking writability."
fi

if runuser -u appuser -- test -w /app/inputs && \
   runuser -u appuser -- test -w /app/outputs && \
   runuser -u appuser -- test -w /debug; then
  exec runuser -u appuser -- "$@"
fi

echo "[entrypoint] Warning: runtime directories are not writable by appuser; continuing as $(id -un)."
exec "$@"
