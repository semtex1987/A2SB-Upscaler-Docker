#!/bin/bash
set -e
# Download release checkpoints once onto persistent storage. Existing files are
# kept; downloads go to a temp name and are renamed only after wget succeeds so
# a dropped connection cannot truncate a previously good file.

CKPT_DIR=/app/ckpts
mkdir -p "$CKPT_DIR"
HF=https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt

download_ckpt() {
  local dest="$1"
  local url="$2"
  local name
  name="$(basename "$dest")"
  if [[ -s "$dest" ]]; then
    echo "[entrypoint] using cached $name"
    return 0
  fi
  local tmp="${dest}.partial.$$"
  echo "[entrypoint] downloading $name"
  if wget -O "$tmp" "$url"; then
    mv -f "$tmp" "$dest"
    return 0
  fi
  rm -f "$tmp"
  if [[ -s "$dest" ]]; then
    echo "[entrypoint] download failed; keeping existing $name"
    return 0
  fi
  echo "[entrypoint] ERROR: missing $name and download failed" >&2
  return 1
}

download_ckpt "$CKPT_DIR/A2SB_twosplit_0.5_1.0_release.ckpt" "$HF/A2SB_twosplit_0.5_1.0_release.ckpt"
download_ckpt "$CKPT_DIR/A2SB_twosplit_0.0_0.5_release.ckpt" "$HF/A2SB_twosplit_0.0_0.5_release.ckpt"
# The one-split checkpoint is unused by the two-split ensemble; cache it if
# present but do not fail startup when offline.
download_ckpt "$CKPT_DIR/A2SB_onesplit_0.0_1.0_release.ckpt" "$HF/A2SB_onesplit_0.0_1.0_release.ckpt" || true

# If fine-tuned checkpoints exist under the training export directory
# (/app/outputs/training/checkpoints, or $A2SB_FINETUNED_CKPT_DIR), point the
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
