#!/usr/bin/env bash
# Prepare a fresh RunPod PyTorch pod for A2SB fine-tuning.
#
#   bash setup_pod.sh            # full setup
#   bash setup_pod.sh --verify   # only re-run the checks
#
# Safe to re-run: every step is skipped if it is already done.
#
# The pod image supplies torch and CUDA; this script adds the rest, fetches the
# release checkpoints, and verifies the combination actually imports. Several of
# the steps exist because of specific failures that are easy to hit and hard to
# read once training has started -- those are called out inline.
set -uo pipefail

REPO_URL="${A2SB_REPO_URL:-https://github.com/semtex1987/A2SB-Upscaler-Docker.git}"
WORK="${A2SB_WORK_DIR:-/workspace}"
REPO_DIR="$WORK/A2SB-Upscaler-Docker"
CKPT_DIR="$WORK/ckpts"
DATA_DIR="$WORK/training_data"
OUT_DIR="${A2SB_OUT_DIR:-/root/training_output}"
TMP_DIR="$OUT_DIR/tmp"
HF="https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '   \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '   \033[33mwarn\033[0m %s\n' "$1"; }
fail() { printf '   \033[31mFAIL\033[0m %s\n' "$1"; }

verify() {
    local rc=0
    step "Verifying"

    # This is the exact import chain that fails when torchaudio does not match
    # torch: libtorchaudio.so links against libtorch, and a mismatch surfaces as
    # "undefined symbol", not as a version warning.
    if python3 - <<'PY'
import sys
import torch
print(f"   torch      {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
import torchaudio
print(f"   torchaudio {torchaudio.__version__}")
if torchaudio.__version__.split('+')[0] != torch.__version__.split('+')[0]:
    print("   torchaudio/torch version mismatch", file=sys.stderr); sys.exit(1)
import librosa, soundfile      # noqa: F401  -- without these every file reads as unreadable
from rotary_embedding_torch import RotaryEmbedding   # noqa: F401
PY
    then ok "python imports"; else fail "python imports"; rc=1; fi

    if [ -f "$REPO_DIR/nvidia-a2sb-original-repo/main.py" ]; then
        ok "A2SB source"
    else fail "A2SB source missing at $REPO_DIR/nvidia-a2sb-original-repo/main.py"; rc=1; fi

    # Import exactly what main.py imports. A narrower check (e.g. just networks)
    # passes while the training run still dies on a module reached only through
    # the lightning module -- matplotlib via plotting_utils being the example
    # that motivated this.
    if (cd "$REPO_DIR/nvidia-a2sb-original-repo" 2>/dev/null && python3 - <<'PYCHK' 2>&1
from A2SB_lightning_module import STFTBridgeModel, LogValidationInpaintingSTFTCallback
from datasets.datamodule import STFTAudioDataModule
from lightning.pytorch.cli import LightningCLI
PYCHK
    ); then
        ok "A2SB training imports"
    else fail "A2SB training imports (see error above)"; rc=1; fi

    local missing=0
    for f in A2SB_twosplit_0.0_0.5_release.ckpt A2SB_twosplit_0.5_1.0_release.ckpt; do
        if [ -s "$CKPT_DIR/$f" ]; then ok "checkpoint $f ($(du -h "$CKPT_DIR/$f" | cut -f1))"
        else fail "checkpoint missing: $CKPT_DIR/$f"; missing=1; rc=1; fi
    done
    [ "$missing" -eq 0 ] || warn "re-run without --verify to download"

    if [ -d "$DATA_DIR" ]; then
        ok "training data: $(find "$DATA_DIR" -iname '*.flac' -o -iname '*.wav' 2>/dev/null | wc -l) audio files"
    else warn "no $DATA_DIR yet -- ship your dataset there before training"; fi

    return $rc
}

if [ "${1:-}" = "--verify" ]; then verify; exit $?; fi

step "System packages"
# ffmpeg/libsndfile back librosa+soundfile; without them every audio file is
# reported unreadable. tmux because training runs for hours and an SSH drop
# would otherwise kill it.
if command -v ffmpeg >/dev/null && command -v tmux >/dev/null; then
    ok "already present"
else
    apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 tmux git wget >/dev/null \
        && ok "installed ffmpeg libsndfile1 tmux git wget" || fail "apt-get failed"
fi

step "Repository"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only -q && ok "updated $REPO_DIR" || warn "pull failed; keeping existing checkout"
else
    git clone -q "$REPO_URL" "$REPO_DIR" && ok "cloned to $REPO_DIR" || fail "clone failed"
fi

step "Python packages"
# torch comes from the pod image and is left alone. torchaudio MUST match it
# exactly, so it is derived rather than pinned -- an unpinned install pulls a
# newer build and dies later with an undefined-symbol OSError.
TORCH_VER=$(python3 -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
TORCH_CU=$(python3 -c "import torch;print((torch.version.cuda or '').replace('.',''))" 2>/dev/null || echo "")
if [ -z "$TORCH_VER" ]; then
    fail "torch not importable -- is this the PyTorch pod image?"
else
    ok "pod torch $TORCH_VER (cuda $TORCH_CU)"
    if python3 -c "
import sys,torch,torchaudio
sys.exit(0 if torchaudio.__version__.split('+')[0]==torch.__version__.split('+')[0] else 1)" 2>/dev/null; then
        ok "torchaudio already matches"
    else
        IDX=""
        [ -n "$TORCH_CU" ] && IDX="--index-url https://download.pytorch.org/whl/cu${TORCH_CU}"
        pip install -q --no-cache-dir "torchaudio==${TORCH_VER}" $IDX \
            && ok "installed torchaudio==${TORCH_VER}" \
            || warn "could not install matching torchaudio; check manually"
    fi
fi

# Left unpinned so pip can resolve against whatever torch the pod ships; the
# verify step below is what actually confirms the combination works.
pip install -q --no-cache-dir \
    numpy scipy matplotlib moviepy "jsonargparse[signatures]" scikit-image \
    torchlibrosa pyyaml librosa soundfile einops pytorch_lightning lightning \
    rotary_embedding_torch tqdm wandb tensorboard >/dev/null \
    && ok "installed training deps" || warn "some deps failed to install"
# ssr_eval declares a dependency on 'wave', which is a stdlib module and cannot
# be installed; --no-deps is the only way it installs at all.
pip install -q --no-cache-dir --no-deps ssr_eval >/dev/null && ok "installed ssr_eval" || warn "ssr_eval failed"

step "Checkpoints"
mkdir -p "$CKPT_DIR"
# Only the two twosplit checkpoints are used for training. The onesplit one is
# inference-only, so skipping it saves a couple of GB and a slow download.
for f in A2SB_twosplit_0.0_0.5_release.ckpt A2SB_twosplit_0.5_1.0_release.ckpt; do
    if [ -s "$CKPT_DIR/$f" ]; then
        ok "$f already present"
    else
        wget -q --show-progress -O "$CKPT_DIR/$f" "$HF/$f" \
            && ok "downloaded $f" \
            || { fail "download failed: $f"; rm -f "$CKPT_DIR/$f"; }
    fi
done

step "Directories and environment"
mkdir -p "$DATA_DIR" "$OUT_DIR" "$TMP_DIR"
ok "created $DATA_DIR, $OUT_DIR, $TMP_DIR"

# Written to .bashrc because these must be set in every shell -- a new tmux pane
# or a reconnect otherwise loses them, and finetune.py then looks under /app.
if ! grep -q 'A2SB_APP_ROOT' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

# --- A2SB fine-tuning (added by setup_pod.sh) ---
export A2SB_APP_ROOT=$REPO_DIR/nvidia-a2sb-original-repo
export A2SB_CKPT_DIR=$CKPT_DIR
export MKL_THREADING_LAYER=GNU
# expandable_segments ALONE. Combining it with max_split_size_mb mixes two
# incompatible allocator strategies and aborts a few steps into training with
# "!block->expandable_segment_ INTERNAL ASSERT FAILED".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep the atomic-save temp file on the same filesystem as the output, or every
# multi-GB checkpoint write becomes a cross-device copy.
export TMPDIR=$TMP_DIR
EOF
    ok "environment written to ~/.bashrc"
else
    ok "environment already in ~/.bashrc"
fi

export A2SB_APP_ROOT="$REPO_DIR/nvidia-a2sb-original-repo"
export A2SB_CKPT_DIR="$CKPT_DIR"
export MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$TMP_DIR"

verify
VERIFY_RC=$?

step "Next steps"
cat <<EOF
   1. source ~/.bashrc
   2. Ship your dataset to $DATA_DIR
   3. tmux new -s train
   4. python3 $REPO_DIR/training/finetune.py \\
          --data-dir $DATA_DIR \\
          --output-dir $OUT_DIR \\
          --splits 0.0-0.5 --steps 5000 --batch-size 16 \\
          --learning-rate 0.0001 --val-samples 16 --val-every 4000 \\
          -- --trainer.precision bf16-mixed

   $OUT_DIR is on the pod's local disk (fast, and not the flaky network volume),
   but it is EPHEMERAL -- copy checkpoints to $WORK when a split finishes.
EOF

exit $VERIFY_RC
