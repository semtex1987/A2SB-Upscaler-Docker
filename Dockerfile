# ---------------------------------------------------------------------------
# Stage 1: build the web frontend.
# Kept separate so the ~200 MB of node_modules never reaches the runtime image.
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS web

WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: the runtime image.
# ---------------------------------------------------------------------------
# Start from an official NVIDIA PyTorch image to ensure CUDA/GPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    libsndfile1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy the vendored A2SB source snapshot from this repository.
COPY nvidia-a2sb-original-repo/ /app/

# 3. Install Python dependencies
#    Versions are pinned so a rebuild months from now produces the same image.
#    The inference stack (torch/lightning) is separate from the web stack, and
#    both must stay on Python 3.10 as shipped by the base image.
#    torch is pinned to the version the base image ships. Without the pin,
#    rotary_embedding_torch >= 0.9 (which requires torch >= 2.4) drags in the
#    newest torch and replaces the CUDA 11.8 build with CUDA 13 wheels, adding
#    several GB and a driver stack this base image cannot use. 0.8.9 is the last
#    release that supports torch 2.x from 2.0 up, and exposes the same
#    RotaryEmbedding/apply_rotary_emb API that networks.py calls.
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    moviepy==1.0.3 \
    "jsonargparse[signatures]==4.35.0" \
    scikit-image==0.25.2 \
    torchlibrosa==0.1.0 \
    pyyaml==6.0.2 \
    numpy==1.26.4 \
    scipy==1.15.3 \
    matplotlib==3.10.0 \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    einops==0.8.1 \
    pytorch_lightning==2.5.0 \
    lightning==2.5.0 \
    rotary_embedding_torch==0.8.9 \
    tqdm==4.67.1 \
    pydub==0.25.1 \
    fastapi==0.121.3 \
    "uvicorn[standard]==0.38.0" \
    python-multipart==0.0.20

RUN pip install --no-cache-dir --no-deps ssr_eval

# 4. Create checkpoints directory
RUN mkdir -p ckpts


# 5. Automate the Config Update
# IMPORTANT: ensemble_2split_sampling expects the two split-domain checkpoints
# (0.0-0.5 and 0.5-1.0). Using the one-split checkpoint here can leave the
# upper split unmodeled, which manifests as zero-filled high-frequency bands.
RUN python3 -c "import yaml; \
    path = 'configs/ensemble_2split_sampling.yaml'; \
    data = yaml.safe_load(open(path)); \
    data['model']['pretrained_checkpoints'] = [ \
        '/app/ckpts/A2SB_twosplit_0.0_0.5_release.ckpt', \
        '/app/ckpts/A2SB_twosplit_0.5_1.0_release.ckpt' \
    ]; \
    trainer = data.setdefault('trainer', {}); \
    trainer['strategy'] = 'auto'; \
    trainer['devices'] = 1; \
    trainer['accelerator'] = 'gpu'; \
    yaml.dump(data, open(path, 'w'), default_flow_style=False, sort_keys=False)"

# 6. Set Environment Variables
ENV CUDA_VISIBLE_DEVICES=0 \
    MKL_THREADING_LAYER=GNU \
    SLURM_NODEID=0 \
    SLURM_PROCID=0 \
    SLURM_LOCALID=0 \
    SLURM_JOB_ID=1 \
    SLURM_NTASKS=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 7. Create a non-root user and setup directories
#    /debug is used by Lightning's CSVLogger as default_root_dir
#    (set via ensembled_inference_api.py checkpoint_callback.dirpath).
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/inputs /app/outputs /app/training_data /debug && \
    chown -R appuser:appuser /app /debug && \
    chmod 1777 /debug

# 8. Setup Entrypoint
# The entrypoint runs as root to fix bind-mount permissions on /app/outputs,
# /app/inputs and /app/training_data, then drops to appuser via runuser.
# update_ckpt_config.py switches to fine-tuned checkpoints if mounted.
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser server/ /app/server/
COPY --chown=appuser:appuser training/ /app/training/
COPY --from=web --chown=appuser:appuser /build/dist/ /app/web/
COPY entrypoint.sh /app/entrypoint.sh
COPY update_ckpt_config.py /app/update_ckpt_config.py
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "app.py"]
