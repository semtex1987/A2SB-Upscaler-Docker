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
RUN pip install --no-cache-dir \
    moviepy==1.0.3 \
    "jsonargparse[signatures]>=4.27.7" \
    scikit-image \
    torchlibrosa \
    pyyaml \
    numpy \
    scipy \
    matplotlib \
    librosa \
    soundfile \
    torchaudio \
    einops \
    pytorch_lightning \
    lightning \
    rotary_embedding_torch \
    tqdm \
    gradio

RUN pip install --no-cache-dir --no-deps ssr_eval

# 4. Create checkpoints directory
RUN mkdir -p ckpts

# 5. Download Checkpoints
RUN wget -O ckpts/A2SB_twosplit_0.5_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt
RUN wget -O ckpts/A2SB_onesplit_0.0_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt
RUN wget -O ckpts/A2SB_twosplit_0.0_0.5_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.0_0.5_release.ckpt

# 6. Automate the Config Update
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

# 7. Set Environment Variables
ENV CUDA_VISIBLE_DEVICES=0 \
    MKL_THREADING_LAYER=GNU \
    SLURM_NODEID=0 \
    SLURM_PROCID=0 \
    SLURM_LOCALID=0 \
    SLURM_JOB_ID=1 \
    SLURM_NTASKS=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 8. Create a non-root user and setup directories
#    /debug is used by Lightning's CSVLogger as default_root_dir
#    (set via ensembled_inference_api.py checkpoint_callback.dirpath).
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/inputs /app/outputs /debug && \
    chown -R appuser:appuser /app /debug && \
    chmod 1777 /debug

# 9. Setup Entrypoint
# The entrypoint runs as root to fix bind-mount permissions on /app/outputs
# and /app/inputs, then drops to appuser via runuser before exec'ing CMD.
# update_ckpt_config.py switches to fine-tuned checkpoints if mounted at /app/ckpts/finetuned.
COPY --chown=appuser:appuser app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
COPY update_ckpt_config.py /app/update_ckpt_config.py
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "app.py"]
