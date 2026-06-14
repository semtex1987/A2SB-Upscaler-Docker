# A2SB Audio Restoration Wrapper

This repository provides a Dockerized interface for [NVIDIA's Audio-to-Audio Schrödinger Bridges (A2SB)](https://github.com/NVIDIA/diffusion-audio-restoration), a diffusion-based model for audio restoration and bandwidth extension. It wraps the upstream inference code in a [Gradio](https://gradio.app/) web UI with stereo support, configurable low-pass simulation, and optional fine-tuned checkpoints.

## Features

- **Web interface**: Gradio UI for uploading audio and viewing before/after spectrograms.
- **Stereo support**: Splits left/right channels, runs A2SB per channel, recombines to stereo.
- **Numeric cutoff input**: Set the lowpass/restoration cutoff in **Hz** (1000–20000, default 14000). Set it at or below where your source’s content actually degrades — MP3 artifacts typically start between 10–16 kHz depending on bitrate. The value is passed directly to the model so the correct frequency mask is applied.
- **High-band energy readout**: After each restoration, the results summary reports the RMS energy level (dBFS) above the cutoff for both the filtered input and the restored output, e.g. `energy ≥14000 Hz: filtered input −42.3 dB → restored −28.7 dB (+13.6 dB)`. This gives a quantitative measure of how much content A2SB added in the target band.
- **Linear-frequency spectrogram**: The before/after comparison plot uses a linear-frequency STFT so the 14–22 kHz band occupies proportional vertical space. The previous mel-128 plot compressed that entire band into just a few pixels, making restoration visually invisible even when it worked.
- **16-bit PCM output**: Restored WAVs are written as 16-bit PCM, matching the bit depth of typical CD/MP3 sources. The previous behaviour wrote 32-bit float WAV (the default scipy encoding for float arrays), producing output files roughly twice the input size with no quality benefit.
- **Sample rate**: Input is normalized to **44.1 kHz** to match the model config and avoid extra resampling.
- **Output**: Restored WAVs and comparison spectrograms are written to a bind-mounted directory (`restored_audio/` by default) with permissions fixed at container startup.
- **Fine-tuned checkpoints**: If you fine-tune the model (see below), you can mount `training_output/checkpoints/` so inference uses your checkpoints instead of the release ones.
- **GPU**: Uses a single NVIDIA GPU via Docker’s GPU support.

## Prerequisites

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for CUDA in containers

## Installation and usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/semtex1987/A2SB-Upscaler-Docker
   cd A2SB-Upscaler-Docker
   ```

2. **Build and start the inference service**:
   ```bash
   docker compose up --build -d
   ```
   The first run can take several minutes while the image is built and NVIDIA checkpoints are downloaded.

3. **Open the UI**:  
   http://localhost:7860

4. **Restore audio**: Upload a file, set the **Lowpass / Restoration Cutoff (Hz)** to match where your source's content degrades (e.g. 14000 for a typical 128–192 kbps MP3), adjust steps if desired, and run. Restored audio and spectrograms are saved under `restored_audio/`. The results summary reports high-band energy before and after restoration so you can confirm the model added content above the cutoff.

## Sample Docker Compose
```
services:
  upsampler:
    build: .
    image: semtex87/a2sb-upscaler
    container_name: nvidia_upsample
    
    ports:
      - "7860:7860"
    
    volumes:
      - ./restored_audio:/app/outputs
      - ./training_output/checkpoints:/app/ckpts/finetuned:ro
    
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

    stdin_open: true
    tty: true
```

## Output and volumes

- **Restored files**: The `restored_audio/` directory (bind-mounted to `/app/outputs` in the container) receives all restored WAVs and spectrogram images. The entrypoint fixes ownership so the app user can write there.
- **Optional fine-tuned checkpoints**: If you have run fine-tuning, mount the checkpoint directory so inference uses your weights:
  ```yaml
  volumes:
    - ./restored_audio:/app/outputs
    - ./training_output/checkpoints:/app/ckpts/finetuned:ro
  ```
  At startup, the container updates the ensemble config to use the two finetuned checkpoints when both are present under `/app/ckpts/finetuned/`.

## Fine-tuning the model

You can fine-tune the two A2SB split checkpoints on your own high-quality, full-bandwidth audio to improve restoration beyond the release checkpoints' ~12 kHz ceiling. The automation uses a **separate training container** and a **manual run** workflow.

### 0. Vet your dataset first

Before spending GPU time on training, check that your audio is genuinely full-bandwidth. Lossy files wearing a `.flac` extension (MP3-derived, re-encoded, etc.) teach the model to output silence in the high band. The vetting tool runs a spectral scan and flags transcodes:

```bash
docker compose -f training/docker-compose.train.yml run trainer \
    python /app/training/vet_dataset.py /data/training_data --csv /data/vet_report.csv
```

Each file is rated **PASS** (real energy to ≥20.5 kHz), **CHECK** (17–20.5 kHz, examine the spectrogram), or **REJECT** (<17 kHz, discard). A `shelf` column flags brickwall lowpass signatures (transcode tells). See `training/vet_dataset.py` at the top of the file for a full column reference.

### 1. Put audio in the training directory

Place full-bandwidth 44.1 kHz WAV/FLAC (or other supported formats) in `training_data/`. Files shorter than ~3 seconds are skipped. The manifest builder also estimates each file's bandwidth via spectral rolloff and excludes files with an estimated cutoff below 16 kHz — no action needed, but you'll see a warning for each excluded file.

### 2. Run fine-tuning

From the repo root:
```bash
docker compose -f training/docker-compose.train.yml run trainer \
  python /app/training/finetune.py --steps 5000
```
This generates a manifest from `training_data/`, fine-tunes both time-splits (0.0–0.5 and 0.5–1.0), and writes checkpoints to `training_output/checkpoints/`. `--steps` is the number of **new** steps to train — the pipeline reads the release checkpoint's current `global_step` and offsets `max_steps` accordingly, so the requested steps are always trained regardless of the checkpoint's starting step count.

### 3. Use the new checkpoints

Ensure the inference `docker-compose.yml` mounts `./training_output/checkpoints:/app/ckpts/finetuned:ro` (see “Output and volumes” above), then restart:
```bash
docker compose down && docker compose up -d
```

For more options (`--splits`, `--batch-size`, `--learning-rate`, data quality guidance, recommended genres) see **[training/README.md](training/README.md)**.

## RunPod and other cloud GPU pods

Pre-configured template https://console.runpod.io/deploy?template=rtpczbd3cl&ref=685x2sbd

When you run this app on RunPod (or similar), use the image **as built from this repo** and configure the Pod so the app can write outputs and the inference subprocess can run.

### Use this repo’s image and entrypoint

1. **Use this project’s image as the template Container Image**  
   Build from this repo’s `Dockerfile`, push to Docker Hub (or your registry), and set that image as the **Container Image** in your RunPod template. Do **not** use RunPod’s “application only” pattern that clears the entrypoint (e.g. `ENTRYPOINT []` and `CMD ["python", "/app/app.py"]`). This image’s **ENTRYPOINT** runs a script that fixes permissions for `/app/inputs`, `/app/outputs`, and `/debug` before starting the app; if you override it, you can get “A2SB inference produced no output file” or permission errors.

2. **Expose the Gradio port**  
   In the template’s **HTTP Ports**, add a port with label e.g. **Gradio** and port **7860**. Use that URL to open the UI once the Pod is running.

3. **Mount a volume to `/app/outputs`**  
   The app and the inference subprocess write restored WAVs to `/app/outputs`. In RunPod, add a **Volume** to the Pod and mount it at **Container path** `/app/outputs`. If this directory isn’t writable (e.g. no volume or wrong path), inference can complete without writing a file and you’ll see “A2SB inference produced no output file.”

4. **Container disk**  
   The image includes large checkpoints; set **Container Disk** to at least **20 GB** (or more if you add fine-tuned checkpoints).

5. **Optional**  
   Mount a volume at `/app/inputs` if you want uploaded files to persist across restarts.

### If you extend a RunPod base image instead

If you build a custom template by extending a RunPod base image (e.g. `runpod/pytorch:...`) and copy this app into it, keep this app’s entrypoint so permissions are fixed. For example, end your Dockerfile with:

```dockerfile
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "/app/app.py"]
```

Do **not** use `ENTRYPOINT []` and `CMD ["python", "/app/app.py"]` only, or the inference subprocess may fail on `/debug` or `/app/outputs`.

### Local Docker run (for comparison)

```bash
docker run -it --gpus all -p 7860:7860 \
  -v /path/on/host/outputs:/app/outputs \
  -v /path/on/host/inputs:/app/inputs \
  your-image-name
```

## Troubleshooting

- **Permission denied on outputs or `/debug`**: The image entrypoint runs as root, fixes ownership on `/app/inputs`, `/app/outputs`, and `/debug`, then drops to the app user. Rebuild the image so the updated entrypoint and `/debug` creation are included.
- **Restored audio sounds wrong or only up to ~12 kHz**: The release checkpoints were trained on data with limited high-frequency content. Use the fine-tuning pipeline with full-bandwidth material and the optional checkpoint mount to improve high-end extension. Set the cutoff (in Hz) at or below where your source content actually degrades — the high-band energy readout in the results summary shows whether the model added anything above that cutoff.
- **“No output file” / inference fails**: Usually the inference subprocess failed earlier (e.g. Permission denied on `/debug` or `/app/outputs`). Check the container logs for the Python traceback just above this message. Mount a writable volume to `/app/outputs` and ensure the image entrypoint is used.
- **vGPU / “Operation not supported”**: Prefer PCIe passthrough for the GPU if possible; otherwise ensure Docker and the NVIDIA stack are configured for your vGPU environment.
- **Port in use**: Change the host port in `docker-compose.yml` (e.g. `"8080:7860"`).

## Credits

- **Upstream**: [NVIDIA diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration) and the paper [Audio-to-Audio Schrödinger Bridges](https://arxiv.org/abs/2305.15083).
- This wrapper adds the Gradio app, stereo handling, numeric Hz cutoff input, 16-bit PCM output, high-band energy readout, linear-frequency STFT comparison plot, 44.1 kHz normalization, bind-mount permission handling, optional fine-tuned checkpoint loading, training automation under `training/`, and the dataset vetting utility.
