# A2SB Audio Restoration Wrapper
![Docker Automated build](https://img.shields.io/docker/automated/semtex87/a2sb-upscaler)

This repository provides a Dockerized interface for [NVIDIA's Audio-to-Audio Schrödinger Bridges (A2SB)](https://github.com/NVIDIA/diffusion-audio-restoration), a diffusion-based model for audio restoration and bandwidth extension. It wraps the upstream inference code in a FastAPI service and a React web app with stereo support, measured cutoff suggestions, live job progress, and A/B evaluation of the result.

## The interface

The app is organised around the three things you actually do, in order.

**Stage** — Drop files in, or point at a directory already on the pod. Every file is scanned on arrival and labelled with what it really contains: a *lossy transcode* with a brickwall cliff, a *genuine master* that fades naturally, or something already at *full bandwidth* with nothing to restore. The cutoff for each file is pre-filled from that measurement and stays editable, so the common case needs no decision and the unusual one is still under your control.

**Run** — Jobs go into a single-slot queue, because there is one GPU. Progress comes from the inference process itself: the real diffusion step count, a per-channel stage, and an ETA. The log streams live and a run can be cancelled, which stops the inference process rather than orphaning it. Job history survives a container restart; anything that was mid-flight comes back marked *interrupted* rather than silently forgotten.

**Evaluate** — The headline number is how much energy the model actually added above the cutoff. Under it, an A/B transport plays the filtered input and the restored output from the same playhead, so switching never moves your position, and a *solo* control high-passes both so you hear only the reconstructed band. That is the honest test: if the band is silent or gritty, the model did not give you anything worth keeping. An interactive spectrogram wipes between input and output with a draggable handle and reads out time, frequency, and level wherever you hover.

**Train** — Scan and vet a directory of high-quality audio directly in the UI, configure the fine-tune (steps, batch size, learning rate, which splits to train), and submit a training job into the same single-slot GPU queue. Progress streams live with per-split step counts. When the run finishes, click **Activate fine-tuned** in the Checkpoints panel — the ensemble config is rewritten without restarting the container, and the next restoration uses your weights. Click **Revert to release** any time to go back.

## Features

- **Measured cutoff suggestions**: Each file's spectrum is scanned for a brickwall cliff. When one is found, the suggested cutoff sits just under it so the model regenerates from the last band that still has real signal. The same scan powers `training/vet_dataset.py`, so the UI and the dataset vetting tool always agree.
- **Real progress and cancellation**: The inference subprocess is streamed rather than buffered, so the progress bar tracks actual diffusion steps. Cancelling signals the whole process group, including the Lightning child process.
- **Stereo support**: Splits left/right channels, runs A2SB per channel, recombines to stereo.
- **Numeric cutoff input**: Set the lowpass/restoration cutoff in **Hz** (1000–20000). Set it at or below where your source's content actually degrades — MP3 artifacts typically start between 10–16 kHz depending on bitrate. The value is passed directly to the model so the correct frequency mask is applied.
- **High-band energy readout**: After each restoration the app reports the RMS energy level (dBFS) above the cutoff for both the filtered input and the restored output, e.g. `energy ≥14000 Hz: −42.3 dB → −28.7 dB (+13.6 dB)`. When the model adds less than 1 dB, the result says so instead of presenting it as a success.
- **Linear-frequency spectrogram**: The comparison uses a linear-frequency STFT so the 14–22 kHz band occupies proportional vertical space. A mel scale compresses that entire band into a few pixels, making restoration visually invisible even when it worked.
- **16-bit PCM output**: Restored WAVs are written as 16-bit PCM, matching the bit depth of typical CD/MP3 sources.
- **Sample rate**: Input is normalized to **44.1 kHz** to match the model config and avoid extra resampling.
- **Output**: Restored WAVs are written per run under a bind-mounted directory (`restored_audio/` by default) with permissions fixed at container startup. Re-running with different settings no longer overwrites an earlier result.
- **Fine-tuned checkpoints**: If you fine-tune the model (see below), you can mount `training_output/checkpoints/` so inference uses your checkpoints instead of the release ones.
- **GPU**: Uses a single NVIDIA GPU via Docker's GPU support.

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

4. **Restore audio**: Drop files on the **Stage** view, check the suggested cutoff for each one, and start the run. Watch it on **Run**, then compare input against output on **Evaluate**. Restored audio is saved under `restored_audio/runs/<job-id>/`.

## Development

The backend and frontend can run separately with hot reload. The API listens on 7860 and Vite proxies to it, so open the Vite URL rather than 7860.

```bash
# Terminal 1: the API, pointed at a local staging tree.
A2SB_INPUT_DIR=./dev/inputs A2SB_OUTPUT_DIR=./dev/outputs python3 app.py

# Terminal 2: the frontend.
cd web && npm install && npm run dev
```

Run the test suite from the repo root:

```bash
python3 -m pytest tests/ -q
```

The tests stub the diffusion step, so no GPU or checkpoint is needed.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `A2SB_INPUT_DIR` | `/app/inputs` | Upload and staging tree. Paths outside it are refused. |
| `A2SB_OUTPUT_DIR` | `/app/outputs` | Restored audio and job history. |
| `A2SB_PORT` | `7860` | Listen port. |
| `A2SB_INFERENCE_CWD` | `/app/inference` | Where `A2SB_upsample_api.py` lives. |
| `A2SB_WEB_DIR` | `/app/web` | Built frontend. Falls back to `web/dist` for local runs. |
| `A2SB_DEFAULT_BATCH_SIZE` | `16` | Starting batch size; lower it if inference hits CUDA OOM. |

## Sample Docker Compose
```yaml
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

    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=5)"]
      # The entrypoint downloads the checkpoints before the server binds.
      start_period: 20m
      interval: 30s
      timeout: 10s
      retries: 3

    # Let a running job be cancelled cleanly before Docker sends SIGKILL.
    stop_grace_period: 30s

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

- **Restored files**: The `restored_audio/` directory (bind-mounted to `/app/outputs` in the container) receives all restored WAVs, one directory per run under `runs/<job-id>/`, alongside the job metadata that makes history survive a restart. The entrypoint fixes ownership so the app user can write there.
- **Optional fine-tuned checkpoints**: If you have run fine-tuning, mount the checkpoint directory so inference uses your weights:
  ```yaml
  volumes:
    - ./restored_audio:/app/outputs
    - ./training_output/checkpoints:/app/ckpts/finetuned:ro
  ```
  At startup, the container updates the ensemble config to use the two finetuned checkpoints when both are present under `/app/ckpts/finetuned/`.

## Fine-tuning the model

You can fine-tune the two A2SB split checkpoints on your own high-quality, full-bandwidth audio to improve restoration beyond the release checkpoints' ~12 kHz ceiling.

### Option A — via the Train tab (recommended)

The **Train** tab runs fine-tuning inside the same container as inference, so no second container or separate compose file is needed.

1. **Mount training data**: Add your high-quality audio to `./training_data/` (bind-mounted to `/app/training_data`). WAV and FLAC work best; 44.1 kHz is ideal.

2. **Scan & vet**: On the Train tab, enter the directory path and click **Scan & vet**. Each file is spectral-scanned with the same algorithm as `training/vet_dataset.py`:
   - **PASS** — real energy to ≥20.5 kHz
   - **CHECK** — 17–20.5 kHz; may be a dark master or a 320 kbps transcode; listen before including
   - **REJECT** — below 17 kHz; training on these teaches the model to output silence in the high band

3. **Configure and start**: Set steps, batch size, learning rate, and which splits to train (both by default), then click **Start fine-tuning**. A preflight check verifies disk space (~25 GB needed), release checkpoints, and framework availability. Training queues behind any active restoration job, and the header GPU badge reads "Training" so you know not to wait on restorations.

4. **Activate**: When training completes, click **Activate fine-tuned** in the Checkpoints panel. The ensemble config is rewritten in place — no container restart. The next restoration uses your weights. Click **Revert to release** to go back at any time.

### Option B — via the CLI / separate training container

For long runs, automated pipelines, or when the main container is busy, use the separate training container:

```bash
# Vet your dataset
docker compose -f training/docker-compose.train.yml run trainer \
    python /app/training/vet_dataset.py /data/training_data --csv /data/vet_report.csv

# Fine-tune
docker compose -f training/docker-compose.train.yml run trainer \
    python /app/training/finetune.py --steps 5000
```

Checkpoints land in `training_output/checkpoints/`. Use the Train tab's **Activate** button to point the running inference container at them, or restart with a volume mount:

```yaml
volumes:
  - ./restored_audio:/app/outputs
  - ./training_output/checkpoints:/app/ckpts/finetuned:ro
```

For more CLI options (`--splits`, `--batch-size`, `--learning-rate`, data quality guidance) see **[training/README.md](training/README.md)**.
## RunPod and other cloud GPU pods

Pre-configured template https://console.runpod.io/deploy?template=rtpczbd3cl&ref=685x2sbd

When you run this app on RunPod (or similar), use the image **as built from this repo** and configure the Pod so the app can write outputs and the inference subprocess can run.

### Use this repo’s image and entrypoint

1. **Use this project’s image as the template Container Image**  
   Build from this repo’s `Dockerfile`, push to Docker Hub (or your registry), and set that image as the **Container Image** in your RunPod template. Do **not** use RunPod’s “application only” pattern that clears the entrypoint (e.g. `ENTRYPOINT []` and `CMD ["python", "/app/app.py"]`). This image’s **ENTRYPOINT** runs a script that fixes permissions for `/app/inputs`, `/app/outputs`, and `/debug` before starting the app; if you override it, you can get “A2SB inference produced no output file” or permission errors.

2. **Expose the web port**  
   In the template's **HTTP Ports**, add port **7860**. Use that URL to open the UI once the Pod is running. The app streams job progress over server-sent events; RunPod's proxy passes these through, and the server sends keepalives so idle streams are not closed.

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
- **Restored audio sounds wrong or only up to ~12 kHz**: The release checkpoints were trained on data with limited high-frequency content. Use the fine-tuning pipeline with full-bandwidth material and the optional checkpoint mount to improve high-end extension. Set the cutoff at or below where your source content actually degrades, then use **Solo** on the Evaluate view to listen to the reconstructed band on its own.
- **The UI loads but says the frontend bundle is missing**: The image builds the frontend in a separate stage. If you are running outside Docker, run `npm ci && npm run build` in `web/`, or set `A2SB_WEB_DIR` to a directory containing `index.html`.
- **“No output file” / inference fails**: Usually the inference subprocess failed earlier (e.g. Permission denied on `/debug` or `/app/outputs`). Check the container logs for the Python traceback just above this message. Mount a writable volume to `/app/outputs` and ensure the image entrypoint is used.
- **vGPU / “Operation not supported”**: Prefer PCIe passthrough for the GPU if possible; otherwise ensure Docker and the NVIDIA stack are configured for your vGPU environment.
- **Port in use**: Change the host port in `docker-compose.yml` (e.g. `"8080:7860"`).

## Credits

- **Upstream**: [NVIDIA diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration) and the paper [Audio-to-Audio Schrödinger Bridges](https://arxiv.org/abs/2305.15083).
- This wrapper adds the FastAPI service and React interface, the job queue with live progress and cancellation, measured cutoff suggestions, A/B evaluation with high-band solo, stereo handling, 16-bit PCM output, high-band energy readout, linear-frequency STFT comparison, 44.1 kHz normalization, bind-mount permission handling, optional fine-tuned checkpoint loading, training automation under `training/`, and the dataset vetting utility.
