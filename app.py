import gradio as gr
import subprocess
import os
import tempfile
import glob
import numpy as np
import matplotlib
# Use 'Agg' backend to prevent errors in Docker (no display)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, sosfilt
from pydub import AudioSegment

# Directories
def _ensure_runtime_dir(preferred_path, fallback_name):
    try:
        os.makedirs(preferred_path, exist_ok=True)
        return preferred_path
    except OSError:
        fallback_path = os.path.join(tempfile.gettempdir(), fallback_name)
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path


INPUT_DIR = _ensure_runtime_dir(os.environ.get("A2SB_INPUT_DIR", "/app/inputs"), "a2sb-inputs")
OUTPUT_DIR = _ensure_runtime_dir(os.environ.get("A2SB_OUTPUT_DIR", "/app/outputs"), "a2sb-outputs")


def _read_int_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


UI_BATCH_MIN = 1
UI_BATCH_MAX = max(UI_BATCH_MIN, _read_int_env("A2SB_UI_BATCH_MAX", 64))
UI_BATCH_DEFAULT = _read_int_env("A2SB_DEFAULT_BATCH_SIZE", 16)
UI_BATCH_DEFAULT = min(max(UI_BATCH_DEFAULT, UI_BATCH_MIN), UI_BATCH_MAX)

# --- Signal Processing Functions ---

def butter_lowpass_filter(data, cutoff, fs, order=10):
    data_arr = np.asarray(data)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        return data_arr

    # pydub provides PCM integer samples (typically int16). Filtering directly on
    # integers and casting back without clipping can wrap overflow and sound
    # severely distorted. Process in float domain, then clip before reconversion.
    int_dtype = np.issubdtype(data_arr.dtype, np.integer)
    if int_dtype:
        type_info = np.iinfo(data_arr.dtype)
        peak = float(max(abs(type_info.min), type_info.max))
        data_float = data_arr.astype(np.float32) / peak
    else:
        data_float = data_arr.astype(np.float32, copy=False)

    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    filtered = sosfilt(sos, data_float, axis=0)

    if not int_dtype:
        return filtered

    # Keep int range stable and avoid wrap-around artifacts.
    filtered = np.clip(filtered, -1.0, 1.0 - (1.0 / peak))
    return np.round(filtered * peak).astype(data_arr.dtype)

def apply_lowpass_to_segment(segment, cutoff_freq_hz):
    channel_data = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        channel_data = channel_data.reshape((-1, 2))
    fs = segment.frame_rate
    filtered_data = butter_lowpass_filter(channel_data, cutoff_freq_hz, fs)
    return segment._spawn(filtered_data.tobytes())

# --- Plotting Function (FIXED LAYOUT) ---

def generate_comparison_plot(original_path, restored_path):
    """
    Generates a side-by-side Mel-spectrogram comparison.
    """
    # Load audio files
    y_orig, sr_orig = librosa.load(original_path)
    y_rest, sr_rest = librosa.load(restored_path)

    # Compute Mel Spectrograms
    # Use the restored sample rate as the max freq reference for both plots so they match visually
    fmax = sr_rest / 2

    S_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr_orig, n_mels=128, fmax=fmax)
    S_db_orig = librosa.power_to_db(S_orig, ref=np.max)

    S_rest = librosa.feature.melspectrogram(y=y_rest, sr=sr_rest, n_mels=128, fmax=fmax)
    S_db_rest = librosa.power_to_db(S_rest, ref=np.max)

    # FIX: Use constrained_layout=True to handle colorbars automatically
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12, 8), constrained_layout=True)
    
    # Plot Original
    img1 = librosa.display.specshow(S_db_orig, x_axis='time', y_axis='mel', sr=sr_orig, fmax=fmax, ax=ax[0], cmap='inferno')
    ax[0].set_title('Original (Filtered Input)')
    ax[0].set(xlabel='') # Hide x label for top plot

    # Plot Restored
    img2 = librosa.display.specshow(S_db_rest, x_axis='time', y_axis='mel', sr=sr_rest, fmax=fmax, ax=ax[1], cmap='inferno')
    ax[1].set_title('Restored Output (A2SB)')

    # Add Colorbar
    # Attaching it to 'ax' makes it span both plots nicely on the right
    fig.colorbar(img2, ax=ax, format='%+2.0f dB', label='Intensity (dB)')
    
    # Save Plot
    output_img_path = restored_path.replace(".wav", "_spectrogram.png")
    
    # FIX: Removed plt.tight_layout() as it conflicts with constrained_layout
    plt.savefig(output_img_path)
    plt.close()
    
    return output_img_path

# --- Inference Functions ---

def run_a2sb_inference(input_path, output_path, steps, cutoff_hz, batch_size):
    script_name = "A2SB_upsample_api.py"

    # UpsampleMask computes FFT bin indices via (n_fft * freq / sampling_rate),
    # so the cutoff MUST be in Hz to produce the correct bin boundary.
    command = [
        "python3", script_name,
        "-f", input_path,
        "-o", output_path,
        "-n", str(int(steps)),
        "-c", str(int(cutoff_hz)),
        "-b", str(int(batch_size)),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    print(
        f"[A2SB] command: {' '.join(command)} "
        f"(steps={int(steps)}, cutoff_hz={int(cutoff_hz)}, batch_size={int(batch_size)})",
        flush=True,
    )
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/app/inference",
        env=env,
    )

    if result.stdout:
        print("[A2SB] stdout (tail):", "\n".join(result.stdout.splitlines()[-20:]))
    if result.stderr:
        print("[A2SB] stderr (tail):", "\n".join(result.stderr.splitlines()[-20:]))

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"A2SB inference produced no output file at {output_path}. "
            "Check stdout/stderr above for details."
        )

    if is_likely_corrupted_audio(output_path):
        raise RuntimeError("A2SB output failed audio validation")

    return result


def is_likely_corrupted_audio(path):
    try:
        segment = AudioSegment.from_file(path)
        samples = np.array(segment.get_array_of_samples())
    except Exception:
        return True

    if samples.size == 0 or not np.isfinite(samples).all():
        return True

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        full_scale = float(info.max)
    else:
        full_scale = 1.0

    abs_samples = np.abs(samples.astype(np.float64))
    peak = float(np.max(abs_samples))
    if peak <= 0.0:
        return True

    rms = float(np.sqrt(np.mean(np.square(abs_samples))))
    if rms < full_scale * 1e-3:
        return True

    clipped_ratio = float(np.mean(abs_samples >= (full_scale * 0.995)))
    if clipped_ratio > 0.25:
        return True

    # Spectral flatness near 1.0 indicates noise-like content, which is a
    # hallmark of failed diffusion outputs (e.g. when the entire spectrum was
    # masked and the model hallucinated from noise).
    try:
        y = samples.astype(np.float32) / max(peak, 1.0)
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = float(np.mean(flatness))
        if mean_flatness > 0.6:
            return True
    except Exception:
        pass

    return False

def ensure_a2sb_input_format(segment):
    """
    The A2SB model configs (ensemble_2split_sampling.yaml, inference_files_upsampling.yaml)
    specify sampling_rate=44100.  The data loader resamples to that rate on load, so
    feeding 48kHz here only adds a needless lossy resample round-trip.
    """
    return segment.set_frame_rate(44100).set_sample_width(2)

# --- Main Logic with Progress ---

def normalize_input_files(input_files):
    if not input_files:
        return []
    if isinstance(input_files, str):
        return [input_files]
    return list(input_files)


def normalize_staged_paths(staged_paths_text):
    if not staged_paths_text:
        return []

    files = []
    lines = [line.strip() for line in staged_paths_text.splitlines() if line.strip()]
    for line in lines:
        matches = sorted(glob.glob(line))
        if matches:
            files.extend(path for path in matches if os.path.isfile(path))
        elif os.path.isfile(line):
            files.append(line)

    # De-duplicate while preserving order.
    deduped = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def merge_input_sources(uploaded_files, staged_paths_text):
    merged = []
    seen = set()
    for path in normalize_input_files(uploaded_files) + normalize_staged_paths(staged_paths_text):
        if path in seen:
            continue
        seen.add(path)
        merged.append(path)
    return merged


def list_staged_files(staged_paths_text):
    if not staged_paths_text or not staged_paths_text.strip():
        return "No staged path pattern provided yet."

    files = normalize_staged_paths(staged_paths_text)
    if not files:
        return "No staged files matched the provided path(s)."

    lines = [f"Matched {len(files)} staged file(s):"]
    for path in files:
        lines.append(f"- `{path}`")
    return "\n".join(lines)


def restore_one_audio(input_file, steps, cutoff_hz, batch_size, progress, file_index, total_files):
    try:
        audio = AudioSegment.from_file(input_file)
        audio = ensure_a2sb_input_format(audio)
    except Exception as e:
        raise gr.Error(f"Failed to load audio: {e}")

    base_name = os.path.splitext(os.path.basename(input_file))[0].replace(" ", "_")
    final_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_restored.wav")
    comparison_input_path = os.path.join(INPUT_DIR, f"{base_name}_filtered_input.wav")

    file_start = file_index / total_files
    file_end = (file_index + 1) / total_files

    def update_file_progress(relative_value, desc):
        absolute_value = file_start + ((file_end - file_start) * relative_value)
        progress(min(max(absolute_value, 0.0), 1.0), desc=desc)

    def process_channel(segment, channel_id, display_name, start_prog, end_prog):
        prog_range = end_prog - start_prog
        update_file_progress(
            start_prog,
            f"[{file_index + 1}/{total_files}] [{display_name}] Applying {cutoff_hz}Hz Filter...",
        )
        filtered = apply_lowpass_to_segment(segment, cutoff_hz)

        temp_in = os.path.join(INPUT_DIR, f"{base_name}_{channel_id}_input.wav")
        temp_out = os.path.join(OUTPUT_DIR, f"{base_name}_{channel_id}_restored.wav")

        if os.path.exists(temp_out):
            os.remove(temp_out)

        filtered.export(temp_in, format="wav")

        restore_point = start_prog + (prog_range * 0.2)
        update_file_progress(
            restore_point,
            f"[{file_index + 1}/{total_files}] [{display_name}] Running A2SB Inference...",
        )

        run_a2sb_inference(temp_in, temp_out, steps, cutoff_hz, batch_size)

        if not os.path.exists(temp_out):
            raise Exception(f"Inference script failed to generate {temp_out}")

        return temp_out, filtered

    final_filtered_audio = None

    if audio.channels == 1:
        restored_path, filtered_seg = process_channel(audio, "mono", "Mono Channel", 0.1, 0.9)
        final_filtered_audio = filtered_seg
        AudioSegment.from_file(restored_path).export(final_output_path, format="wav")

    elif audio.channels == 2:
        update_file_progress(0.1, f"[{file_index + 1}/{total_files}] Splitting Stereo Channels...")
        channels = audio.split_to_mono()

        out_l_path, filtered_l = process_channel(channels[0], "left", "Left Channel", 0.15, 0.5)
        out_r_path, filtered_r = process_channel(channels[1], "right", "Right Channel", 0.5, 0.85)

        update_file_progress(0.85, f"[{file_index + 1}/{total_files}] Recombining Stereo Channels...")
        restored_l = AudioSegment.from_file(out_l_path)
        restored_r = AudioSegment.from_file(out_r_path)

        restored_stereo = AudioSegment.from_mono_audiosegments(restored_l, restored_r)
        restored_stereo.export(final_output_path, format="wav")

        final_filtered_audio = AudioSegment.from_mono_audiosegments(filtered_l, filtered_r)

    else:
        raise gr.Error(f"Unsupported channels: {audio.channels}")

    update_file_progress(0.9, f"[{file_index + 1}/{total_files}] Generating Spectral Analysis...")
    final_filtered_audio.export(comparison_input_path, format="wav")
    plot_path = generate_comparison_plot(comparison_input_path, final_output_path)
    update_file_progress(1.0, f"[{file_index + 1}/{total_files}] Done!")
    return final_output_path, plot_path


def restore_audio(input_files, steps, cutoff_choice, batch_size, progress=gr.Progress()):
    files = normalize_input_files(input_files)
    if not files:
        raise gr.Error(
            "No input files found. Upload file(s), or use staged paths like /app/inputs/*.wav."
        )

    cutoff_hz = int(cutoff_choice.lower().replace("khz", "")) * 1000
    restored_outputs = []
    plot_outputs = []

    try:
        for idx, input_file in enumerate(files):
            progress(idx / len(files), desc=f"[{idx + 1}/{len(files)}] Initializing & Loading Audio...")
            restored_path, plot_path = restore_one_audio(
                input_file,
                steps,
                cutoff_hz,
                batch_size,
                progress,
                idx,
                len(files),
            )
            restored_outputs.append(restored_path)
            plot_outputs.append(plot_path)

        progress(1.0, desc=f"Finished processing {len(files)} file(s)")
        return restored_outputs, plot_outputs

    except subprocess.CalledProcessError as e:
        print("STDERR:", e.stderr)
        raise gr.Error(f"Restoration failed: {e.stderr}")
    except Exception as e:
        print("Error:", str(e))
        raise gr.Error(f"Processing error: {str(e)}")


def summarize_results(restored_outputs):
    if not restored_outputs:
        return "No files processed yet."
    lines = [f"Processed {len(restored_outputs)} file(s):"]
    for path in restored_outputs:
        lines.append(f"- {os.path.basename(path)}")
    return "\n".join(lines)


def build_preview_choices(restored_outputs):
    return [os.path.basename(path) for path in restored_outputs]


def select_preview(selection, restored_outputs, plot_outputs):
    if not selection or not restored_outputs:
        return None, None

    choices = build_preview_choices(restored_outputs)
    try:
        selected_index = choices.index(selection)
    except ValueError:
        selected_index = 0

    return restored_outputs[selected_index], plot_outputs[selected_index]


def process_batch(input_files, staged_paths, steps, cutoff_choice, batch_size, progress=gr.Progress()):
    files = merge_input_sources(input_files, staged_paths)
    if not files:
        raise gr.Error(
            "No input files found. Upload file(s), or use staged paths like /app/inputs/*.wav."
        )
    print(
        f"[A2SB] batch_start files={len(files)} steps={int(steps)} cutoff={cutoff_choice} "
        f"batch_size={int(batch_size)}",
        flush=True,
    )
    restored_outputs, plot_outputs = restore_audio(
        files,
        steps,
        cutoff_choice,
        batch_size,
        progress=progress,
    )

    preview_choices = build_preview_choices(restored_outputs)
    initial_selection = preview_choices[0] if preview_choices else None
    preview_audio, preview_plot = select_preview(initial_selection, restored_outputs, plot_outputs)

    return (
        restored_outputs,
        plot_outputs,
        summarize_results(restored_outputs),
        gr.update(choices=preview_choices, value=initial_selection),
        restored_outputs,
        plot_outputs,
        preview_audio,
        preview_plot,
    )

# --- Interface ---

if __name__ == "__main__":
    custom_css = "body { background-color: #121212; color: white; }"
    with gr.Blocks() as demo:
        gr.Markdown("# NVIDIA A2SB Stereo Restorer")
        gr.Markdown(
            "Upload one or many audio files to restore them sequentially. "
            "Lower batch size reduces VRAM usage at the cost of longer inference time. "
            "For H100/H200, 16-32 is a good starting range. For bandwidth extension, "
            "50-100 steps is usually the practical range."
        )
        gr.Markdown(
            "Optional: stage files directly on the pod (for example with runpodctl) and "
            "enter paths like /app/inputs/*.wav below to bypass browser upload."
        )

        restored_state = gr.State([])
        plot_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                input_files = gr.File(
                    file_count="multiple",
                    type="filepath",
                    file_types=["audio"],
                    label="Upload Audio File(s)",
                )
                staged_paths = gr.Textbox(
                    label="Staged File Paths (Optional)",
                    placeholder="/app/inputs/*.wav\n/app/inputs/song_a.flac",
                    lines=2,
                    info="Use glob patterns like /app/inputs/*.wav to process files already on the server.",
                )
                list_staged_button = gr.Button("List Staged Files")
                staged_preview = gr.Markdown("No staged files listed yet.")
                steps = gr.Slider(
                    minimum=10, maximum=200, value=50, step=10, label="Steps (Quality)",
                    info="More steps = higher quality but slower processing. 50-100 is recommended.",
                )
                cutoff_choice = gr.Dropdown(
                    choices=["4kHz", "14kHz", "16kHz"],
                    value="14kHz",
                    label="Input Lowpass Filter (Cutoff)",
                    info="Set to the approximate highest frequency in your original audio (e.g., 4kHz for phone calls).",
                )
                batch_size = gr.Slider(
                    minimum=UI_BATCH_MIN,
                    maximum=UI_BATCH_MAX,
                    value=UI_BATCH_DEFAULT,
                    step=1,
                    label="Inference Batch Size",
                    info="Higher = faster but uses more VRAM. Start with 16-32 for H100/H200.",
                )
                run_button = gr.Button("Process Batch", variant="primary")
                summary = gr.Markdown("No files processed yet.")
                download_files = gr.Files(label="Download Restored Result(s)")

            with gr.Column(scale=1):
                preview_choice = gr.Dropdown(choices=[], label="Preview Restored File", interactive=True)
                preview_audio = gr.Audio(label="Restored Preview")
                preview_plot = gr.Image(label="Spectral Analysis")
                gallery = gr.Gallery(label="All Spectrograms", columns=2, height="auto")

        run_button.click(
            fn=process_batch,
            inputs=[input_files, staged_paths, steps, cutoff_choice, batch_size],
            outputs=[
                download_files,
                gallery,
                summary,
                preview_choice,
                restored_state,
                plot_state,
                preview_audio,
                preview_plot,
            ],
        )

        list_staged_button.click(
            fn=list_staged_files,
            inputs=[staged_paths],
            outputs=[staged_preview],
        )

        preview_choice.change(
            fn=select_preview,
            inputs=[preview_choice, restored_state, plot_state],
            outputs=[preview_audio, preview_plot],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(),
        css=custom_css,
    )
