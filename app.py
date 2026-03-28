import gradio as gr
import subprocess
import os
import tempfile
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

def run_a2sb_inference(input_path, output_path, steps, cutoff_hz):
    script_name = "A2SB_upsample_api.py"

    # UpsampleMask computes FFT bin indices via (n_fft * freq / sampling_rate),
    # so the cutoff MUST be in Hz to produce the correct bin boundary.
    command = [
        "python3", script_name,
        "-f", input_path,
        "-o", output_path,
        "-n", str(int(steps)),
        "-c", str(int(cutoff_hz)),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    print(f"[A2SB] command: {' '.join(command)}")
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

def restore_audio(input_file, steps, cutoff_choice, progress=gr.Progress()):
    if not input_file:
        return None, None

    # Step 1: Initialization
    progress(0.05, desc="Initializing & Loading Audio...")
    cutoff_hz = int(cutoff_choice.lower().replace("khz", "")) * 1000
    
    try:
        audio = AudioSegment.from_file(input_file)
        audio = ensure_a2sb_input_format(audio)
    except Exception as e:
        raise gr.Error(f"Failed to load audio: {e}")

    # Create safe base name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    base_name = base_name.replace(" ", "_") # Sanitize spaces
    
    final_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_restored.wav")
    comparison_input_path = os.path.join(INPUT_DIR, f"{base_name}_filtered_input.wav")

    # Helper for channel processing
    def process_channel(segment, channel_id, display_name, start_prog, end_prog):
        prog_range = end_prog - start_prog
        
        # A. Filter
        progress(start_prog, desc=f"[{display_name}] Applying {cutoff_hz}Hz Filter...")
        filtered = apply_lowpass_to_segment(segment, cutoff_hz)
        
        temp_in = os.path.join(INPUT_DIR, f"temp_{channel_id}.wav")
        temp_out = os.path.join(OUTPUT_DIR, f"temp_{channel_id}_restored.wav")
        
        if os.path.exists(temp_out):
            os.remove(temp_out)
            
        filtered.export(temp_in, format="wav")
        
        # B. Restore
        restore_point = start_prog + (prog_range * 0.2)
        progress(restore_point, desc=f"[{display_name}] Running A2SB Inference...")
        
        # Pass cutoff_hz here
        run_a2sb_inference(temp_in, temp_out, steps, cutoff_hz)
        
        if not os.path.exists(temp_out):
             raise Exception(f"Inference script failed to generate {temp_out}")
        
        return temp_out, filtered

    try:
        final_filtered_audio = None

        if audio.channels == 1:
            restored_path, filtered_seg = process_channel(audio, "mono", "Mono Channel", 0.1, 0.9)
            final_filtered_audio = filtered_seg
            subprocess.run(["cp", restored_path, final_output_path])

        elif audio.channels == 2:
            progress(0.1, desc="Splitting Stereo Channels...")
            channels = audio.split_to_mono()
            
            out_l_path, filtered_l = process_channel(channels[0], "left", "Left Channel", 0.15, 0.5)
            out_r_path, filtered_r = process_channel(channels[1], "right", "Right Channel", 0.5, 0.85)

            progress(0.85, desc="Recombining Stereo Channels...")
            restored_l = AudioSegment.from_file(out_l_path)
            restored_r = AudioSegment.from_file(out_r_path)
            
            restored_stereo = AudioSegment.from_mono_audiosegments(restored_l, restored_r)
            restored_stereo.export(final_output_path, format="wav")
            
            final_filtered_audio = AudioSegment.from_mono_audiosegments(filtered_l, filtered_r)

        else:
            raise gr.Error(f"Unsupported channels: {audio.channels}")
            
        # Step 3: Analysis
        progress(0.9, desc="Generating Spectral Analysis...")
        final_filtered_audio.export(comparison_input_path, format="wav")
        plot_path = generate_comparison_plot(comparison_input_path, final_output_path)

        # Step 4: Finalize
        progress(1.0, desc="Done!")
        return final_output_path, plot_path

    except subprocess.CalledProcessError as e:
        print("STDERR:", e.stderr)
        raise gr.Error(f"Restoration failed: {e.stderr}")
    except Exception as e:
        print("Error:", str(e))
        raise gr.Error(f"Processing error: {str(e)}")

# --- Interface ---

if __name__ == "__main__":
    custom_css = "body { background-color: #121212; color: white; }"

    iface = gr.Interface(
        fn=restore_audio,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio"),
            gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Steps (Quality)"),
            gr.Dropdown(choices=["4kHz", "14kHz", "16kHz"], value="14kHz", label="Input Lowpass Filter (Cutoff)")
        ],
        outputs=[
            gr.Audio(label="Restored Result"),
            gr.Image(label="Spectral Analysis (Before vs After)")
        ],
        title="NVIDIA A2SB Stereo Restorer",
        description="Upload audio to simulate bandwidth loss and restore it. Progress bar tracks splitting, restoring, and analysis steps.",
    )

    iface.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
