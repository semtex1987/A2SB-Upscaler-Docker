## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.

## 2024-05-24 - Speeding up corrupted audio check
**Learning:** `pydub` is slow for loading WAV files directly as it checks the format using fallback (via ffmpeg), introducing significant overhead.
**Action:** Always use `soundfile` with `dtype='int16'` to load WAV files directly into memory where format is known to be WAV to bypass this overhead, like `is_likely_corrupted_audio` function.
## 2024-05-24 - Faster spectral flatness approximation
**Learning:** `librosa.feature.spectral_flatness` is extremely slow with default `n_fft` and `hop_length` because it performs fine-grained FFT on the entire audio track. For high-level macro checks like "is this noise?" we do not need this resolution.
**Action:** Pass explicitly large `n_fft` and `hop_length` (e.g. `n_fft=2048`, `hop_length=2048`) to `librosa.feature.spectral_flatness` to speed up the operation by magnitudes while maintaining macro accuracy.
