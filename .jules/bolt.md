## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.

## 2025-02-28 - Fast reading for predictable WAV files
**Learning:** `pydub.AudioSegment.from_file()` relies on ffmpeg for fallback multi-format support, which incurs massive spin-up overhead. This codebase was using it even for internal `.wav` outputs.
**Action:** For operations where the file is guaranteed to be a `.wav` (like inference results), use `soundfile.read(path, dtype='int16')` to directly load samples into a NumPy array, skipping pydub completely.

## 2025-02-28 - Fast Spectral Flatness Validation
**Learning:** `librosa.feature.spectral_flatness` defaults to `n_fft=2048` and `hop_length=512`. The small hop length is excessively slow for high-level validation heuristics (like checking if audio is purely noise).
**Action:** When calculating spectral flatness for validation approximations, increase both `hop_length` and `n_fft` (e.g. to 2048) to significantly speed up processing without compromising the heuristic outcome.
