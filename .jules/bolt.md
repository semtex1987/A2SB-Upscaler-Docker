## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.
## 2025-05-18 - Optimize macro-level audio checks by bypassing pydub overhead and increasing hop_length/n_fft for spectral_flatness
**Learning:** `librosa.feature.spectral_flatness` is a severe computational bottleneck with default arguments (`hop_length=512`, `n_fft=2048`). Additionally, using `pydub`'s `AudioSegment.from_file` induces slow generic audio loading overhead (often falling back to ffmpeg) which is unnecessary when we guarantee the file format is `.wav`.
**Action:** When validating intermediate or inferred `.wav` files (e.g. noise presence checks like `is_likely_corrupted_audio`), use `soundfile.read` directly instead of `pydub` to quickly read samples into a numpy array. Further, explicitly pass larger values to `n_fft` and `hop_length` (e.g. 2048) in `librosa.feature.spectral_flatness` to dramatically speed up macro-level metric approximations without sacrificing check validity.
