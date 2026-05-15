## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.

## 2024-05-18 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.

## 2024-05-18 - Precalculate temporal embeddings
**Learning:** In DDPM sampling loops, computing temporal embeddings inside the loop using `t_to_emb` introduces unnecessary overhead. Since `t_steps` are known ahead of time, we can precalculate all embeddings in a single batched pass.
**Action:** Always precalculate and vectorize deterministic tensor operations like temporal embeddings outside of iterative loops. Use `.unsqueeze()` and `.repeat()` to construct a tensor matching the required loop output shape, so the loop can simply perform a fast index lookup.
## 2025-05-15 - Optimize `librosa.feature` Bottleneck
**Learning:** `librosa.feature.spectral_flatness` and `librosa.feature.spectral_rolloff` are severe computational bottlenecks with default arguments (`hop_length=512`, `n_fft=2048`). By explicitly increasing `hop_length` and `n_fft` (e.g., to 2048), we eliminate the 75% STFT overlap. The flatness feature computation became about 1000x faster, and rolloff became 5x faster in a simple micro-benchmark.
**Action:** Explicitly set `n_fft=2048` and `hop_length=2048` when calling these functions if using them for macroscopic heuristic checks.
