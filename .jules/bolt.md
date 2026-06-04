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

## 2025-02-28 - Optimize lazy loading of audio segments with offset and duration
**Learning:** Loading an entire audio file into memory just to crop a tiny segment (using librosa.load defaults) creates a massive memory and CPU bottleneck, primarily due to full-file resampling. Using librosa.get_duration combined with librosa.load's offset and duration parameters allows for lazy loading only the required chunk.
**Action:** When extracting short segments from long audio files, avoid loading the full file. Calculate boundaries and use offset and duration parameters to load exactly the segment needed.
