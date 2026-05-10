## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.

## 2024-05-14 - Vectorized Temporal Embeddings
**Learning:** PyTorch neural network module calls inside rapid sampling loops (like diffusion reverse steps) add high CPU dispatch overhead. Calling `.repeat()` and calculating step embeddings independently inside a loop of length `N` invokes the network module `N` times with a batch size of 1.
**Action:** Always inspect iterative deep learning sampling loops (like `ddpm_sample`). Precalculate static mappings—such as step embeddings `t_to_emb`—for all steps simultaneously before the loop by passing an array of `t_steps`. This vectorizes the operation to a single `O(1)` module call, and the resulting tensor can just be indexed at each `t_idx` inside the loop, substantially decreasing Python overhead.
