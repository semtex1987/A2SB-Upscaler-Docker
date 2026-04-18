## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.
## 2024-06-25 - Precompute temporal embeddings in sampling loops
**Learning:** In the DDPM sampling loops, `t_to_emb` (which maps timestep tensors to embedding vectors) is a PyTorch operation that repeatedly calculates the exact same embedding values on every iteration, since the timestep schedule (`t_steps`) is predefined. This creates unnecessary PyTorch overhead inside tight loops.
**Action:** Always precalculate and hoist deterministic tensor operations like temporal embeddings (e.g. `all_t_embs = [self.t_to_emb(t_steps[:, t_idx]).repeat(...) for t_idx in range(n_steps)]`) outside of the sampling loops to reduce per-iteration overhead.
