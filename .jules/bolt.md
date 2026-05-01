## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.
## 2024-05-15 - Vectorize temporal embedding inside DDPM sampling loop
**Learning:** Evaluating the time embedding (`t_to_emb`) inside the DDPM sampling loop for a single scalar/tensor at a time introduces redundant neural network evaluation overhead and kernel launches.
**Action:** Precalculate and hoist deterministic tensor operations like temporal embeddings (`t_to_emb`) outside of the loops. Ensure you *vectorize* the input across all timesteps (e.g., `t_embs = self.t_to_emb(t_steps[0, :n_steps])`) in a single batched pass rather than using a list comprehension, as list comprehensions still execute the network O(N) times. Inside the loop, slice the precomputed embeddings (e.g. `t_embs[t_idx:t_idx+1]`) and use `repeat` to match the required batch size.
