## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.
## 2026-04-06 - Precalculate ML model refs before loops
**Learning:** Hoisting model retrieval logic (e.g., `get_vf_model`) outside of tight diffusion sampling loops by pre-calculating model references for all timesteps significantly reduces Python overhead per iteration.
**Action:** When implementing iterative sampling algorithms, precompute state variables such as model partitions to avoid redundant O(N) lookup overhead on every step.
## 2024-05-18 - Precalculate temporal embeddings outside sampling loops
**Learning:** In DDPM sampling loops, computing temporal embeddings iteratively via `t_emb = self.t_to_emb(t_steps[:, t_idx]).repeat(x_1.shape[0], 1)` incurs severe PyTorch overhead, performing O(N) model lookups and tensor repeat operations.
**Action:** Always vectorize and hoist deterministic embedding lookups outside tight loops. For example, compute the entire embedding space (`t_embs = self.t_to_emb(t_steps[0, :-1]).unsqueeze(1).repeat(1, x_1.shape[0], 1)`) beforehand, and simply slice `t_emb = t_embs[t_idx]` during each loop iteration. This can reduce embedding generation time from ~0.020s down to ~0.001s for 100 steps.
