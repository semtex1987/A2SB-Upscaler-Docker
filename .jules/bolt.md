## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.

## 2025-02-28 - Hoisting Model Retrieval from Diffusion Loop
**Learning:** In the `ddpm_sample` function, calling `self.get_vf_model(t[0].item())` inside the tight inference loop introduced redundant Python overhead during each timestep iteration, significantly slowing down the multidiffusion sampling process.
**Action:** Always hoist model retrieval logic outside of tight diffusion sampling loops by pre-calculating model references for all timesteps (e.g., `vf_models = [self.get_vf_model(...) for ... ]`), effectively eliminating the redundant Python overhead per iteration.
