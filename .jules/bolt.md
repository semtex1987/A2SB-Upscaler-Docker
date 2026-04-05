## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Hoist model retrieval outside diffusion loop
**Learning:** Calling `get_vf_model(t)` inside the tight diffusion sampling loop incurs noticeable Python overhead, especially when combined with multi-diffusion slicing. Model retrieval is deterministic per timestep.
**Action:** Pre-calculate `step_models = [self.get_vf_model(t_steps[0, i].item()) for i in range(n_steps)]` before the sampling loop to eliminate repetitive lookups.
