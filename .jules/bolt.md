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

## 2026-06-13 - Optimize librosa spectral feature extraction
**Learning:** `librosa.feature.spectral_flatness` and `librosa.feature.spectral_rolloff` default to `hop_length=n_fft//4` (512 when `n_fft=2048`), producing 75% STFT frame overlap. For coarse heuristic checks that aggregate over time (mean, percentile), this overlap computes far more frames than needed.
**Action:** Set `hop_length=n_fft` (e.g., `hop_length=2048`) to eliminate the overlap when using these features for macroscopic heuristic checks. Keep `n_fft` at its default 2048 — only `hop_length` needs to change. This yields ~4× fewer STFT frames and proportionally less compute, with no meaningful accuracy loss for mean/percentile aggregations.

## 2026-07-14 - librosa.stft Overlap in Heuristics
**Learning:** Replacing an STFT bin-average RMS calculation with a time-domain IIR filter changes the mathematical output entirely. To safely optimize macroscopic heuristic functions that aggregate STFT data (like high-band RMS), do not reduce overlap to 0% (`hop_length=n_fft`) as it causes blind spots with a Hann window or severe spectral leakage with a boxcar window. Instead, use a 50% overlap (`hop_length=n_fft//2`) for a safe 2x speedup without sacrificing signal integrity.
**Action:** Use `hop_length=n_fft//2` when calculating `librosa.stft` for macroscopic heuristic functions that aggregate STFT data.
