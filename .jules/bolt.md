## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2024-05-18 - scipy.signal.butter optimization
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`).
**Action:** For functions using static cutoff frequencies and order, memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing.
