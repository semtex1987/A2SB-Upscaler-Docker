## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2025-02-28 - Memoize scipy filter generation
**Learning:** `scipy.signal.butter` is computationally expensive relative to actual filtering operations (`sosfilt`). For functions using static cutoff frequencies and order, this is redundant overhead.
**Action:** Memoize the filter coefficient generation using `@functools.lru_cache` to eliminate redundant calculations during sequential or batch processing, which speeds up operations like stereo channel splitting where the same filter is generated multiple times.

## 2025-03-09 - Top-level import bottleneck
**Learning:** Heavy modules like `librosa`, `scipy`, and `matplotlib` at the module scope can severely slow down script startup time. In this application, global imports added ~4.5 seconds to start-up.
**Action:** Keep heavy dependencies localized within functions (lazy-loading) if they are only needed sporadically or during specific user actions, removing them from the module's top level.
