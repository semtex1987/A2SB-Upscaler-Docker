## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2024-05-24 - [Scipy Filter Bottleneck]
**Learning:** `scipy.signal.butter` is computationally expensive relative to the actual filtering operation (`sosfilt`), taking roughly 2x longer for small to medium data arrays. When cutoff frequencies and sampling rates are relatively static (e.g. predefined UI choices), computing these coefficients repeatedly wastes time.
**Action:** Always memoize `scipy.signal.butter` coefficient generation using `@functools.lru_cache` when processing sequential streams or batches with a fixed set of filter parameters.
