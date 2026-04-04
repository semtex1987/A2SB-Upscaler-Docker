## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2024-05-18 - Memoize scipy.signal.butter coefficients
**Learning:** In audio processing loops or batch operations, `scipy.signal.butter()` is computationally expensive relative to the actual filtering operation (`sosfilt`), taking around ~1.5ms per call. For static filter configurations (e.g., fixed cutoff frequencies and orders), generating identical coefficients repeatedly is a severe performance anti-pattern.
**Action:** Always use `@functools.lru_cache` (or similar memoization) on the coefficient generation logic for Butterworth filters when processing sequential or batch audio with static parameters.
