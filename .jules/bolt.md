## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2024-05-19 - Fast Audio Duration Lookups
**Learning:** `librosa.get_duration()` is extremely slow for larger audio files because it decodes the entire file. `soundfile.info().duration` is much faster because it only reads the file header/metadata.
**Action:** Use `soundfile.info(path).duration` for fast audio duration lookups instead of `librosa.get_duration()`. Fallback to librosa if soundfile fails.
