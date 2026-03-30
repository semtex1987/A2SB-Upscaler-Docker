## 2024-05-18 - Avoid librosa default resampling
**Learning:** `librosa.load()` defaults to `sr=22050`, which triggers an extremely slow resampling process. In this codebase, where we want to preserve high native sampling rates (e.g., 44.1kHz), this default behavior is a severe and hidden performance bottleneck.
**Action:** Always specify `sr=None` when using `librosa.load()` to load files at their native sample rate and avoid unnecessary downsampling overhead.

## 2024-05-19 - Fast audio duration lookup
**Learning:** `librosa.get_duration()` is very slow because it actually decodes the entire audio file to determine duration. For long files, this takes many seconds.
**Action:** Use `soundfile.info(path).duration` instead of `librosa.get_duration()` when possible, which only reads the metadata/header and finishes almost instantaneously.
