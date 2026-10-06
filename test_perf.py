import time
import librosa
import numpy as np

print("Generating random audio...")
y = np.random.randn(44100 * 120)  # 2 minutes of audio at 44.1kHz
sr = 44100
n_fft = 4096

print("Warming up JIT...")
_ = np.abs(librosa.stft(y[:4096*2], n_fft=n_fft, hop_length=1024))
_ = np.abs(librosa.stft(y[:4096*2], n_fft=n_fft, hop_length=n_fft//2))

print("Testing baseline (hop_length=1024)...")
t0 = time.time()
spec_base = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=1024))
p95_base = np.percentile(spec_base, 95, axis=1)
t1 = time.time()
print(f"Baseline took: {t1 - t0:.3f}s")

print("Testing 50% overlap (hop_length=n_fft//2)...")
t0 = time.time()
spec_opt1 = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft//2))
p95_opt1 = np.percentile(spec_opt1, 95, axis=1)
t1 = time.time()
print(f"50% overlap took: {t1 - t0:.3f}s")

diff1 = np.max(np.abs(p95_base - p95_opt1))
print(f"Max diff with 50% overlap: {diff1}")
