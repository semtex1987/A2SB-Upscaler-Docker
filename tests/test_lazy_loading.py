import sys
import unittest.mock as mock

class MockModule(mock.MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return mock.MagicMock()

# Setup mocks before importing anything else
librosa_mock = mock.MagicMock()
sys.modules['librosa'] = librosa_mock
sys.modules['torch'] = mock.MagicMock()
sys.modules['torch.utils'] = mock.MagicMock()
sys.modules['torch.utils.data'] = mock.MagicMock()
sys.modules['torch.nn'] = MockModule()
sys.modules['torch.nn.functional'] = MockModule()
sys.modules['torch.linalg'] = MockModule()
sys.modules['torchaudio'] = MockModule()
sys.modules['torchaudio.functional'] = MockModule()
sys.modules['torchaudio.transforms'] = MockModule()
sys.modules['numpy'] = mock.MagicMock()
sys.modules['einops'] = MockModule()
sys.modules['jsonargparse'] = MockModule()
sys.modules['rotary_embedding_torch'] = MockModule()
sys.modules['csv'] = mock.MagicMock()

# Make Dataset a real class so subclassing works
class DummyDatasetClass:
    pass
sys.modules['torch'].utils.data.Dataset = DummyDatasetClass

import datasets.datasets as ds

def test_lazy_loading_called():
    # It's important to use the same librosa_mock we injected into sys.modules
    librosa_mock.get_duration.return_value = 10.0

    audio_mock = mock.MagicMock()
    audio_mock.shape = (1000,)
    audio_mock.__len__.return_value = 1000
    audio_mock.__getitem__.return_value = audio_mock
    librosa_mock.load.return_value = (audio_mock, 44100)

    dataset = ds.MixAudioDataset.__new__(ds.MixAudioDataset)
    dataset.segment_length = 44100 * 2
    dataset.sampling_rate = 44100

    try:
        dataset.load_wav_to_torch('dummy.wav', start_time=1.0)
    except Exception as e:
        pass

    librosa_mock.get_duration.assert_called_once_with(path='dummy.wav')
    librosa_mock.load.assert_called_once_with('dummy.wav', sr=None, offset=1.0, duration=2.0)
