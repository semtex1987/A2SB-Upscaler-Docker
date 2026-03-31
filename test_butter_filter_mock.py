import unittest
from unittest.mock import MagicMock
import sys

# Mock all dependencies of app.py
mock_numpy = MagicMock()
mock_scipy = MagicMock()
mock_gradio = MagicMock()
mock_matplotlib = MagicMock()
mock_librosa = MagicMock()
mock_pydub = MagicMock()

sys.modules['numpy'] = mock_numpy
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.signal'] = mock_scipy.signal
sys.modules['gradio'] = mock_gradio
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib.pyplot
sys.modules['librosa'] = mock_librosa
sys.modules['librosa.display'] = mock_librosa.display
sys.modules['pydub'] = mock_pydub

import app


class TestButterLowpassFilterMock(unittest.TestCase):
    def setUp(self):
        # Clear lru_cache for get_butter_sos so tests aren't using cached responses
        # from a previous test and bypass patched mock behavior
        if hasattr(app.get_butter_sos, 'cache_clear'):
            app.get_butter_sos.cache_clear()

    def test_butter_lowpass_filter_calls_sosfilt_with_correct_axis(self):
        app.butter = MagicMock(return_value='dummy_sos')
        app.sosfilt = MagicMock(return_value='filtered_data')

        fake_data = MagicMock()
        fake_data.dtype = 'float32'
        app.np.asarray = MagicMock(return_value=fake_data)
        app.np.issubdtype = MagicMock(return_value=False)

        result = app.butter_lowpass_filter(fake_data, 4000, 44100)

        self.assertEqual(result, 'filtered_data')
        app.sosfilt.assert_called_once()
        _, kwargs = app.sosfilt.call_args
        self.assertEqual(kwargs.get('axis'), 0)

    def test_integer_path_clips_before_cast(self):
        app.butter = MagicMock(return_value='dummy_sos')

        fake_data = MagicMock()
        fake_data.dtype = 'int16'
        app.np.asarray = MagicMock(return_value=fake_data)
        app.np.issubdtype = MagicMock(return_value=True)

        type_info = MagicMock()
        type_info.min = -32768
        type_info.max = 32767
        app.np.iinfo = MagicMock(return_value=type_info)

        data_float = MagicMock()
        fake_data.astype.return_value = data_float
        normalized = MagicMock()
        data_float.__truediv__.return_value = normalized

        filtered = MagicMock()
        app.sosfilt = MagicMock(return_value=filtered)

        clipped = MagicMock()
        app.np.clip = MagicMock(return_value=clipped)
        rounded = MagicMock()
        app.np.round = MagicMock(return_value=rounded)

        casted = MagicMock()
        rounded.__mul__.return_value = rounded
        rounded.astype.return_value = casted

        result = app.butter_lowpass_filter(fake_data, 4000, 44100)

        self.assertEqual(result, casted)
        app.np.clip.assert_called_once()


if __name__ == '__main__':
    unittest.main()
