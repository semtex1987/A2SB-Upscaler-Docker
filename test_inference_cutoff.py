import unittest
from unittest.mock import MagicMock
import sys
import subprocess
import os

# Mock external deps for app import
mock_gradio = MagicMock()
mock_matplotlib = MagicMock()
mock_librosa = MagicMock()
mock_pydub = MagicMock()
mock_numpy = MagicMock()
mock_scipy = MagicMock()

sys.modules['gradio'] = mock_gradio
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib.pyplot
sys.modules['librosa'] = mock_librosa
sys.modules['librosa.display'] = mock_librosa.display
sys.modules['pydub'] = mock_pydub
sys.modules['numpy'] = mock_numpy
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.signal'] = mock_scipy.signal

import app


class TestInferenceCutoffUnits(unittest.TestCase):
    def test_passes_cutoff_in_hz(self):
        ok = MagicMock(stdout='', stderr='')
        app.subprocess.run = MagicMock(return_value=ok)
        app.os.path.exists = MagicMock(return_value=True)
        app.is_likely_corrupted_audio = MagicMock(return_value=False)

        app.run_a2sb_inference('/tmp/in.wav', '/tmp/out.wav', 50, 14000)

        args, _ = app.subprocess.run.call_args
        cmd = args[0]
        self.assertIn('-c', cmd)
        self.assertEqual(cmd[cmd.index('-c') + 1], '14000')


if __name__ == '__main__':
    unittest.main()
