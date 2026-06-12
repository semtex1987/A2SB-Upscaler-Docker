import sys
import unittest
from unittest.mock import MagicMock

# Mock all dependencies that are not available in the test environment
mock_modules = [
    "gradio",
    "matplotlib",
    "matplotlib.use",
    "matplotlib.pyplot",
    "librosa",
    "librosa.display",
    "scipy",
    "scipy.signal",
    "pydub",
    "numpy",
]

for module in mock_modules:
    sys.modules[module] = MagicMock()


class TestAppImport(unittest.TestCase):
    def setUp(self):
        if "app" in sys.modules:
            del sys.modules["app"]

    def test_app_importable(self):
        import gradio
        gradio.Interface.return_value.launch.reset_mock()

        try:
            import app
        except Exception as e:
            self.fail(f"Importing app failed with error: {e}")

        assert not gradio.Interface.return_value.launch.called, \
            "gr.Interface().launch() was called on import!"

    def test_app_functions_accessible(self):
        import app
        self.assertTrue(callable(app.butter_lowpass_filter))
        self.assertTrue(callable(app.restore_audio))

    def test_iface_not_global_without_main(self):
        import app
        self.assertFalse(hasattr(app, 'iface'))


if __name__ == "__main__":
    unittest.main()
