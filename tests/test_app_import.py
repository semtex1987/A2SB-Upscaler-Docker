import sys
from unittest.mock import MagicMock
import pytest

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

def test_app_importable():
    """
    Test that app.py can be imported without starting the Gradio server.
    """
    # Reset mock to clear any previous calls
    import gradio
    gradio.Interface.return_value.launch.reset_mock()

    try:
        # Clear app from sys.modules if it was already imported
        if "app" in sys.modules:
            del sys.modules["app"]
        import app

        # Verify that launch was NOT called
        import gradio
        assert not gradio.Interface.return_value.launch.called, "gr.Interface().launch() was called on import!"

    except Exception as e:
        pytest.fail(f"Importing app failed with error: {e}")

def test_app_functions_accessible():
    """
    Test that functions in app.py are accessible after import.
    """
    import app
    assert callable(app.butter_lowpass_filter)
    assert callable(app.restore_audio)

def test_iface_not_global_without_main():
    """
    Since we moved iface inside the if __name__ == "__main__": block,
    it should not be available as a global variable when app is imported.
    """
    import app
    assert not hasattr(app, 'iface')
