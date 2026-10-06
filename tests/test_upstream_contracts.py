"""Source contracts for bounded training I/O and fenced dormant upstream paths."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "nvidia-a2sb-original-repo" / "datasets" / "datasets.py"
API_MODULE = REPO / "nvidia-a2sb-original-repo" / "A2SB_lightning_module_api.py"
LIGHTNING_MODULE = REPO / "nvidia-a2sb-original-repo" / "A2SB_lightning_module.py"
NETWORKS = REPO / "nvidia-a2sb-original-repo" / "networks.py"
AUDIO_UTILS = REPO / "nvidia-a2sb-original-repo" / "audio_utils.py"


def test_training_loader_reads_segments_not_whole_files() -> None:
    text = DATASETS.read_text(encoding="utf-8")
    assert "offset=offset" in text
    assert "duration=duration" in text
    assert "SEGMENT_LOAD_CONTEXT_SEC" in text


def test_dataset_error_fallback_is_not_modulo_99() -> None:
    text = DATASETS.read_text(encoding="utf-8")
    assert "% 99" not in text
    assert "(index + 1) % n" in text


def test_inference_drops_the_template_network() -> None:
    text = API_MODULE.read_text(encoding="utf-8")
    assert "self.vf_model = nn.Identity()" in text
    assert "vocode_stft(x_0_corrupted)" not in text


def test_dormant_helpers_are_fenced() -> None:
    networks = NETWORKS.read_text(encoding="utf-8")
    audio_utils = AUDIO_UTILS.read_text(encoding="utf-8")
    api = API_MODULE.read_text(encoding="utf-8")
    lightning = LIGHTNING_MODULE.read_text(encoding="utf-8")
    assert "NotImplementedError" in networks
    assert "NotImplementedError" in audio_utils
    assert "breakpoint()" not in api
    assert "breakpoint()" not in lightning
    assert "leftover debug stub" in api
    assert "leftover debug stub" in lightning
