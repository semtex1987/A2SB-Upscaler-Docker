"""Casing conversion, which every API response and persisted job passes through."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from server.serialization import camelize, snakeize, to_camel, to_snake


@pytest.mark.parametrize(
    "snake,camel",
    [
        ("cutoff_hz", "cutoffHz"),
        ("high_band_delta_db", "highBandDeltaDb"),
        ("id", "id"),
        ("eta_sec", "etaSec"),
    ],
)
def test_key_names_round_trip(snake, camel):
    assert to_camel(snake) == camel
    assert to_snake(camel) == snake


def test_an_already_camel_key_is_left_alone():
    """Restored jobs are re-camelised on the way out; this must be idempotent."""
    assert to_camel("cutoffHz") == "cutoffHz"


@dataclass
class Inner:
    sample_rate: int


@dataclass
class Outer:
    job_id: str
    inner_items: list = field(default_factory=list)
    nested_map: dict = field(default_factory=dict)


def test_dataclasses_convert_recursively():
    payload = camelize(
        Outer(job_id="abc", inner_items=[Inner(44100)], nested_map={"source_path": "/a.wav"})
    )

    assert payload == {
        "jobId": "abc",
        "innerItems": [{"sampleRate": 44100}],
        "nestedMap": {"sourcePath": "/a.wav"},
    }


def test_values_are_not_touched():
    payload = camelize({"source_path": "/some_dir/my_file.wav", "eta_sec": None})
    assert payload == {"sourcePath": "/some_dir/my_file.wav", "etaSec": None}


def test_a_persisted_job_survives_a_round_trip():
    original = {
        "id": "abc",
        "batch_size": 16,
        "files": [{"cutoff_hz": 14000, "result": {"high_band_delta_db": 12.5}}],
    }
    assert snakeize(camelize(original)) == original
