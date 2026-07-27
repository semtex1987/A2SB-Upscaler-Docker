"""Dataclass -> JSON conversion.

The wire format is camelCase throughout so the frontend never mixes casing
styles. Persisted job files use the same shape, so a job written by one release
reads back unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def to_snake(name: str) -> str:
    return "".join(f"_{char.lower()}" if char.isupper() else char for char in name)


def camelize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return camelize(asdict(value))
    if isinstance(value, dict):
        return {to_camel(str(key)): camelize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [camelize(item) for item in value]
    return value


def snakeize(value: Any) -> Any:
    if isinstance(value, dict):
        return {to_snake(str(key)): snakeize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snakeize(item) for item in value]
    return value
