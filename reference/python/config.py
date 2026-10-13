"""Validated public configuration for the Temporal-V5 example."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamConfig:
    source_name: str = "demo-camera"
    fps: float = 12.0
    window_size: int = 32
    min_quality: float = 0.20
    device: str = "cpu"

    def validate(self) -> "StreamConfig":
        if not self.source_name.strip():
            raise ValueError("source_name cannot be empty")
        if not 1.0 <= self.fps <= 120.0:
            raise ValueError("fps must be between 1 and 120")
        if not 2 <= self.window_size <= 512:
            raise ValueError("window_size must be between 2 and 512")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda, or mps")
        return self

    @classmethod
    def from_env(cls) -> "StreamConfig":
        def number(name: str, default: str) -> float:
            return float(os.getenv(name, default))

        config = cls(
            source_name=os.getenv("SENTINEL_SOURCE", "demo-camera"),
            fps=number("SENTINEL_FPS", "12"),
            window_size=int(os.getenv("SENTINEL_WINDOW", "32")),
            min_quality=number("SENTINEL_MIN_QUALITY", "0.20"),
            device=os.getenv("SENTINEL_DEVICE", "cpu"),
        )
        return config.validate()
