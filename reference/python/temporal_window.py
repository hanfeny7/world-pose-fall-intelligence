"""Streaming temporal utilities used by the public Temporal-V5 example.

The private runtime feeds learned tokens into :mod:`temporal_v5`. This module
shows the surrounding streaming mechanics: bounded history, causal deltas,
robust smoothing, and a small vote accumulator for event review. Everything is
deterministic and independent of camera vendors or private checkpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from statistics import median
from typing import Deque, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """One frame-level signal after perception has completed."""

    timestamp: float
    confidence: float
    quality: float
    phase: str = "unknown"
    source: str = "stream"

    def __post_init__(self) -> None:
        for name, value in (("confidence", self.confidence), ("quality", self.quality)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class WindowSummary:
    """Causal summary presented to the policy layer."""

    size: int
    confidence_mean: float
    confidence_peak: float
    quality_mean: float
    confidence_slope: float
    phase: str
    stable: bool


class TemporalWindow:
    """A bounded, append-only window with no look-ahead access."""

    def __init__(self, capacity: int = 32, min_quality: float = 0.20) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        self.capacity = capacity
        self.min_quality = min_quality
        self._items: Deque[FrameEvidence] = deque(maxlen=capacity)

    def append(self, item: FrameEvidence) -> bool:
        """Append usable evidence and return whether it entered the window."""

        if item.quality < self.min_quality:
            return False
        if self._items and item.timestamp < self._items[-1].timestamp:
            raise ValueError("timestamps must be monotonic")
        self._items.append(item)
        return True

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    @property
    def ready(self) -> bool:
        return len(self._items) == self.capacity

    def values(self) -> list[FrameEvidence]:
        return list(self._items)

    def _slope(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        start = values[0]
        end = values[-1]
        elapsed = self._items[-1].timestamp - self._items[0].timestamp
        return (end - start) / elapsed if elapsed > 0 else 0.0

    def summarize(self, stable_delta: float = 0.18) -> WindowSummary | None:
        if not self._items:
            return None
        values = [item.confidence for item in self._items]
        qualities = [item.quality for item in self._items]
        phases = [item.phase for item in self._items]
        phase = max(set(phases), key=phases.count)
        slope = self._slope(values)
        spread = max(values) - min(values)
        return WindowSummary(
            size=len(values),
            confidence_mean=sum(values) / len(values),
            confidence_peak=max(values),
            quality_mean=sum(qualities) / len(qualities),
            confidence_slope=slope,
            phase=phase,
            stable=spread <= stable_delta,
        )


class RobustSmoother:
    """Median-first smoother that resists isolated confidence spikes."""

    def __init__(self, width: int = 5, blend: float = 0.35) -> None:
        if width < 3 or width % 2 == 0:
            raise ValueError("width must be an odd integer >= 3")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self.width = width
        self.blend = blend
        self._history: Deque[float] = deque(maxlen=width)
        self._value: float | None = None

    def update(self, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        self._history.append(value)
        center = median(self._history)
        self._value = center if self._value is None else self.blend * center + (1 - self.blend) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


@dataclass(slots=True)
class VoteAccumulator:
    """Hysteresis-style vote gate for frame-to-event promotion."""

    required: int = 4
    horizon: int = 8
    _votes: Deque[bool] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.required < 1 or self.horizon < self.required:
            raise ValueError("required must be positive and <= horizon")
        self._votes = deque(maxlen=self.horizon)

    def push(self, positive: bool) -> bool:
        self._votes.append(bool(positive))
        return self.positive_count >= self.required

    @property
    def positive_count(self) -> int:
        return sum(self._votes)

    @property
    def ratio(self) -> float:
        return self.positive_count / len(self._votes) if self._votes else 0.0

    def reset(self) -> None:
        self._votes.clear()


def rms(values: Iterable[float]) -> float:
    """Small diagnostic helper used by tests and CLI output."""

    values = list(values)
    return sqrt(sum(value * value for value in values) / len(values)) if values else 0.0
