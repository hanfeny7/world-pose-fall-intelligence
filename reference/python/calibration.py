"""Confidence calibration and decision diagnostics.

These helpers operate on already-produced probabilities. They do not expose
the private model's logits, checkpoint, or training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    temperature: float = 1.0
    prior: float = 0.0
    quality_floor: float = 0.25

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must be in [0, 1]")


class ConfidenceCalibrator:
    """Apply a transparent quality-aware calibration curve."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()

    def calibrate(self, probability: float, quality: float = 1.0) -> float:
        probability = clamp(probability)
        quality = clamp(quality)
        if quality < self.config.quality_floor:
            return probability * (quality / self.config.quality_floor)
        centered = (probability - 0.5) / self.config.temperature
        adjusted = 0.5 + centered + self.config.prior * (quality - 0.5)
        return clamp(adjusted)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> BinaryMetrics:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and equally sized")
    predicted = [int(probability >= threshold) for probability in probabilities]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predicted))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predicted))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predicted))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        accuracy=(tp + tn) / len(labels),
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / (fp + tn) if fp + tn else 0.0,
    )


def best_threshold(labels: Sequence[int], probabilities: Sequence[float], steps: int = 101) -> tuple[float, BinaryMetrics]:
    if steps < 2:
        raise ValueError("steps must be >= 2")
    candidates = [index / (steps - 1) for index in range(steps)]
    scored = [(threshold, binary_metrics(labels, probabilities, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1].f1, item[1].recall, -item[0]))
