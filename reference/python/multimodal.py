"""Auditable visual/audio fusion primitives for public integration builds."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def logit(probability: float, eps: float = 1e-6) -> float:
    value = clamp(probability, eps, 1.0 - eps)
    return log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class ModalityEvidence:
    name: str
    probability: float
    quality: float
    available: bool = True
    latency_ms: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("modality name cannot be empty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FusionDecision:
    probability: float
    label: str
    severity: str
    gate: float
    modalities: tuple[str, ...]
    reason: str
    audit: dict[str, float]


class ReliabilityGate:
    """Compute a conservative modality gate from quality and availability."""

    def __init__(self, minimum_quality: float = 0.20, audio_bias: float = 0.15) -> None:
        if not 0.0 <= minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in [0, 1]")
        self.minimum_quality = minimum_quality
        self.audio_bias = audio_bias

    def __call__(self, visual: ModalityEvidence, audio: ModalityEvidence | None) -> float:
        if audio is None or not audio.available or audio.quality < self.minimum_quality:
            return 0.0
        visual_quality = max(visual.quality, self.minimum_quality)
        audio_quality = max(audio.quality, self.minimum_quality)
        balance = min(visual_quality, audio_quality) / max(visual_quality, audio_quality)
        return clamp(0.50 * audio_quality + 0.35 * balance + self.audio_bias)


class AudioVisualFusion:
    """Fuse evidence in logit space while preserving a visual fallback."""

    def __init__(
        self,
        gate: ReliabilityGate | None = None,
        visual_weight: float = 0.75,
        audio_weight: float = 0.25,
        high_threshold: float = 0.78,
        critical_threshold: float = 0.92,
    ) -> None:
        if visual_weight < 0 or audio_weight < 0 or visual_weight + audio_weight <= 0:
            raise ValueError("fusion weights must be non-negative and non-zero")
        self.gate = gate or ReliabilityGate()
        total = visual_weight + audio_weight
        self.visual_weight = visual_weight / total
        self.audio_weight = audio_weight / total
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold

    def fuse(self, visual: ModalityEvidence, audio: ModalityEvidence | None = None) -> FusionDecision:
        gate = self.gate(visual, audio)
        visual_logit = logit(visual.probability)
        if audio is None or not audio.available:
            probability = sigmoid(visual_logit)
            reason = "visual-only fallback; audio unavailable"
            modalities = (visual.name,)
        else:
            audio_logit = logit(audio.probability)
            weighted = self.visual_weight * visual_logit + self.audio_weight * audio_logit
            cross_probability = sigmoid(weighted)
            probability = (1.0 - gate) * visual.probability + gate * cross_probability
            reason = "gated visual/audio evidence" if gate > 0 else "audio below reliability floor"
            modalities = (visual.name, audio.name)
        probability = clamp(probability)
        severity = self._severity(probability)
        label = "FALL" if probability >= self.high_threshold else "NO_FALL"
        return FusionDecision(
            probability=probability,
            label=label,
            severity=severity,
            gate=gate,
            modalities=modalities,
            reason=reason,
            audit={"visual_probability": visual.probability, "audio_probability": audio.probability if audio else 0.0, "visual_quality": visual.quality, "audio_quality": audio.quality if audio else 0.0},
        )

    def _severity(self, probability: float) -> str:
        if probability >= self.critical_threshold:
            return "critical"
        if probability >= self.high_threshold:
            return "high"
        if probability >= 0.50:
            return "guarded"
        return "none"


def aggregate_modalities(evidence: Iterable[ModalityEvidence]) -> dict[str, float]:
    """Return a redacted audit summary suitable for a JSON event."""

    values = [item for item in evidence if item.available]
    if not values:
        return {"available": 0.0, "mean_quality": 0.0, "mean_probability": 0.0}
    return {
        "available": float(len(values)),
        "mean_quality": sum(item.quality for item in values) / len(values),
        "mean_probability": sum(item.probability for item in values) / len(values),
    }
