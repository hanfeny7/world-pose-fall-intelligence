"""Reviewable event policy skeleton.

The scoring and learned fusion logic are intentionally injected by the private
runtime. This module only demonstrates lifecycle semantics and guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
from uuid import uuid4

from models import EventState, Evidence, Observation, SafetyEvent, Severity


ScoreFn = Callable[[Observation], float]


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    verify_at: float = 0.70
    escalate_at: float = 0.88
    minimum_quality: float = 0.35


class EventPolicy:
    """Turns one observation into an auditable event decision.

    In production, ``score`` is supplied by a private multimodal runtime.
    The public default is deliberately conservative and deterministic.
    """

    def __init__(self, config: PolicyConfig | None = None, score: ScoreFn | None = None):
        self.config = config or PolicyConfig()
        self.score = score or (lambda observation: observation.quality)

    def evaluate(self, observation: Observation) -> SafetyEvent | None:
        confidence = max(0.0, min(1.0, self.score(observation)))
        if observation.quality < self.config.minimum_quality:
            return None
        if confidence < self.config.verify_at:
            return None

        event = SafetyEvent(
            id=f"evt_{uuid4().hex[:12]}",
            state=EventState.OBSERVED,
            severity=Severity.GUARDED,
            reason_code="evidence_threshold_reached",
            evidence=Evidence(
                confidence=confidence,
                quality=observation.quality,
                modalities=observation.contributing,
                reason_code="evidence_threshold_reached",
            ),
        )
        event.transition(EventState.VERIFIED)
        if confidence >= self.config.escalate_at:
            event.severity = Severity.HIGH
            event.transition(EventState.ESCALATED)
        return event


def summarize(events: Iterable[SafetyEvent]) -> list[dict[str, object]]:
    """Create a stable, redacted view for a console or audit sink."""

    return [
        {
            "id": event.id,
            "state": event.state.value,
            "severity": event.severity.value,
            "reason_code": event.reason_code,
            "confidence": round(event.evidence.confidence, 3),
            "modalities": sorted(item.value for item in event.evidence.modalities),
        }
        for event in events
    ]
