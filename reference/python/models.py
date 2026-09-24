"""Public data contracts for the SentinelFlow showcase.

The production repository contains richer, private evidence objects. These
small contracts are intentionally stable and safe to use in integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Mapping, Sequence


class Modality(StrEnum):
    VIDEO = "video"
    POSE = "pose"
    AUDIO = "audio"
    TEMPORAL = "temporal"


class EventState(StrEnum):
    OBSERVED = "observed"
    VERIFIED = "verified"
    ESCALATED = "escalated"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Severity(StrEnum):
    INFO = "info"
    GUARDED = "guarded"
    HIGH = "high"
    CRITICAL = "critical"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    observed_at: datetime
    source: str
    quality: float
    available: frozenset[Modality]
    contributing: frozenset[Modality]
    pose_summary: Mapping[str, float] = field(default_factory=dict)
    model_revision: str = "public-contract-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Evidence:
    confidence: float
    quality: float
    modalities: frozenset[Modality]
    reason_code: str


@dataclass(slots=True)
class SafetyEvent:
    id: str
    state: EventState
    severity: Severity
    reason_code: str
    evidence: Evidence
    created_at: datetime = field(default_factory=utc_now)
    history: list[EventState] = field(default_factory=list)

    def transition(self, next_state: EventState) -> None:
        allowed = {
            EventState.OBSERVED: {EventState.VERIFIED, EventState.RESOLVED},
            EventState.VERIFIED: {EventState.ESCALATED, EventState.RESOLVED},
            EventState.ESCALATED: {EventState.ACKNOWLEDGED, EventState.RESOLVED},
            EventState.ACKNOWLEDGED: {EventState.RESOLVED},
            EventState.RESOLVED: set(),
        }
        if next_state not in allowed[self.state]:
            raise ValueError(f"invalid event transition: {self.state} -> {next_state}")
        self.history.append(self.state)
        self.state = next_state


def modalities(*items: Modality) -> frozenset[Modality]:
    return frozenset(items)
