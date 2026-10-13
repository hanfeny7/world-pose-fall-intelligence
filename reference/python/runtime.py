"""Vendor-neutral streaming runtime around the public reference components."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Protocol

from event_store import EventStore
from fall_impact import FallAnalysisReport, FallImpactAnalyzer
from models import EventState, Evidence, Modality, SafetyEvent, Severity
from multimodal import AudioVisualFusion, FusionDecision, ModalityEvidence
from temporal_window import FrameEvidence, RobustSmoother, TemporalWindow, VoteAccumulator


class VisualProvider(Protocol):
    def __call__(self, frame: object) -> ModalityEvidence: ...


class AudioProvider(Protocol):
    def __call__(self, frame: object) -> ModalityEvidence | None: ...


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    window_size: int = 32
    vote_horizon: int = 8
    vote_required: int = 4
    event_cooldown_seconds: float = 8.0
    threshold: float = 0.72


@dataclass(frozen=True, slots=True)
class RuntimeFrame:
    index: int
    timestamp: float
    frame: object
    pose: object | None = None
    audio: object | None = None


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    index: int
    decision: FusionDecision
    promoted: bool
    event_id: str | None
    impact: FallAnalysisReport | None = None


class SentinelRuntime:
    """Compose perception providers, temporal policy, and event persistence."""

    def __init__(
        self,
        visual_provider: VisualProvider,
        audio_provider: AudioProvider | None = None,
        config: RuntimeConfig | None = None,
        store: EventStore | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.visual_provider = visual_provider
        self.audio_provider = audio_provider
        self.fusion = AudioVisualFusion()
        self.smoother = RobustSmoother()
        self.window = TemporalWindow(self.config.window_size)
        self.votes = VoteAccumulator(self.config.vote_required, self.config.vote_horizon)
        self.impact = FallImpactAnalyzer()
        self.store = store or EventStore()
        self._last_event_at = -float("inf")

    def process(self, item: RuntimeFrame) -> RuntimeResult:
        visual = self.visual_provider(item.frame)
        audio = self.audio_provider(item.audio) if self.audio_provider and item.audio is not None else None
        decision = self.fusion.fuse(visual, audio)
        smoothed = self.smoother.update(decision.probability)
        self.window.append(FrameEvidence(item.timestamp, smoothed, visual.quality, decision.label.lower()))
        positive = self.votes.push(smoothed >= self.config.threshold)
        can_emit = item.timestamp - self._last_event_at >= self.config.event_cooldown_seconds
        promoted = bool(positive and can_emit)
        event_id = None
        if promoted:
            event = SafetyEvent(
                id=f"runtime_{item.index:08d}",
                state=EventState.OBSERVED,
                severity=Severity.HIGH if decision.severity in {"high", "critical"} else Severity.GUARDED,
                reason_code="temporal_v5_vote_gate",
                evidence=Evidence(
                    confidence=smoothed,
                    quality=visual.quality,
                    modalities=frozenset({Modality.VIDEO, Modality.POSE} | ({Modality.AUDIO} if audio else set())),
                    reason_code=decision.reason,
                ),
            )
            event.transition(EventState.VERIFIED)
            self.store.create(event)
            event_id = event.id
            self._last_event_at = item.timestamp
            self.votes.reset()
        return RuntimeResult(item.index, decision, promoted, event_id)

    def reset(self) -> None:
        self.smoother = RobustSmoother()
        self.window.clear()
        self.votes.reset()
        self.impact.reset()
        self._last_event_at = -float("inf")


def make_static_provider(probability: float, quality: float = 0.9, name: str = "visual") -> VisualProvider:
    """Create a deterministic provider for demos and integration tests."""

    def provider(frame: object) -> ModalityEvidence:
        del frame
        return ModalityEvidence(name, probability, quality)

    return provider
