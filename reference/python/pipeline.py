"""Dependency-inverted inference pipeline skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from models import Modality, Observation
from policy import EventPolicy, SafetyEvent


class EvidenceProvider(Protocol):
    modality: Modality

    def observe(self, frame: object) -> dict[str, float]: ...


@dataclass(slots=True)
class PipelineResult:
    observation: Observation
    event: SafetyEvent | None


class InferencePipeline:
    """Wires providers to policy without exposing private model internals."""

    def __init__(self, providers: Sequence[EvidenceProvider], policy: EventPolicy):
        self.providers = tuple(providers)
        self.policy = policy

    def process(self, frame: object, *, source: str, observation_id: str) -> PipelineResult:
        summaries: dict[str, float] = {}
        available: set[Modality] = set()
        contributing: set[Modality] = set()

        for provider in self.providers:
            available.add(provider.modality)
            result = provider.observe(frame)
            summaries.update(result)
            if result:
                contributing.add(provider.modality)

        quality = float(summaries.get("quality", 0.0))
        observation = Observation(
            id=observation_id,
            observed_at=__import__("models").utc_now(),
            source=source,
            quality=quality,
            available=frozenset(available),
            contributing=frozenset(contributing),
            pose_summary=summaries,
        )
        return PipelineResult(observation, self.policy.evaluate(observation))
