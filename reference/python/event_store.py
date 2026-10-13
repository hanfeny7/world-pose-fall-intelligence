"""In-memory event lifecycle and deduplication example.

Production deployments replace this store with a secured persistence layer.
The public implementation focuses on state transitions and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Iterable

from models import EventState, SafetyEvent, Severity


@dataclass(frozen=True, slots=True)
class AuditEntry:
    event_id: str
    action: str
    state: EventState
    created_at: datetime
    actor: str
    metadata: dict[str, str] = field(default_factory=dict)


class EventStore:
    """Thread-safe bounded event store suitable for a local demo."""

    def __init__(self, max_events: int = 500) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: dict[str, SafetyEvent] = {}
        self._audits: list[AuditEntry] = []
        self._lock = RLock()

    def create(self, event: SafetyEvent, actor: str = "policy") -> SafetyEvent:
        with self._lock:
            if event.id in self._events:
                raise ValueError(f"event already exists: {event.id}")
            self._events[event.id] = event
            self._record(event, "created", actor)
            self._trim()
            return event

    def transition(self, event_id: str, state: EventState, actor: str = "operator") -> SafetyEvent:
        with self._lock:
            event = self._events[event_id]
            event.transition(state)
            self._record(event, "transition", actor)
            return event

    def active(self) -> list[SafetyEvent]:
        with self._lock:
            return [event for event in self._events.values() if event.state != EventState.RESOLVED]

    def recent(self, limit: int = 20) -> list[SafetyEvent]:
        with self._lock:
            values = list(self._events.values())
            return values[-max(0, limit):]

    def audit(self, event_id: str | None = None) -> list[AuditEntry]:
        with self._lock:
            entries = self._audits if event_id is None else [entry for entry in self._audits if entry.event_id == event_id]
            return list(entries)

    def deduplicate(self, reason_code: str, source: str, window_seconds: float = 8.0) -> bool:
        """Return whether a matching active event already exists recently."""

        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            for event in self.active():
                if event.reason_code != reason_code or event.evidence.modalities is None:
                    continue
                age = now - event.created_at.timestamp()
                if age <= window_seconds and source in event.id:
                    return True
            return False

    def _record(self, event: SafetyEvent, action: str, actor: str) -> None:
        self._audits.append(
            AuditEntry(
                event_id=event.id,
                action=action,
                state=event.state,
                created_at=datetime.now(timezone.utc),
                actor=actor,
            )
        )

    def _trim(self) -> None:
        while len(self._events) > self.max_events:
            oldest = next(iter(self._events))
            del self._events[oldest]
