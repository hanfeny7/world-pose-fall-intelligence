"""Explainable fall-impact and body-contact analysis.

This module is intentionally model-free. It converts pose trajectories into
interpretable impact events, contact candidates, body angle, and a bounded
risk report. A learned Temporal-V5 score can be attached as an additional
signal without replacing these explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

import numpy as np


class BodyPart(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass(frozen=True, slots=True)
class ImpactEvent:
    frame_index: int
    body_part: BodyPart
    velocity: float
    confidence: float
    supported: bool = False

    @property
    def severity(self) -> str:
        speed = abs(self.velocity)
        if self.supported:
            return "guarded"
        if speed >= 2.8:
            return "critical"
        if speed >= 1.8:
            return "high"
        if speed >= 1.0:
            return "medium"
        return "low"


@dataclass(frozen=True, slots=True)
class FallAnalysisReport:
    frame_index: int
    impact: ImpactEvent | None
    body_angle_degrees: float
    vertical_drop: float
    support_detected: bool
    risk_level: str
    risk_score: float
    explanation: str
    evidence: dict[str, float] = field(default_factory=dict)


def _valid(point: np.ndarray, confidence: float) -> bool:
    return bool(confidence > 1e-5 and np.isfinite(point).all())


def _mean_point(points: np.ndarray, confidence: np.ndarray, indexes: Iterable[int]) -> np.ndarray | None:
    selected = []
    for index in indexes:
        if index < len(points) and _valid(points[index], confidence[index]):
            selected.append(points[index])
    return np.mean(selected, axis=0) if selected else None


class FallImpactAnalyzer:
    """Stateful analyzer for one tracked person."""

    def __init__(
        self,
        history_length: int = 45,
        velocity_threshold: float = 1.0,
        impact_threshold: float = 1.8,
    ) -> None:
        if history_length < 3:
            raise ValueError("history_length must be at least 3")
        self.history_length = history_length
        self.velocity_threshold = float(velocity_threshold)
        self.impact_threshold = float(impact_threshold)
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._confidences: list[np.ndarray] = []
        self._reports: list[FallAnalysisReport] = []
        self._frame_index = -1

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray | None = None,
        frame_index: int | None = None,
    ) -> FallAnalysisReport:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (17, 2):
            raise ValueError("keypoints must have shape [17, 2]")
        scores = np.ones(17, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32).reshape(17)
        scores = np.clip(scores, 0.0, 1.0)
        self._frame_index = self._frame_index + 1 if frame_index is None else int(frame_index)
        self._frames.append(points.copy())
        self._confidences.append(scores.copy())
        self._frames = self._frames[-self.history_length:]
        self._confidences = self._confidences[-self.history_length:]
        report = self._report()
        self._reports.append(report)
        self._reports = self._reports[-self.history_length:]
        return report

    def _report(self) -> FallAnalysisReport:
        current = self._frames[-1]
        scores = self._confidences[-1]
        hip = _mean_point(current, scores, (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP))
        shoulder = _mean_point(current, scores, (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER))
        ankle = _mean_point(current, scores, (BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE))
        angle = self._body_angle(shoulder, hip)
        drop = self._vertical_drop()
        support, support_part = self._support_signal(current, scores)
        impact = self._impact_event(support_part, support)
        score = self._risk_score(drop, angle, impact, support)
        risk = self._risk_level(score)
        explanation = self._explanation(impact, drop, angle, support, risk)
        evidence = {
            "drop": float(drop),
            "angle": float(angle),
            "pose_quality": float(scores.mean()),
            "impact_velocity": float(abs(impact.velocity) if impact else 0.0),
        }
        return FallAnalysisReport(
            frame_index=self._frame_index,
            impact=impact,
            body_angle_degrees=angle,
            vertical_drop=drop,
            support_detected=support,
            risk_level=risk,
            risk_score=score,
            explanation=explanation,
            evidence=evidence,
        )

    def _vertical_drop(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        before = self._frames[-2]
        after = self._frames[-1]
        before_score = self._confidences[-2]
        after_score = self._confidences[-1]
        before_hip = _mean_point(before, before_score, (11, 12))
        after_hip = _mean_point(after, after_score, (11, 12))
        if before_hip is None or after_hip is None:
            return 0.0
        return float(after_hip[1] - before_hip[1])

    @staticmethod
    def _body_angle(shoulder: np.ndarray | None, hip: np.ndarray | None) -> float:
        if shoulder is None or hip is None:
            return 0.0
        vector = shoulder - hip
        return float(np.degrees(np.arctan2(abs(vector[0]), abs(vector[1]) + 1e-6)))

    def _support_signal(self, points: np.ndarray, scores: np.ndarray) -> tuple[bool, BodyPart | None]:
        hip = _mean_point(points, scores, (11, 12))
        if hip is None:
            return False, None
        candidates = [(BodyPart.LEFT_WRIST, 9), (BodyPart.RIGHT_WRIST, 10), (BodyPart.LEFT_KNEE, 13), (BodyPart.RIGHT_KNEE, 14)]
        for part, index in candidates:
            if _valid(points[index], scores[index]) and points[index, 1] >= hip[1] - 0.05:
                return True, part
        return False, None

    def _impact_event(self, support_part: BodyPart | None, supported: bool) -> ImpactEvent | None:
        if len(self._frames) < 2:
            return None
        previous = self._frames[-2]
        current = self._frames[-1]
        previous_conf = self._confidences[-2]
        current_conf = self._confidences[-1]
        velocities = current[:, 1] - previous[:, 1]
        candidates = [
            BodyPart.NOSE, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER,
            BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP, BodyPart.LEFT_KNEE,
            BodyPart.RIGHT_KNEE, BodyPart.LEFT_WRIST, BodyPart.RIGHT_WRIST,
        ]
        valid = [part for part in candidates if _valid(current[part], current_conf[part]) and _valid(previous[part], previous_conf[part])]
        if not valid:
            return None
        part = max(valid, key=lambda item: abs(float(velocities[item])))
        speed = float(velocities[part])
        if abs(speed) < self.velocity_threshold:
            return None
        return ImpactEvent(self._frame_index, part, speed, float(current_conf[part]), supported and part == support_part)

    def _risk_score(self, drop: float, angle: float, impact: ImpactEvent | None, supported: bool) -> float:
        drop_score = np.clip(drop / 0.35, 0.0, 1.0)
        angle_score = np.clip((angle - 35.0) / 55.0, 0.0, 1.0)
        impact_score = np.clip(abs(impact.velocity) / 3.0, 0.0, 1.0) if impact else 0.0
        score = 0.34 * drop_score + 0.32 * angle_score + 0.34 * impact_score
        if supported:
            score *= 0.62
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.50:
            return "high"
        if score >= 0.25:
            return "guarded"
        return "low"

    @staticmethod
    def _explanation(impact: ImpactEvent | None, drop: float, angle: float, supported: bool, risk: str) -> str:
        if impact is None:
            return f"No decisive impact; posture risk is {risk}."
        support_text = " with possible active support" if supported else " without detected support"
        return f"{impact.body_part.name.lower()} shows vertical velocity {impact.velocity:.2f}{support_text}; drop={drop:.2f}, angle={angle:.1f}°."

    @property
    def last_report(self) -> FallAnalysisReport | None:
        return self._reports[-1] if self._reports else None

    def reports(self) -> tuple[FallAnalysisReport, ...]:
        return tuple(self._reports)
