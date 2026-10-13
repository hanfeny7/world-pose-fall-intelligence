"""Public pose-feature pipeline used by the Temporal-V5 reference runtime.

The production detector can be swapped behind this module. The feature
contract is deliberately detector-agnostic: COCO-style keypoints, visibility,
bounding boxes, and optional frame timestamps become a stable numerical
sequence for a temporal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Normalized pose observation for one video frame."""

    keypoints: np.ndarray
    confidence: np.ndarray
    bbox: np.ndarray | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        if keypoints.shape != (17, 2):
            raise ValueError("keypoints must have shape [17, 2]")
        if confidence.shape != (17,):
            raise ValueError("confidence must have shape [17]")
        if self.bbox is not None and np.asarray(self.bbox).shape != (4,):
            raise ValueError("bbox must have shape [4] as x, y, width, height")


@dataclass(frozen=True, slots=True)
class PoseQuality:
    visible_ratio: float
    mean_confidence: float
    motion_energy: float
    bbox_valid: float
    stable: bool

    @property
    def score(self) -> float:
        return float(np.clip(
            0.45 * self.visible_ratio
            + 0.35 * self.mean_confidence
            + 0.10 * self.bbox_valid
            + 0.10 * (1.0 if self.stable else 0.0),
            0.0,
            1.0,
        ))


def _as_keypoints(value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("keypoints require a [N, >=2] array")
    output = np.zeros((17, 2), dtype=np.float32)
    output[: min(17, array.shape[0])] = array[:17, :2]
    return output


def _as_confidence(value: np.ndarray | Sequence[float] | None, count: int) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    output = np.zeros(17, dtype=np.float32)
    output[: min(17, array.size)] = array[:17]
    return np.clip(output, 0.0, 1.0)


def make_pose_frame(
    keypoints: np.ndarray | Sequence[Sequence[float]],
    confidence: np.ndarray | Sequence[float] | None = None,
    bbox: np.ndarray | Sequence[float] | None = None,
    timestamp: float | None = None,
) -> PoseFrame:
    points = _as_keypoints(keypoints)
    scores = _as_confidence(confidence, points.shape[0])
    box = None if bbox is None else np.asarray(bbox, dtype=np.float32).reshape(4)
    return PoseFrame(points, scores, box, timestamp)


def normalize_keypoints(frame: PoseFrame, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Center and scale keypoints around the torso, preserving visibility."""

    points = frame.keypoints.astype(np.float32).copy()
    if image_size is not None:
        width, height = image_size
        points[:, 0] /= max(float(width), 1.0)
        points[:, 1] /= max(float(height), 1.0)
    shoulders = points[[5, 6]].mean(axis=0)
    hips = points[[11, 12]].mean(axis=0)
    center = (shoulders + hips) * 0.5
    torso = float(np.linalg.norm(shoulders - hips))
    if torso < 1e-5:
        torso = float(np.ptp(points[:, 1]) or 1.0)
    return (points - center) / torso


def interpolate_missing(sequence: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Linearly fill missing joints without altering observed coordinates."""

    values = np.asarray(sequence, dtype=np.float32).copy()
    scores = np.asarray(confidence, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape [T, 17, 2]")
    for joint in range(17):
        valid = scores[:, joint] > 1e-5
        if valid.sum() == 0:
            continue
        indexes = np.arange(values.shape[0])
        for axis in range(2):
            values[:, joint, axis] = np.interp(indexes, indexes[valid], values[valid, joint, axis])
    return values


def velocity(sequence: np.ndarray, timestamps: Sequence[float] | None = None) -> np.ndarray:
    """Compute causal frame-to-frame velocity for a keypoint sequence."""

    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (17, 2):
        raise ValueError("sequence must have shape [T, 17, 2]")
    result = np.zeros_like(values)
    if values.shape[0] < 2:
        return result
    deltas = values[1:] - values[:-1]
    if timestamps is None:
        result[1:] = deltas
    else:
        times = np.asarray(timestamps, dtype=np.float32)
        if times.shape != (values.shape[0],):
            raise ValueError("timestamps must have shape [T]")
        dt = np.maximum(np.diff(times), 1e-4).reshape(-1, 1, 1)
        result[1:] = deltas / dt
    return result


def acceleration(sequence: np.ndarray, timestamps: Sequence[float] | None = None) -> np.ndarray:
    speeds = velocity(sequence, timestamps)
    return velocity(speeds, timestamps)


def feature_vector(
    frames: Sequence[PoseFrame],
    include_velocity: bool = True,
    include_acceleration: bool = False,
) -> np.ndarray:
    """Create a compact temporal feature tensor with deterministic ordering."""

    if not frames:
        return np.zeros((0, 17 * 2), dtype=np.float32)
    normalized = np.stack([normalize_keypoints(frame) for frame in frames])
    confidence = np.stack([frame.confidence for frame in frames])
    points = interpolate_missing(normalized, confidence)
    features = [points, confidence[..., None]]
    times = [frame.timestamp for frame in frames]
    timestamps = None if any(value is None for value in times) else [float(value) for value in times]
    if include_velocity:
        features.append(velocity(points, timestamps))
    if include_acceleration:
        features.append(acceleration(points, timestamps))
    return np.concatenate(features, axis=2).astype(np.float32)


def quality_report(frames: Sequence[PoseFrame]) -> PoseQuality:
    if not frames:
        return PoseQuality(0.0, 0.0, 0.0, 0.0, False)
    confidence = np.stack([frame.confidence for frame in frames])
    visible = confidence > 1e-5
    points = np.stack([normalize_keypoints(frame) for frame in frames])
    motion = np.diff(points, axis=0)
    motion_energy = float(np.mean(np.linalg.norm(motion, axis=-1))) if len(frames) > 1 else 0.0
    bbox_valid = float(np.mean([
        frame.bbox is not None and frame.bbox[2] > 0 and frame.bbox[3] > 0 for frame in frames
    ]))
    return PoseQuality(
        visible_ratio=float(visible.mean()),
        mean_confidence=float(confidence.mean()),
        motion_energy=motion_energy,
        bbox_valid=bbox_valid,
        stable=motion_energy < 0.35,
    )


class SkeletonSequenceBuffer:
    """Bounded frame buffer with a stable model-ready export."""

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self.capacity = capacity
        self._frames: list[PoseFrame] = []

    def add(self, frame: PoseFrame) -> bool:
        if self._frames and frame.timestamp is not None and self._frames[-1].timestamp is not None:
            if frame.timestamp < self._frames[-1].timestamp:
                raise ValueError("timestamps must be monotonic")
        self._frames.append(frame)
        if len(self._frames) > self.capacity:
            self._frames.pop(0)
        return self.ready

    @property
    def ready(self) -> bool:
        return len(self._frames) >= self.capacity

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> tuple[PoseFrame, ...]:
        return tuple(self._frames)

    def tensor(self, **kwargs) -> np.ndarray:
        return feature_vector(self._frames, **kwargs)

    def quality(self) -> PoseQuality:
        return quality_report(self._frames)

    def reset(self) -> None:
        self._frames.clear()
