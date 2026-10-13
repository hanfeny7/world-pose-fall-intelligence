"""Tests for the extended public runtime modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(ROOT))

from fall_impact import BodyPart, FallImpactAnalyzer  # noqa: E402
from multimodal import AudioVisualFusion, ModalityEvidence, ReliabilityGate, aggregate_modalities  # noqa: E402
from pose_features import SkeletonSequenceBuffer, feature_vector, make_pose_frame, quality_report  # noqa: E402
from runtime import RuntimeConfig, RuntimeFrame, SentinelRuntime, make_static_provider  # noqa: E402


def pose_at(hip_y: float, shoulder_y: float = 0.35) -> np.ndarray:
    points = np.zeros((17, 2), dtype=np.float32)
    points[:, 0] = 0.5
    points[:, 1] = 0.5
    points[[5, 6], 1] = shoulder_y
    points[[11, 12], 1] = hip_y
    points[[15, 16], 1] = min(1.0, hip_y + 0.25)
    points[[9, 10], 1] = min(1.0, hip_y + 0.10)
    return points


class PoseFeatureTests(unittest.TestCase):
    def test_feature_vector_contains_confidence_and_velocity(self) -> None:
        frames = [make_pose_frame(pose_at(0.5 + index * 0.02), np.ones(17), timestamp=index) for index in range(4)]
        features = feature_vector(frames)
        self.assertEqual(features.shape, (4, 17, 5))
        self.assertGreater(float(np.abs(features[-1, :, 3:]).mean()), 0.0)

    def test_buffer_exports_quality(self) -> None:
        buffer = SkeletonSequenceBuffer(capacity=3)
        for index in range(3):
            ready = buffer.add(make_pose_frame(pose_at(0.5), np.ones(17), timestamp=index))
        self.assertTrue(ready)
        self.assertEqual(len(buffer.tensor()), 3)
        self.assertGreater(buffer.quality().score, 0.5)

    def test_buffer_rejects_time_regression(self) -> None:
        buffer = SkeletonSequenceBuffer(capacity=3)
        buffer.add(make_pose_frame(pose_at(0.5), timestamp=2.0))
        with self.assertRaises(ValueError):
            buffer.add(make_pose_frame(pose_at(0.5), timestamp=1.0))

    def test_quality_report_empty_is_safe(self) -> None:
        report = quality_report([])
        self.assertEqual(report.score, 0.0)


class ImpactTests(unittest.TestCase):
    def test_fast_drop_creates_impact(self) -> None:
        analyzer = FallImpactAnalyzer(velocity_threshold=0.4, impact_threshold=0.4)
        analyzer.update(pose_at(0.3), np.ones(17), frame_index=0)
        report = analyzer.update(pose_at(0.9), np.ones(17), frame_index=1)
        self.assertIsNotNone(report.impact)
        assert report.impact is not None
        self.assertIn(
            report.impact.body_part,
            {
                BodyPart.LEFT_HIP,
                BodyPart.RIGHT_HIP,
                BodyPart.LEFT_ANKLE,
                BodyPart.RIGHT_ANKLE,
                BodyPart.LEFT_KNEE,
                BodyPart.RIGHT_KNEE,
                BodyPart.LEFT_WRIST,
                BodyPart.RIGHT_WRIST,
            },
        )
        self.assertGreaterEqual(report.vertical_drop, 0.0)

    def test_slow_motion_is_not_critical(self) -> None:
        analyzer = FallImpactAnalyzer()
        for index in range(5):
            report = analyzer.update(pose_at(0.3 + index * 0.01), np.ones(17), frame_index=index)
        self.assertNotEqual(report.risk_level, "critical")

    def test_report_explanation_is_human_readable(self) -> None:
        analyzer = FallImpactAnalyzer()
        report = analyzer.update(pose_at(0.5), np.ones(17))
        self.assertIsInstance(report.explanation, str)
        self.assertTrue(report.explanation)


class FusionTests(unittest.TestCase):
    def test_visual_fallback_when_audio_missing(self) -> None:
        fusion = AudioVisualFusion()
        result = fusion.fuse(ModalityEvidence("visual", 0.84, 0.9))
        self.assertEqual(result.gate, 0.0)
        self.assertEqual(result.modalities, ("visual",))
        self.assertEqual(result.label, "FALL")

    def test_audio_quality_controls_gate(self) -> None:
        fusion = AudioVisualFusion(ReliabilityGate(minimum_quality=0.4))
        visual = ModalityEvidence("visual", 0.7, 0.9)
        weak_audio = ModalityEvidence("audio", 0.99, 0.1)
        strong_audio = ModalityEvidence("audio", 0.99, 0.9)
        self.assertEqual(fusion.fuse(visual, weak_audio).gate, 0.0)
        self.assertGreater(fusion.fuse(visual, strong_audio).gate, 0.0)

    def test_aggregate_modalities_is_redacted(self) -> None:
        result = aggregate_modalities([
            ModalityEvidence("visual", 0.8, 0.9),
            ModalityEvidence("audio", 0.6, 0.5),
        ])
        self.assertEqual(result["available"], 2.0)
        self.assertNotIn("source_path", result)


class RuntimeTests(unittest.TestCase):
    def test_vote_gate_emits_after_required_frames(self) -> None:
        runtime = SentinelRuntime(
            make_static_provider(0.95),
            config=RuntimeConfig(window_size=4, vote_horizon=4, vote_required=3, event_cooldown_seconds=10.0),
        )
        results = [runtime.process(RuntimeFrame(index, float(index), object())) for index in range(6)]
        emitted = [result for result in results if result.promoted]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(runtime.store.active()), 1)

    def test_cooldown_prevents_duplicate_events(self) -> None:
        runtime = SentinelRuntime(
            make_static_provider(0.95),
            config=RuntimeConfig(window_size=2, vote_horizon=2, vote_required=1, event_cooldown_seconds=10.0),
        )
        results = [runtime.process(RuntimeFrame(index, float(index), object())) for index in range(8)]
        self.assertLessEqual(sum(result.promoted for result in results), 1)

    def test_reset_allows_new_sequence(self) -> None:
        runtime = SentinelRuntime(
            make_static_provider(0.95),
            config=RuntimeConfig(window_size=2, vote_horizon=2, vote_required=1, event_cooldown_seconds=0.0),
        )
        runtime.process(RuntimeFrame(0, 0.0, object()))
        runtime.process(RuntimeFrame(1, 1.0, object()))
        runtime.reset()
        self.assertEqual(len(runtime.window), 0)
        self.assertEqual(runtime.votes.positive_count, 0)


if __name__ == "__main__":
    unittest.main()
