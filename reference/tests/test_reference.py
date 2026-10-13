"""Standard-library tests for the public reference implementation."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(ROOT))

from calibration import ConfidenceCalibrator, binary_metrics, best_threshold  # noqa: E402
from config import StreamConfig  # noqa: E402
from event_store import EventStore  # noqa: E402
from models import EventState, Evidence, Modality, Observation, SafetyEvent, Severity, modalities  # noqa: E402
from policy import EventPolicy  # noqa: E402
from temporal_window import FrameEvidence, RobustSmoother, TemporalWindow, VoteAccumulator, rms  # noqa: E402


class WindowTests(unittest.TestCase):
    def test_window_is_bounded_and_causal(self) -> None:
        window = TemporalWindow(capacity=3)
        for index in range(5):
            self.assertTrue(window.append(FrameEvidence(index, 0.5 + index * 0.05, 0.9)))
        self.assertEqual(len(window), 3)
        self.assertEqual(window.values()[0].timestamp, 2)

    def test_window_rejects_low_quality(self) -> None:
        window = TemporalWindow(capacity=3, min_quality=0.5)
        self.assertFalse(window.append(FrameEvidence(0, 0.9, 0.2)))
        self.assertEqual(len(window), 0)

    def test_window_detects_slope(self) -> None:
        window = TemporalWindow(capacity=4)
        for index, value in enumerate((0.2, 0.3, 0.5, 0.8)):
            window.append(FrameEvidence(float(index), value, 0.9, "falling"))
        summary = window.summarize()
        assert summary is not None
        self.assertGreater(summary.confidence_slope, 0.1)
        self.assertEqual(summary.phase, "falling")

    def test_smoother_resists_spike(self) -> None:
        smoother = RobustSmoother(width=3, blend=1.0)
        values = [smoother.update(item) for item in (0.2, 0.2, 1.0, 0.2)]
        self.assertEqual(values[-1], 0.2)

    def test_votes_require_hysteresis(self) -> None:
        votes = VoteAccumulator(required=3, horizon=4)
        self.assertFalse(votes.push(True))
        self.assertFalse(votes.push(False))
        self.assertFalse(votes.push(True))
        self.assertTrue(votes.push(True))
        self.assertAlmostEqual(votes.ratio, 3 / 4)

    def test_rms(self) -> None:
        self.assertAlmostEqual(rms([3, 4]), 3.5355339, places=5)


class CalibrationTests(unittest.TestCase):
    def test_low_quality_reduces_confidence(self) -> None:
        calibrator = ConfidenceCalibrator()
        self.assertLess(calibrator.calibrate(0.9, 0.1), 0.9)

    def test_metrics_are_consistent(self) -> None:
        metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.f1, 1.0)

    def test_best_threshold_prefers_separation(self) -> None:
        threshold, metrics = best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertGreaterEqual(threshold, 0.2)
        self.assertEqual(metrics.f1, 1.0)


class PolicyAndStoreTests(unittest.TestCase):
    def _observation(self) -> Observation:
        return Observation(
            id="obs_1",
            observed_at=__import__("models").utc_now(),
            source="test",
            quality=0.9,
            available=modalities(Modality.POSE, Modality.TEMPORAL),
            contributing=modalities(Modality.POSE, Modality.TEMPORAL),
        )

    def test_policy_promotes_high_confidence(self) -> None:
        event = EventPolicy(score=lambda _: 0.95).evaluate(self._observation())
        assert event is not None
        self.assertEqual(event.state, EventState.ESCALATED)
        self.assertEqual(event.severity, Severity.HIGH)

    def test_policy_drops_low_quality(self) -> None:
        observation = self._observation()
        low = replace(observation, quality=0.1)
        self.assertIsNone(EventPolicy(score=lambda _: 0.99).evaluate(low))

    def test_store_audits_transition(self) -> None:
        event = EventPolicy(score=lambda _: 0.8).evaluate(self._observation())
        assert event is not None
        store = EventStore()
        store.create(event)
        store.transition(event.id, EventState.RESOLVED, actor="test")
        self.assertEqual(len(store.audit(event.id)), 2)
        self.assertEqual(store.active(), [])

    def test_event_rejects_invalid_transition(self) -> None:
        event = SafetyEvent("e", EventState.RESOLVED, Severity.INFO, "done", Evidence(0, 1, frozenset(), "done"))
        with self.assertRaises(ValueError):
            event.transition(EventState.OBSERVED)


class ConfigTests(unittest.TestCase):
    def test_config_defaults_validate(self) -> None:
        self.assertEqual(StreamConfig().validate().window_size, 32)

    def test_config_rejects_invalid_device(self) -> None:
        with self.assertRaises(ValueError):
            StreamConfig(device="quantum").validate()


if __name__ == "__main__":
    unittest.main()
