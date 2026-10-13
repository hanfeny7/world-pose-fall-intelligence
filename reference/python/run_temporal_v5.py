"""Small CLI that exercises the public Temporal-V5 surrounding stack."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from calibration import ConfidenceCalibrator
from config import StreamConfig
from temporal_window import FrameEvidence, RobustSmoother, TemporalWindow, VoteAccumulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic Temporal-V5 stream simulation")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.72)
    return parser


def simulate(frames: int, threshold: float) -> dict[str, object]:
    if frames < 2:
        raise ValueError("frames must be >= 2")
    config = StreamConfig.from_env()
    window = TemporalWindow(config.window_size, config.min_quality)
    smoother = RobustSmoother()
    votes = VoteAccumulator(required=4, horizon=8)
    calibrator = ConfidenceCalibrator()

    for index in range(frames):
        raw = min(0.98, 0.40 + index * 0.055)
        confidence = calibrator.calibrate(raw, quality=0.92)
        smooth = smoother.update(confidence)
        votes.push(smooth >= threshold)
        window.append(FrameEvidence(index / config.fps, smooth, 0.92, "falling" if smooth > 0.7 else "normal"))

    summary = window.summarize()
    return {
        "config": {"source": config.source_name, "fps": config.fps, "window_size": config.window_size},
        "summary": asdict(summary) if summary else None,
        "vote_ratio": votes.ratio,
        "promoted": votes.positive_count >= votes.required,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(simulate(args.frames, args.threshold), indent=2))


if __name__ == "__main__":
    main()
