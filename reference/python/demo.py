"""Run a synthetic end-to-end public architecture example."""

from dataclasses import dataclass

from models import Modality
from pipeline import InferencePipeline
from policy import EventPolicy, summarize


@dataclass(frozen=True, slots=True)
class DemoPoseProvider:
    modality: Modality = Modality.POSE

    def observe(self, frame: object) -> dict[str, float]:
        del frame
        return {"quality": 0.94, "motion_energy": 0.81}


@dataclass(frozen=True, slots=True)
class DemoTemporalProvider:
    modality: Modality = Modality.TEMPORAL

    def observe(self, frame: object) -> dict[str, float]:
        del frame
        return {"temporal_consistency": 0.91}


def main() -> None:
    pipeline = InferencePipeline(
        providers=[DemoPoseProvider(), DemoTemporalProvider()],
        policy=EventPolicy(score=lambda observation: observation.pose_summary["motion_energy"]),
    )
    result = pipeline.process(object(), source="demo-camera", observation_id="obs_demo_001")
    print(summarize([result.event] if result.event else []))


if __name__ == "__main__":
    main()
