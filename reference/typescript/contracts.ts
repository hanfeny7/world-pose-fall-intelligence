export type Modality = "video" | "pose" | "audio" | "temporal";
export type EventState =
  | "observed"
  | "verified"
  | "escalated"
  | "acknowledged"
  | "resolved";
export type Severity = "info" | "guarded" | "high" | "critical";

export interface Observation {
  id: string;
  observed_at: string;
  source: string;
  quality: number;
  modalities: {
    available: Modality[];
    contributing: Modality[];
  };
  model_revision?: string;
}

export interface SafetyEvent {
  id: string;
  state: EventState;
  severity: Severity;
  reason_code: string;
  created_at: string;
  evidence: {
    confidence: number;
    quality: number;
    modalities: Modality[];
    pointer?: string | null;
  };
}

export function isActive(event: SafetyEvent): boolean {
  return event.state !== "resolved";
}
