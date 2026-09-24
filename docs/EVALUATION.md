# Evaluation note

## What the published figures mean

The README reports a small set of engineering signals from the underlying research workspace. They are intentionally presented as a snapshot, not as a benchmark claim.

| Signal | Scope | Interpretation |
|---|---|---|
| **93.75% validation accuracy** | Pose-transformer branch, internal UR-Fall split | Best recorded validation point in that experiment |
| **<5 ms pose path** | GPU deployment path on an RTX 4060 Ti-class setup | Detector-side timing, excluding capture/network overhead |
| **<1 ms temporal head** | Lightweight temporal classifier | Head timing, excluding upstream feature extraction |
| **4 evidence streams** | Video, pose, audio, temporal context | Capability surface, not a guarantee that every deployment enables all streams |

## Reproducibility boundary

The public repository does not include the private dataset mirror, checkpoints, training recipe, or raw captured media. As a result, the figures above cannot be reproduced from this repository alone.

For a responsible technical review, request the matching evaluation package and record:

- dataset identity, split strategy, and subject separation;
- model revision and checkpoint hash;
- hardware, driver, CUDA, and batch configuration;
- end-to-end latency, not only model-kernel latency;
- false-positive rate, missed-event rate, and time-to-alert;
- performance under occlusion, lighting changes, audio absence, and camera dropout.

## Recommended acceptance matrix

| Dimension | Minimum review question |
|---|---|
| Detection | Does the event remain stable across adjacent frames? |
| Verification | Is there a quality or evidence reason code? |
| Timing | How long from first observable evidence to operator alert? |
| Robustness | What happens when a modality disappears? |
| Operations | Can an operator acknowledge, resolve, and audit the event? |
| Privacy | Are retention, access, and deletion policies explicit? |

No medical, emergency-response, or safety-critical claim should be inferred from the snapshot alone.
