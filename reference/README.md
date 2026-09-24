# Public reference implementation

This directory is the small amount of code intentionally published with the system profile. It shows how the public architecture fits together without exposing the private model stack.

## Included

- typed observation and event envelopes;
- provider interfaces for pose, temporal, and audio evidence;
- the actual Temporal-V5 causal residual expert (`python/temporal_v5.py`);
- a reviewable event-policy state machine;
- a dependency-free demo that emits one verified event;
- TypeScript types for a console or mobile client.

## Withheld by design

The following are represented by interfaces only: feature extraction, world-model tokenization, multimodal fusion weights, learned thresholds, checkpoint loading, training code, and production transport adapters.

Run the dependency-free event demo from this directory:

```powershell
python .\python\demo.py
```

The output is synthetic and contains no private media or credentials.

The Temporal-V5 module uses PyTorch and is intentionally published as an
inference-side component. It expects projected current/future tokens from the
private frozen world-model producer; the producer, checkpoints, training
losses, and final alarm head are not included here.
