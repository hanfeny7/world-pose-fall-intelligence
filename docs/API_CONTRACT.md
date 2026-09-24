# Public API contract

The examples in `public/contracts/` describe the vocabulary used at the system boundary. They are deliberately small and redacted: there are no credentials, device identifiers, internal URLs, model names, or proprietary feature vectors.

## Observation envelope

An observation represents evidence at a point in time. It can contain pose geometry, audio availability, quality, and a model revision without exposing the model internals.

See [`observation.json`](../public/contracts/observation.json).

## Event envelope

An event represents a policy decision over one or more observations. It includes lifecycle state, severity, contributing modalities, and a human-readable reason code.

See [`event.json`](../public/contracts/event.json).

## Integration rules

- Timestamps are ISO 8601 UTC.
- IDs are opaque and should not contain names, phone numbers, or camera addresses.
- `evidence.modalities` must state which modalities were available and which contributed.
- A client must tolerate new enum values and unknown optional fields.
- Event payloads must not contain raw media by default; use an access-controlled evidence pointer.

## Example health response

```json
{
  "status": "ok",
  "service": "inference-gateway",
  "revision": "public-contract-v1",
  "capabilities": ["pose", "temporal", "event-policy"]
}
```
