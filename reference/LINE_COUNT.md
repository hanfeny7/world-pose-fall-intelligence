# Public reference code line count

Counts are physical lines from the tracked reference tree. Python includes
inference, streaming, calibration, storage, fusion, impact analysis, runtime,
and tests; TypeScript includes the public client contract.

| Language / artifact | Lines |
|---|---:|
| Python source | 1854 |
| TypeScript source | 38 |
| **Executable source total** | **1892** |
| Markdown documentation | 46 |
| **Reference tree total** | **1938** |

## File breakdown

| File | Lines |
|---|---:|
| `python/calibration.py` | 80 |
| `python/config.py` | 42 |
| `python/demo.py` | 38 |
| `python/event_store.py` | 95 |
| `python/fall_impact.py` | 242 |
| `python/models.py` | 90 |
| `python/multimodal.py` | 141 |
| `python/pipeline.py` | 53 |
| `python/policy.py` | 76 |
| `python/pose_features.py` | 232 |
| `python/run_temporal_v5.py` | 52 |
| `python/runtime.py` | 117 |
| `python/temporal_v5.py` | 152 |
| `python/temporal_window.py` | 172 |
| `tests/test_reference.py` | 123 |
| `tests/test_extended_runtime.py` | 149 |
| `typescript/contracts.ts` | 38 |

The implementation is intentionally substantive rather than a skeleton: it
includes the Temporal-V5 causal residual expert, bounded streaming windows,
pose quality and velocity features, impact analysis, multimodal reliability
gating, a vote-gated runtime with event storage, configuration validation, a
synthetic CLI, and 28 unit tests.