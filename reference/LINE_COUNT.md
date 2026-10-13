# Public reference code line count

This count covers physical lines in tracked Python and TypeScript source files under `reference/`. Blank lines and comments are included because they are part of the readable source artifact. Markdown documentation is reported separately.

| Language / artifact | Lines |
|---|---:|
| Python source | 766 |
| TypeScript source | 35 |
| **Executable source total** | **801** |
| Markdown documentation | 48 |
| **Reference tree total** | **849** |

## File breakdown

| File | Lines |
|---|---:|
| `python/calibration.py` | 62 |
| `python/config.py` | 35 |
| `python/demo.py` | 26 |
| `python/event_store.py` | 77 |
| `python/models.py` | 69 |
| `python/pipeline.py` | 39 |
| `python/policy.py` | 59 |
| `python/run_temporal_v5.py` | 39 |
| `python/temporal_v5.py` | 129 |
| `python/temporal_window.py` | 134 |
| `tests/test_reference.py` | 97 |
| `typescript/contracts.ts` | 35 |

The source is intentionally substantive: the additional lines implement bounded streaming windows, robust smoothing, confidence calibration, event storage, configuration validation, a CLI simulation, and 15 unit tests.
