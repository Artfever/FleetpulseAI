# P22-A3 history replay profile

**Run date:** 2026-11-20  
**Reproduction:** `.venv/Scripts/python.exe scripts/profile_p22_a3_replay.py`

## Scope and provenance

The profile compares the old row-selection/validation loop with a small optimized selector. Both paths use the same saved model, feature configuration, replay implementation, and selected history. The baseline selector reproduces the pre-change `isinstance(record, Mapping)` plus `_positive_integer(unit_id, ...)` checks. The optimized selector fast-paths only built-in `dict` rows and built-in `int` engine IDs, which are what the FD001 loader returns; all other mapping and numeric types still use the original validation checks.

| Input | Value |
| --- | --- |
| Dataset | NASA C-MAPSS FD001/train (simulated) |
| Dataset SHA-256 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| Replay identity | `FD001/train`, engine 4, selected cycle 164 |
| Replay cutoffs | 16, 33, 49, 66, 82, 98, 115, 131, 148, 164 |
| Rows scanned / selected | 20,631 / 189 |
| Model | `decision-tree-rul-v1` (`decision-tree-rul`) |
| Model SHA-256 | `2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018` |
| Manifest SHA-256 | `41f16e1397e1e2780133fd8a6aae8e8990d4af260055111b595c9831c8cb13b3` |
| Features | `past-only-v1`, 30-cycle window, ordered 24 settings/sensor channels |
| Runtime | Python 3.12; scikit-learn 1.9.1 |

The saved model was already loaded before the timed comparison. For uncached replay, each timing clears the replay-result cache, then runs the full ten-cutoff replay. For warm-cache timing, replay results are retained. Baseline and optimized runs alternate order for 30 uncached rounds and 100 warm rounds. The profile output also includes the cumulative cProfile breakdown.

## Results

| Full-fleet replay mode | Baseline median / p95 | Optimized median / p95 | Median reduction |
| --- | ---: | ---: | ---: |
| Cache cleared before each replay | 66.43 / 67.52 ms | 53.41 / 56.68 ms | 19.6% |
| Replay-result cache retained | 33.49 / 35.63 ms | 18.77 / 21.63 ms | 43.9% |

On every run, baseline and optimized results matched exactly for all ten cutoffs, including raw and displayed RUL, model name/version, observed cycle, input row count, and warning list. The raw RULs were:

| Cycle | Raw RUL (cycles) | Rows used |
| ---: | ---: | ---: |
| 16 | 181.30208333333334 | 16 |
| 33 | 117.1125 | 30 |
| 49 | 202.7296819787986 | 30 |
| 66 | 202.7296819787986 | 30 |
| 82 | 169.92432432432432 | 30 |
| 98 | 135.72246696035242 | 30 |
| 115 | 116.55555555555556 | 30 |
| 131 | 59.99331550802139 | 30 |
| 148 | 38.857142857142854 | 30 |
| 164 | 34.19607843137255 | 30 |

The profiled cold replay identified substantial cumulative time in validation and repeated `isinstance` calls while scanning and validating rows. After the selector fast path, the unchanged per-prefix `validate_engine_history` work is still visible in the profile; no broader validation changes were made because this task only changes the proven row-selection overhead.

## Limits

These are local timings on one machine and one simulated training history, not a product latency target or evidence of RUL accuracy. The model is the saved P06 development Decision Tree, seed 42, depth 8, minimum leaf size 5; no training was performed. The separate P12 final-tree bundle wrapper does not match the current predictor service manifest interface and was not used. No official-test labels were read, and the replay does not provide maintenance advice.
