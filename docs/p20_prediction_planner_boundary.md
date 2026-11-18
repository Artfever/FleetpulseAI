# P20-S1 saved prediction to planner boundary check

**Run date:** 2026-11-18
**Command:** `.venv/Scripts/python.exe scripts/check_p20_planner_boundary.py`
**Result:** PASS; the script loaded the saved Fleet Overview model, generated predictions for the packaged FD001/train prefixes, built one `SchedulingScenario`, and verified each planner RUL and due cycle against the actual raw estimate and two-cycle buffer.

## Model and data provenance

- Fleet Overview currently calls `predict_fleet` with `artifacts/development/decision_tree_validation.json`. This is the saved development Decision Tree, not a P12 final bundle.
- Model: `decision-tree-rul`, version `decision-tree-rul-v1`, scikit-learn 1.9.1.
- Model file SHA-256: `2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`.
- Manifest SHA-256: `41f16e1397e1e2780133fd8a6aae8e8990d4af260055111b595c9831c8cb13b3`.
- Training manifest: seed 42; `max_depth=8`, `min_samples_leaf=5`; 80 fitting engines and 16,738 fitting rows.
- Feature config: `past-only-v1`, up to 30 cycles, 24 channels and 96 ordered features.
- Input: `data/demo/fd001_training_prefixes.txt`, a deliberately small set of three FD001/train prefixes. It is not a full-fleet or held-out performance result.

## Observed values

The planner uses the same nonnegative, unrounded RUL value as the raw result for these three positive predictions. The due completion cycle is `max(0, planner RUL - buffer)`. Scenario duration, priority, team count, buffer and horizon are assumptions for this simulation; FD001 does not provide maintenance durations or priority weights.

| Partition | Engine | Last observed cycle | Raw RUL (cycles) | Planner RUL (cycles) | Due completion (cycles) | Assumed duration | Assumed weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001/train | 4 | 164 | 34.19607843137255 | 34.19607843137255 | 32.19607843137255 | 1 | 1 |
| FD001/train | 5 | 58 | 158.16 | 158.16 | 156.16 | 1 | 1 |
| FD001/train | 12 | 7 | 184.35714285714286 | 184.35714285714286 | 182.35714285714286 | 1 | 1 |

Shared scenario settings: one team, two-cycle buffer, six-cycle soft horizon. The one-cycle durations and unit weights are explicit positive synthetic defaults. No EDD/GA schedule comparison or maintenance-effect claim is part of this check; P20-A3 covers the paired scheduler dispatch.

## Limits and review

This verifies data routing and arithmetic at the planner boundary. It does not validate maintenance policy, estimate physical failure risk, compare model quality, or use the final frozen model registry. The manifest and model are local ignored artifacts and are intentionally not committed. Shaheer or Taha should inspect the output values and provenance against P20-S1 before issue closure.
