# P23-A3 remote prediction and schedule parity

**Status:** Local references reproduced; remote value comparison remains incomplete.
**Task:** [P23-A3 #261](https://github.com/Artfever/FleetpulseAI/issues/261)
**Run date:** 2026-11-20

## Local reference bundle

The reference uses the saved development Decision Tree, `decision-tree-rul-v1`,
and the three packaged simulated FD001/train prefixes. No fitting or retraining
was performed.

- Model SHA-256: `2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`
- Manifest SHA-256: `41f16e1397e1e2780133fd8a6aae8e8990d4af260055111b595c9831c8cb13b3`
- Source file SHA-256: `953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`
- Feature schema: `past-only-v1`, up to 30 observed cycles, 96 ordered features.
- Prediction input histories: `data/demo/fd001_training_prefixes.txt`.

| Dataset | Engine | Observed cycle | Raw RUL (cycles) | Planner RUL (cycles) |
| --- | ---: | ---: | ---: | ---: |
| FD001/train | 4 | 164 | 34.19607843137255 | 34.19607843137255 |
| FD001/train | 5 | 58 | 158.16 | 158.16 |
| FD001/train | 12 | 7 | 184.35714285714286 | 184.35714285714286 |

The paired scheduler used one team, a two-cycle buffer, six-cycle soft horizon,
and illustrative duration and priority weight of one for every job. Both EDD
and seeded GA used the same scenario hash,
`3c58edea62bf88da3eb79b66e14779273e523a77cd7cc84b2f58b7560ac93049`. Each
scheduled engines 4, 5, and 12, was feasible, had zero weighted tardiness,
three-cycle makespan, and zero horizon overflow. These are scheduling
assumptions and a three-prefix wiring example, not a full-fleet result or
evidence that one scheduler is superior.

## Remote comparison status

Abdullah confirmed that the deployed app is consistent with the `dev` release
at `ffe8a4555ba278ce4dd061bc14abdcbc40239873`. The issue's earlier deployment
dependency #229 was closed on that confirmation. This is owner-reported release
identity; the exact hosted commit is not exposed by the hosting platform here.

The hosted UI outputs required for numeric parity were not observable from this
environment. An anonymous request to
`https://fleetpulseai.streamlit.app/` returned HTTP 303 to Streamlit sign-in.
Therefore the remote prediction values and remote EDD/GA outputs are recorded
as **NOT OBSERVED**, not assumed equal to the local references. A logged-in
screen inspection or copied values from the hosted app are still needed to
complete the comparison. The app being live and the deployment matching the
release do not by themselves prove these numeric outputs match.

## Reproduction

Commands, each PASS on this checkout:

```powershell
.venv/Scripts/python.exe scripts/check_p20_planner_boundary.py
.venv/Scripts/python.exe scripts/check_p20_a3_schedule_service.py
.venv/Scripts/python.exe scripts/check_p20_s3_paired_artifact.py
```

The paired-artifact check reports `artifact_round_trip: PASS` and reproduces
the model, manifest, source, and scenario hashes above. Full output and the
planner prediction values are also recorded in
[`docs/p20_prediction_planner_boundary.md`](p20_prediction_planner_boundary.md)
and [`docs/p20_s3_paired_comparison_artifact.md`](p20_s3_paired_comparison_artifact.md).
