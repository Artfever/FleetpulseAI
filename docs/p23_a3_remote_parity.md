# P23-A3 remote prediction and schedule parity

**Status:** Remote prediction and paired planner values inspected in the signed-in public app on 2026-11-21; numeric parity matches the local references. Artifact hashes and hosted revision are not exposed in the app.
**Task:** [P23-A3 #261](https://github.com/Artfever/FleetpulseAI/issues/261)
**Run date:** 2026-11-20 local reference; 2026-11-21 remote inspection

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

The signed-in Chrome tab at `https://fleetpulseai.streamlit.app/` was inspected
on 2026-11-21. The History source was set to “Packaged FD001/train parity
subset”; the visible dataset notice identifies the three simulated training
prefixes and their cutoffs. The selected model was the development
`decision-tree-rul-v1`.

| Dataset | Engine | Observed cycle | Remote raw RUL (cycles) | Local raw RUL (cycles) | Match |
| --- | ---: | ---: | ---: | ---: | --- |
| FD001/train | 4 | 164 | 34.19607843137255 | 34.19607843137255 | Yes |
| FD001/train | 5 | 58 | 158.16 | 158.16 | Yes |
| FD001/train | 12 | 7 | 184.35714285714286 | 184.35714285714286 | Yes |

The remote Fleet Overview showed the same model/version, observed cycles,
30-row feature window for engines 4 and 5, seven rows and `short_window` for
engine 12. Its displayed fit membership was “Not seen during fitting
(development validation engine)” for all three.

The selected-fleet planner loaded those same three records. With one team,
buffer 2 cycles, soft horizon 6 cycles, and assumed duration/priority weight 1
per engine, both remote schedules were feasible and matched the local metrics:
order 4, 5, 12; weighted and total tardiness 0; makespan 3 cycles; zero late
jobs; zero horizon overflow. Remote EDD finishes were 1/2/3 cycles with due
completions 32.19607843137255/156.16/182.35714285714286. Remote GA used seed
42, population 16, 20 generations, and also finished 4/5/12 at cycles 1/2/3.
This establishes numeric parity for this three-prefix example. The live page
did not expose its scenario hash, so local and remote hashes were not compared.
The duration/weight and planner limits remain illustrative assumptions; this
is not evidence about a full fleet, maintenance readiness, or scheduler
superiority.

The app did not expose a hosted commit SHA or artifact hashes. The owner had
previously reported the deployment consistent with dev revision
`ffe8a4555ba278ce4dd061bc14abdcbc40239873`; that statement is separate from
this browser observation.

Related remote checks during the same session:

- The RUL Prediction page returned all five raw frozen-model estimates for
  FD001/train engine 4 through cycle 164. Versions and raw outputs matched the
  existing local reference: Decision Tree 29.794117647058822, Random Forest
  28.02408950947574, XGBoost 26.14615821838379, constant baseline
  107.80786195530997, and Dense MLP 17.24176597595215 cycles. Displayed raw
  spread was 90.57 cycles, consistent with the local 90.566096-cycle
  max-minus-min result. The app correctly labels this as model disagreement,
  not uncertainty or confidence.
- The final Decision Tree feature explanation did not run. The app displayed
  `FD001 training histories are unavailable` for
  `<deployment checkout>/data/raw/train_FD001.txt`. Thus P23-S2’s explanation
  portion is blocked by a missing hosted dataset file even though the model
  comparison portion works.
- The live Model Laboratory showed P22-S1/P22-S2/P22-S3 study sections, but
  its “Official held-out test results (FD001/test)” section explicitly said
  verified results are unavailable in the deployment. P23-S1 cannot claim
  that the deployment’s final official-result view points to the intended
  manifest based on this check alone.
- The live RUL Prediction page retained `bad_history.csv` (15 bytes) and showed
  an `invalid_header` rejection listing the missing 26 required raw-history
  columns and unexpected `bad, header`. This confirms the visible malformed
  header error path. No new file was uploaded in this audit. The separate
  screenshot supplied by the owner also shows the same error. Missing-file
  behavior was not newly exercised in a clean session.

Therefore P23-A3’s three prediction values and selected-fleet scheduler values
are observed and numerically match local references. The page’s missing
scenario hash and unavailable hosted training history remain explicit limits;
they do not invalidate the numeric values observed above.

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

## Later P23-S1 remote update

After PR #521 added the checksum-sealed P13 display report to `dev`, Abdullah supplied a fresh signed-in screenshot of the Model Laboratory checked at 2026-11-21 12:54:58 AM PKT. The dedicated official-results panel was visible, and all five full-precision values matched the local report. The hosted commit was not displayed. See `docs/p23_s1_metric_source_audit.md` for exact rows, source checksum, and evidence limits.
