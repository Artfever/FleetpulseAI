# P21-S1: Scientific label audit across app screens

P21-S1 (#91) checks that users can distinguish observed cycles, development
split membership, saved model/version, simulated inputs, and error/planner
metrics. The audit reviewed the Home view and all five Streamlit pages, their
current UI copy, and the relevant AppTest coverage.

## Findings and corrections

| Screen | Exact prior wording or gap | Evidence and correction |
| --- | --- | --- |
| Home | “Trained models and evaluation results are not available yet.” | Availability is deployment-dependent: this workspace has local saved artifacts while clean/public checkouts may not. The message now says verified bundles/reports appear only when matching artifacts are installed and points users to Model Laboratory. |
| Engine Explorer | “Known training RUL ? analysis only” | The question mark obscured the label. It now reads “Known training RUL (analysis only).” The adjacent text says these labels use the full training lifetime/future failure cycle and are not model predictions or predictor inputs. |
| Fleet Overview | Prediction rows showed dataset, cycles, model and version but no fitting/validation membership. | Added a `Fit membership` column using the saved split manifest: “Seen during fitting,” “Not seen during fitting (development validation engine),” or an explicit unknown label. A missing or invalid split never becomes “unseen.” |
| RUL Prediction — uploaded CSV | Upload predictions had no statement about fitting membership. | Added “Engine fit membership: Not verified for the uploaded history.” It does not infer that an uploaded engine is unseen from the `user/upload` partition name. |

## Screen-by-screen review

- **Home:** identifies the project as simulated turbofan maintenance planning;
  artifact availability is conditional and described accurately.
- **Fleet Overview:** data source identifies `FD001/train`; output rows identify
  engine, observed cycle, displayed/raw RUL in cycles, fit membership, model and
  version. The threshold is labeled as a display filter, not an alarm. Invalid
  or unavailable membership remains unknown.
- **Engine Explorer:** shows observed training records sorted by cycle;
  ground-truth training RUL is separated and explicitly uses future lifetime;
  feature preview reports its observed cutoff, 30-cycle window/config version,
  feature source units and per-cycle slope units.
- **RUL Prediction:** separates development candidate from frozen final model;
  reports observed cutoff/window, raw cycles and display estimate separately;
  labels fitting vs development-validation status for local train histories,
  marks uploaded fit membership unverified, and identifies the synthetic
  example as artificial. Explanation values are in cycles and described as
  model attribution rather than cause; raw model range is labeled disagreement,
  not confidence.
- **Maintenance Planner:** selected fleet retains dataset/engine/model/version
  and raw/display RUL. Planner settings are labeled assumptions; completion,
  tardiness, weighted cycles and soft-horizon overflow are defined in relative
  cycles and distinguished from model-estimate metrics.
- **Model Laboratory:** separates candidate selection, held-out development
  validation, and the FD001/test reporting section. Tables specify engine
  samples and MAE/RMSE cycles; MSE curves use cycles squared and distinguish
  cycle-level fitting rows from engine-prefix validation samples. Captions say
  these are error measures, not accuracy percentages or future guarantees.

## Actual saved-model output and provenance

The local saved Decision Tree manifest reports `decision-tree-rul-v1`, seed 42,
`max_depth=8`, `min_samples_leaf=5`, 16,738 fitting rows from 80 fitting
engines, and `past-only-v1` with a 30-cycle window and 96 ordered features.
Using the local bundle with packaged simulated FD001/train engine 4 through
cycle 164 returned raw RUL `34.19607843137255` cycles. The saved split marks
engine 4 as a development-validation engine, so this example is held out from
fitting but is not official-test performance. The same output is exercised by
the existing built-in workflow AppTest. No model was retrained or modified.

This point estimate from one simulated training prefix does not establish
fleet-wide accuracy, future performance, a physical fault, or maintenance need.
The P13 Model Laboratory's `FD001/test` wording follows the separate
checksum-verified final-report contract; its nearby copy identifies it as a
held-out dataset split used for final reporting only, not for fitting or model
selection.

## Checks and review status

- `.venv/Scripts/python.exe -m pytest -q tests/test_bootstrap.py` PASS (3 passed in 5.58s). The initial full-suite run caught a stale expectation for the old Home empty-state copy; the assertion now checks conditional artifact-availability wording.
- `.venv/Scripts/python.exe -m pytest -q` PASS (687 passed in 158.42s).
- `.venv/Scripts/python.exe -m ruff check .` PASS (`All checks passed!`).
- `.venv/Scripts/python.exe -m streamlit run app/Home.py --server.port 18604 --server.headless true` started; `Invoke-WebRequest http://localhost:18604/_stcore/health` returned `ok` (HTTP 200); server stopped afterward.
- `git diff --check` PASS.
- The issue's required teammate inspection/discussion of the audit and actual output is pending. Do not close #91 until that review is confirmed and recorded.
