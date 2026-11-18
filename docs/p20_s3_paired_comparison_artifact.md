# P20-S3 paired comparison artifact

The selected-fleet flow now captures provenance with the chosen prediction snapshot. After the user submits planning assumptions, the planner runs EDD and the seeded Genetic Algorithm on the same immutable scenario and offers a downloadable JSON artifact. The artifact is generated at submission time; changing controls alone does not overwrite the saved comparison.

## Artifact contract

Schema version: `p20-paired-schedule-comparison-v1`.

The artifact contains:

- Source name, dataset partition, source path, source record count, canonical-record SHA-256, and source-file SHA-256 when a source file exists.
- Model name/version, model artifact filename and SHA-256, manifest file and payload SHA-256, feature config/schema, training seed/parameters, and scikit-learn version.
- Each selected engine's dataset-qualified identity, observed cycle, raw RUL, planner RUL, number of feature rows, and prediction warnings.
- The exact shared scenario hash, assumptions, team count, buffer, horizon, and per-engine RUL/duration/priority inputs.
- Both checked schedule reports, feasibility, weighted tardiness, late jobs, makespan, overflow, and measured runtime; GA seed/search settings are included as well.

True RUL and validation labels are not inputs to this artifact builder. Raw predictions remain intact; planner RUL follows the existing `max(0, raw RUL)` rule. A build fails for missing provenance, failed predictions, mismatched engine sets, mixed model versions, inconsistent planner RUL, or incomplete manifest feature details.

## Verification

Synthetic contract tests use the Appendix B jobs and assert EDD weighted tardiness 6 and GA weighted tardiness 2, JSON round-trip, omission of labels, and rejection of missing provenance. The Streamlit interaction test saves a selected fleet, checks captured provenance, builds the comparison, and confirms the download control is present.

The saved-model integration check is:

```powershell
.venv/Scripts/python.exe scripts/check_p20_s3_paired_artifact.py
```

It uses only the packaged three-engine FD001/train prefix subset and the existing development Decision Tree bundle; it does not train or score the model. The outputs demonstrate artifact provenance and matching scenario hashes. One team, duration/weight 1, buffer 2, and horizon 6 are illustrative simulation assumptions. They are not measurements or operational recommendations. This small wiring example is not a full-fleet result, and its schedule scores are not evidence of prediction quality or schedule superiority.
