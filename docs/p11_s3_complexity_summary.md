# P11-S3: does added model complexity help?

## Observed validation comparison

This summary compares existing P06/P11 artifacts; it performs no additional
training. All six rows below use the same 20 saved FD001/train validation
engine/cutoff pairs and identical scoring targets. The P11 runs all used the
verified source SHA and saved split manifest. The mean baseline predicts one
constant fitted-label mean for every engine.

| Candidate | Validation MAE (cycles) | Validation RMSE (cycles) |
| --- | ---: | ---: |
| Mean constant baseline | 65.728002 | 74.258987 |
| Decision Tree, 30-cycle window | 22.494270 | 29.449574 |
| Random Forest, depth 16 | 18.360242 | 24.563262 |
| XGBoost, depth 4 | 15.108697 | **21.092113** |
| MLP, hidden `(64, 32)` | 15.511258 | 22.937993 |
| MLP, hidden `(128, 64)` | **15.080604** | 22.066152 |

The added model complexity clearly reduced error on this development split
relative to the constant baseline. Among the P11 candidates shown, the larger
MLP had the lowest MAE, while XGBoost had the lowest RMSE. Their MAE difference
is only about 0.028 cycles, and their RMSE differs by about 0.974 cycles; this
comparison does not establish one universally best model. The P11 T3/T2
selection work and shared phase gate still own the final configuration and
production-model decision.

## One engine example and provenance

For original FD001/train engine 4 at the saved cutoff cycle 164, all candidates
scored the same target of 25 cycles. Predictions were 109.61 (mean baseline),
34.20 (Decision Tree), 34.12 (Random Forest), 28.69 (XGBoost), 15.85 (P10 MLP),
and 14.36 (wider MLP) cycles. The validation history contains observations
through cycle 164; the target is full training lifetime minus cutoff and was
used only for scoring. This example shows that errors vary by engine and that
aggregate MAE/RMSE do not guarantee a close estimate for an individual engine.
RUL is expressed in operating cycles; no physical-time conversion is implied.

The comparison script is:

```powershell
.venv/Scripts/python.exe scripts/summarize_p11_complexity.py
```

It verifies source/split provenance, the 20 unique validation identities and
labels, then recalculates each candidate's MAE/RMSE from per-engine
predictions. Observed output: all 6 candidates matched the same 20 identities
and labels; recomputed metrics matched all 6 recorded metric pairs. It writes
the reproducible local index `artifacts/experiments/P11_S3_complexity_summary.json`
(Git-ignored).

## Tuning budget, limits, and decision

The declared P11 budget was reached: three Decision Tree windows, no more than
three Random Forest and three XGBoost configurations, three feature sets, and
three MLP architectures. No further tuning was performed for this summary.
Every attempted weaker configuration remains saved in its separate ignored
run folder with its timing and predictions. The P11-A3 audit found 12 saved
S1/A1/A2 runs and no execution failures in those budgets.

These validation results were repeatedly used for window, depth, feature, and
architecture selection. They are development-selection evidence from only 20
engines, not an untouched final performance estimate. No official test
histories or RUL answers were read. Runtime values are machine-specific. A
lower validation error alone does not establish safety or maintenance
readiness.

Provenance: Python 3.12.10, scikit-learn 1.9.1, XGBoost 3.4.1, PyTorch
2.15.0+cpu; training-source SHA-256
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`; semantic
split-manifest SHA-256
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
