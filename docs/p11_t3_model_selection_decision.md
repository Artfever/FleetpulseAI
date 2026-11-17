# P11-T3: production-model candidate decision

## Recorded choice

Under Shaheer's explicit instruction to complete the P08-P16 work on his
behalf, select **XGBoost depth 4** as the model candidate for the next production
integration step. This records the owner-delegated choice; no separate teammate
consensus or approval is claimed.

The comparison uses the six saved candidates from the checked P11 selection
manifest. All six use the same 20 held-out FD001/train engine/cutoff identities
and target labels. Their validation MAE/RMSE in cycles were:

| Candidate | MAE | RMSE |
| --- | ---: | ---: |
| Mean constant baseline | 65.728002 | 74.258987 |
| Decision Tree, 30-cycle window | 22.494270 | 29.449574 |
| Random Forest, depth 16 | 18.360242 | 24.563262 |
| **XGBoost, depth 4 (selected)** | 15.108697 | **21.092113** |
| MLP, hidden `(64, 32)` | 15.511258 | 22.937993 |
| MLP, hidden `(128, 64)` | **15.080604** | 22.066152 |

## Rationale and trade-off

XGBoost depth 4 has the lowest observed RMSE and was the model candidate with
the lowest squared-error score. It also provides a tree-based candidate for
the planned explanation work. Its recorded configuration is seed 42, 100
estimators, depth 4, learning rate 0.05, subsample 0.8, column sample 0.8,
CPU `hist` tree method, and one job. The saved P11-A1 result reports a
machine-specific fit-and-predict time of 1.373 seconds.

The selected model version is `xgboost-rul-v1`. The saved P11-A1 depth-4
checkpoint SHA-256 is
`20801e897816f879c0f93f56dc199c49735c3afc3a544b595247f5c62d419612` and
matches the same-version P09 service model file byte-for-byte. The P11-A1 run
archive reloaded all six saved models and reproduced 20/20 validation
predictions for each; the P09 service verification separately reproduced this
model's 20/20 saved outputs. These are local artifacts and replay checks; they
do not mean the candidate has been deployed.

The MLP with hidden widths `(128, 64)` has the lowest MAE: 15.080604 cycles,
0.027907 cycles below XGBoost. Its RMSE is 22.066152 cycles, 0.974039 cycles
higher than XGBoost. Selecting XGBoost therefore gives up that small MAE edge
in favor of the best observed RMSE and tree explainability. The stronger MLP
MAE result is retained in the manifest and Model Laboratory. The MLP's saved
run used seed 42, batch size 128, learning rate 0.001, CPU, and selected epoch
8 of 11 trained; its machine-specific runtime was 2.083 seconds.

This choice is a candidate configuration decision from a repeatedly used
20-engine development-validation selection cohort. It is not an untouched
final performance estimate, a statistical significance claim, evidence of
reliable per-engine maintenance predictions, or deployment approval. The
planned explanation work has not yet been performed. Runtime is machine
specific. No official test histories or RUL answers were opened, and no new
training or additional hyperparameter search was run for this decision.

The compact tracked decision is stored in
[`artifacts/selection_manifest.json`](../artifacts/selection_manifest.json).
To reproduce its comparison from the ignored local P11 result files, run:

```powershell
.venv/Scripts/python.exe scripts/audit_p11_validation_comparability.py
.venv/Scripts/python.exe scripts/summarize_p11_complexity.py
.venv/Scripts/python.exe scripts/build_p11_selection_manifest.py
```

The six selected run files' hashes must match the P11-T1 comparability audit.
The manifest's canonical payload SHA-256 is recorded in the Shaheer handoff.
