# P11-T2: saved candidate selection comparison

## Delivered view

The Model Laboratory reads the integrity-checked `artifacts/selection_manifest.json`
and displays six saved candidate runs in cycles: the fitting-label mean baseline,
30-cycle Decision Tree, depth-16 Random Forest, depth-4 XGBoost, and MLPs with
hidden widths `(64, 32)` and `(128, 64)`. The view identifies the lowest MAE and
lowest RMSE candidates separately. It leaves the production-model choice for the
P11-T3 team decision because the metrics favor different candidates.

## Observed comparison

All six result artifacts use the same 20 FD001/train validation engine/cutoff
identities and scoring targets, training-source hash, and P04 split hash.
Independently recomputed values:

| Candidate | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Mean constant baseline | 65.728002 | 74.258987 |
| Decision Tree, 30-cycle window | 22.494270 | 29.449574 |
| Random Forest, depth 16 | 18.360242 | 24.563262 |
| XGBoost, depth 4 | 15.108697 | **21.092113** |
| MLP, hidden `(64, 32)` | 15.511258 | 22.937993 |
| MLP, hidden `(128, 64)` | **15.080604** | 22.066152 |

The wider MLP has the lowest observed MAE; XGBoost has the lowest observed
RMSE. This is development-validation selection evidence from a repeatedly used
20-engine cohort, not an untouched performance estimate, individual-engine
accuracy guarantee, or maintenance-readiness result. No official test data was
opened.

## Reproduction and validation

Run these commands from the repository root with the locally supplied FD001
training file and saved experiment outputs:

```powershell
.venv/Scripts/python.exe scripts/audit_p11_validation_comparability.py
.venv/Scripts/python.exe scripts/summarize_p11_complexity.py
.venv/Scripts/python.exe scripts/build_p11_selection_manifest.py
```

The first audit checked 20 candidate artifacts; all used the same 20 identities,
cutoffs, and labels. Its 19 stored metric pairs matched recomputation; P10's
metrics were recomputed from its saved predictions because that manifest does
not contain aggregate metrics. The six-candidate summary recomputed all six
MAE/RMSE pairs. The manifest builder additionally matched each selected result
file's SHA-256 to the independent P11-T1 audit before writing the compact JSON.
Current generated manifest SHA-256:
`e37a387be33205e15f884d44ab76c725059ffe12be2e9317f25a164cf9761b15`.

The experiment outputs remain ignored local files. The compact selection
manifest is tracked so the comparison is available in a fresh clone and the
hosted app; it records result paths and hashes to make the locally reproduced
source evidence explicit. Its embedded content hash detects accidental edits;
it is not a cryptographic signature.
