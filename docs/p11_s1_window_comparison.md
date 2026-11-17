# P11-S1: Decision Tree window comparison

## Protocol

Compared 10-, 20-, and 30-cycle past-only windows using the same Decision Tree
implementation (`decision-tree-rul-v1`), Python 3.12.10, scikit-learn 1.9.1,
random seed 42, `max_depth=8`, and `min_samples_leaf=5`. Each run rebuilt the
features from the same complete FD001/train histories and saved P04 split
manifest. This produced 16,738 fitting rows across 80 engines and 20 saved
validation prefixes, with 96 ordered features per row. Only the feature window
changed. No official test histories or RUL answers were opened.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe scripts/compare_p11_window_sizes.py
```

The script refuses to overwrite an existing experiment and verifies source and
manifest hashes before and after rebuilding. It saves separate datasets, a
model, and per-engine predictions under the ignored path
`artifacts/experiments/P11_S1_window_comparison/window_{10,20,30}/`. A second
process reloaded each saved model and validation table; all 20 predictions per
window matched the recorded values exactly.

## Results

| Past-only window | Validation MAE (cycles) | Validation RMSE (cycles) | Reload replay |
| ---: | ---: | ---: | --- |
| 10 cycles | 23.710807 | 31.458965 | 20/20 exact |
| 20 cycles | 23.478525 | 32.424615 | 20/20 exact |
| 30 cycles | **22.494270** | **29.449574** | 20/20 exact |

All three configurations scored the same 20 engines at the same saved cutoffs.
The 30-cycle window had the lowest MAE and RMSE in this comparison. It is a
candidate for the P11 configuration discussion; it is not a claim of final
model selection or production readiness.

## Limits and artifacts

This is one fixed development split of FD001/train, with only 20 validation
engines and one Decision Tree configuration. Results may change on another
split, another model, or final refit. The per-engine output and serialized
models are local, Git-ignored experiment artifacts; reproduce them with the
script above. The pinned original training source SHA-256 is
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`; the
semantic split-manifest SHA-256 is
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.

Outputs were stored separately from `artifacts/development/`; the existing
30-cycle development model and metrics were not overwritten. The next P11
consumer can use these results alongside the other model/window experiments
before the team records a final configuration decision.
