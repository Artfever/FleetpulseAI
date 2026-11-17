# P11-T1: validation cohort comparability audit

## Result

Audited 20 saved candidate result artifacts spanning the constant baseline,
P06 Decision Tree, P08 Random Forest, P09 XGBoost, P10 MLP, and all saved
P11-S1/A1/A2/S2 experiment configurations. Every candidate contains the same
20 FD001/train validation engine-prefix identities, saved cutoffs, and scoring
targets from the P04 split. The audit found no missing, duplicate, reordered,
or changed identity/target rows.

The shared engine/cutoff pairs, in saved engine order, are:

`4:164, 5:58, 12:7, 14:71, 15:63, 18:58, 28:36, 29:27, 30:174, 32:190, 36:140, 55:23, 65:152, 70:109, 72:9, 76:8, 78:24, 82:56, 87:60, 95:259`.

Each row is qualified as `FD001/train`; the numeric unit IDs alone are not
treated as global identities. All candidate artifacts identify the same
original training source SHA-256
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` and
semantic P04 split SHA-256
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.

The audit independently recomputed MAE/RMSE from all per-engine predictions.
All 19 artifacts that stored a metric pair matched their saved metrics. The
P10 MLP manifest stores its 20 predictions but not MAE/RMSE; those metrics
were recomputed from the predictions for this audit. As one row-level example,
all candidates score FD001/train engine 4 at cycle 164 against the same target
of 25 RUL cycles. This confirms a shared scoring problem; it does not mean the
models use identical feature sets or that any metric is an untouched final
estimate.

## Reproduce

```powershell
.venv/Scripts/python.exe scripts/audit_p11_validation_comparability.py
```

Observed output: 20 candidate artifacts; 20 validation prefixes; all keys,
cutoffs, and targets matched; 19/19 stored metric pairs matched after
recomputation; the P10 metric pair was recomputed from its prediction rows.
The script writes `artifacts/experiments/P11_T1_validation_comparability.json`
locally; it is Git-ignored. The separate P11 S1/A1/A2/S2 per-run records are
also ignored local outputs, reproduced by their tracked experiment scripts.

No official test histories or RUL answers were opened. This cohort-comparison
check supports fair comparison on the development split but does not select a
production model; P11-T2/T3 and the shared gate own that decision.
