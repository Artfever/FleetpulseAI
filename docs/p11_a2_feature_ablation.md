# P11-A2: controlled channel ablation

## Protocol

Compared three predeclared FD001 channel sets with a fixed XGBoost model:

| Configuration | Channels | Feature columns |
| --- | ---: | ---: |
| Full baseline | 3 operating settings + 21 sensors | 96 |
| Sensors only | 21 sensors (settings ablated) | 84 |
| Settings only | 3 operating settings | 12 |

All three used the same `past-only-v1` 30-cycle window, 80 fitting engines,
and 20 saved validation engine/cutoff pairs. XGBoost was fixed at version
`xgboost-rul-v1`, seed 42, 100 estimators, depth 4, learning rate 0.05,
min-child-weight 1, subsample/column sample 0.8, L2 1, CPU histogram method,
and one worker. This was the leading XGBoost setting in P11-A1 #60. Reusing
the same development validation set makes this an exploratory comparison, not
an independent confirmation of that model choice.

Reproduce from repository root:

```powershell
.venv/Scripts/python.exe scripts/run_p11_a2_feature_ablation.py
```

The script rebuilds both fitting and validation tables from the verified
FD001/train histories and P04 manifest for each channel set. It checks that
all configurations retain exactly the same validation identities/cutoffs and
refuses to overwrite a nonempty output directory. Separate tables, models,
and per-engine records are saved under the ignored directory
`artifacts/experiments/P11_A2_feature_ablation/`.

## Results

| Configuration | Validation MAE (cycles) | Validation RMSE (cycles) | Reload replay |
| --- | ---: | ---: | --- |
| Full baseline | **15.108697** | **21.092113** | 20/20 exact |
| Sensors only | 16.718605 | 21.282901 | 20/20 exact |
| Settings only | 53.064679 | 59.503527 | 20/20 exact |

Settings-only performed much worse on this split. Removing settings modestly
increased both aggregate error measures versus the full baseline; this is
evidence that the settings contributed useful signal in this experiment, not a
causal conclusion or a guarantee on new engines. Per-engine results vary: for
FD001/train engine 4 at saved cutoff cycle 164, target RUL was 25 cycles and
predictions were 28.693811 (full), 26.937719 (sensors only), and 88.952713
(settings only). The aggregate and single-engine results therefore tell
different parts of the story.

## Independent feature example and limits

The validation history for engine 4 is from the original FD001/train source,
not FD001/test. At its saved cutoff 164, the feature builder saw 164 observed
rows and used only the inclusive 30-cycle window. Independently rebuilding the
same vector directly from rows through cycle 164 matched the saved vector
exactly. In the sensors-only config, `sensor_2_latest` was 643.2,
`sensor_2_mean` was 642.810667, and its least-squares slope was
0.018959 sensor units per cycle. The raw sensor unit is dataset-specific and
not given a physical interpretation here. Its full-history RUL label of 25
cycles was used only for validation scoring, never as a feature.

All scores are from a single fixed development split of 20 engines. No official
test histories or answers were read. Other P11 experiments and the shared
phase decision remain in scope; this result alone does not freeze production
features or certify model quality.

- Python 3.12.10; XGBoost 3.4.1.
- Training source SHA-256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Semantic P04 split SHA-256: `6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
