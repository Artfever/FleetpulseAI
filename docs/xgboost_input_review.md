# P09-A1: XGBoost feature and target review

Reviewed on 16 November 2026 against `dev` e03bdfb and Taha's trainer from
PR [#359](https://github.com/Artfever/FleetpulseAI/pull/359), merge d091f98.
Scope: [#54](https://github.com/Artfever/FleetpulseAI/issues/54).

## Finding and interface

The inspected trainer follows the common development-table contract:
`train_xgboost(fitting: FeatureDataset, validation: FeatureDataset, **parameters)`.
It fits exactly `fitting.X, fitting.y` once, predicts `validation.X`, then joins
validation targets for scoring. No validation `eval_set`, early stopping,
parameter search, source-file read, official test history, or answer-file read
occurs in this function. The result carries the fitted estimator, validation
metrics/predictions, ordered feature names, seed, parameters and runtime.
The existing shared contract is unchanged; no new teammate agreement is claimed.

- `features/build_dataset.py` obtains uncapped fitting targets from complete
  original training trajectories, separately from past-only feature vectors.
  `FeatureDataset.X` projects only vector values; IDs, cycles, full lifetimes
  and true RUL are outside X. Validation targets come from the audited split.
- `features/dataset_io.py` validates content hashes, declared feature order,
  widths, finite readings, row-key/metadata alignment and `FD001/train`
  partition. Pin source SHA, semantic split-manifest SHA and feature config
  when loading. The supplied audit preflight does so explicitly.
- `models/training_validation.py` rejects mismatched roles/schema/provenance,
  overlapping fitting/validation engines and repeated validation engines.
- `models/train_xgboost.py` preserves the table's ordered numeric columns and
  does not scale, cap, re-sort, reconstruct targets or choose a new split.

This boundary relies on checked table loading/building. A caller can manually
construct a `FeatureDataset` dataclass; the trainer's shared guard does not
independently authenticate every vector or its declared source partition.
Hashes establish integrity and lineage comparisons, not authenticity. Do not
pass arbitrary manually relabelled tables or treat this review as proof of an
unobserved experiment. No trainer/shared-feature code was rewritten.

## Independent synthetic checks

`tests/test_xgboost_input_audit.py` uses project-generated toy histories, not
NASA measurements or model-quality evidence. Saved feature config:
`past-only-v1`, channels `(sensor_2, sensor_1)`, window 2 cycles.
For engine 2, full lifetime 6, observed cycle 2:

| Quantity | Value |
| --- | --- |
| sensor_2 readings | 10, 20 |
| sensor_2 latest / mean / population std / actual-cycle slope | 20 / 15 / 5 / 10 |
| sensor_1 readings | 2, 4 |
| sensor_1 latest / mean / population std / actual-cycle slope | 4 / 3 / 1 / 2 |
| ordered X | `[20,15,5,10,4,3,1,2]` |
| separate target RUL | 6 - 2 = **4 operating cycles** |

The test intercepts real XGBoost fit/predict calls and requires exactly fitting
X/y with no additional fit arguments, followed by only validation X.
Changing validation labels by +1000 changes the score, but leaves parameters,
fitted booster and raw predictions identical. Changing validation features also
leaves the fitted booster unchanged. Replacing hidden future validation readings
with +/-1,000,000 leaves both fitting data and validation features/targets/keys
identical. Rehashed saved tables declaring `FD001/test` or a reversed feature
order are rejected by the loader before fitting.

Toy fits use 8 trees, depth 2, seed 42, one CPU worker, the trainer's remaining
fixed defaults (learning rate .05, min child weight 1, subsample .8,
colsample .8, reg lambda 1, square-error objective, histogram method).
Observed toy validation raw predictions: `[3.678098678588867, 3.678098678588867]`
cycles. Those values demonstrate executed code, not useful FD001 accuracy.
Abdullah explicitly confirmed his independent check and acceptance of the
numeric input/target example on 16 November. No teammate review or acceptance
of a real XGBoost candidate is claimed.

## Real input preflight, with no model fit

```powershell
.venv/Scripts/python.exe scripts/audit_xgboost_inputs.py
```

PASS: original `data/raw/train_FD001.txt` SHA
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`;
semantic split SHA
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
The source and split are audited before loading the two ignored common tables.
Fitting membership matches all 16,738 expected engine/cycle keys across 80
engines. Every fitting label equals that engine's complete last cycle minus
its observed cycle: range **0-361 cycles**, with no cap. All 20 validation
keys/targets match the saved manifest; all 20 validation feature vectors are
rebuilt from only their original raw prefixes and match exactly. Three sampled
fitting vectors also match their prefixes. All other fitting vectors receive
the existing saved-table schema/order/finite checks; the preflight does not
recompute every fitting feature value. Default config is 24 channels x four
statistics = 96 features, `past-only-v1`, 30-cycle window.

No official test history or official answer file is opened. This verifies
available inputs for the future experiment; it does not certify the provenance
of a real XGBoost fit that has not occurred here.

## Verification and remaining work

Python 3.12.16, XGBoost 3.4.1, scikit-learn 1.9.1.

```powershell
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pytest -q -s tests/test_xgboost_input_audit.py tests/test_train_xgboost.py
.venv/Scripts/python.exe -m pytest -q
```

Ruff PASS. Focused **19 passed in 4.78s**, including six new audit cases.
Full suite **320 passed in 29.60s**. `git diff --check` PASS.
Local `streamlit run app/Home.py`: **NOT RUN**, no app change.

Taha still owns #56 closeout, #133's budgeted real experiment/run record, and
#200's saved candidate/registry. No real XGBoost training, metrics, artifact,
reload parity, service integration, or deployment is claimed here. Abdullah's
#131 needs the actual saved candidate plus its input-batch/timing record;
#248 then audits loading-version metadata. The shared P09 gate remains separate.
