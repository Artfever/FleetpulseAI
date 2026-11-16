# P16-S1 future-reading invariance check

The prediction service accepts one observed engine prefix; the caller must
truncate future rows at the selected cutoff. This check uses each of the 20
saved FD001/train development-validation prefixes twice. For the second pass,
it replaces every later reading with alternating extreme finite values and
appends five more extreme cycles, then applies the same cutoff before inference.
The future values and appended rows therefore never enter the model request.

The check uses the saved `decision-tree-rul-v1` **development model**, not a
final refitted model. Its manifest records 80 fitting engines; the verified
split has 20 validation engines, with no engine overlap. Thus the validation
histories were not used to fit this development model. This is a data-flow and
leakage check, not a new model-quality estimate. FD001 test histories and RUL
answers are not read.

## Reproduction

With the reviewed, ignored FD001/train file and development Decision Tree
bundle installed locally, run:

```powershell
.venv\Scripts\python.exe scripts\check_future_prediction_invariance.py
.venv\Scripts\python.exe -m pytest -q tests\test_feature_future_invariance.py
```

The script verifies the source hash and fitting/validation identity separation,
then prints model/data hashes, configuration, per-engine cutoffs and raw
predictions, and the maximum prediction difference. Tests skip the real-data
inference case when the ignored files are absent from a clone; synthetic
feature-level invariance tests still run.

## Local observed result (2026-11-16)

- Source: FD001/train SHA-256 `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
- Model: `decision-tree-rul-v1`, development bundle; scikit-learn 1.9.1, seed 42, `past-only-v1`, 30-cycle window.
- Model SHA-256: `2a693e6ecc09de5d9549e287a6798fdb339c7e945c700231444f0c5bcd425018`; split manifest SHA-256: `47eeae52d9099035f1db2780f2e11e8ec7718eec807aa248d11141dd67072c78`.
- Split: 80 fitting engines, 20 validation engines, zero overlap.
- Observed: 20 of 20 earlier raw predictions matched exactly after future edits; maximum absolute difference was 0.0 cycles. Predictions ranged from 2.5204 to 216.1084 cycles.
- The result covers only these saved development-validation prefixes and this saved model. It does not establish behavior for arbitrary caller code that fails to truncate history, nor final-model evaluation quality.
