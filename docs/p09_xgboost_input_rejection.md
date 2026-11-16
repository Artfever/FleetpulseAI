# P09-S3: reject nonfinite readings before XGBoost inference

The common prediction service validates a supplied engine history before it
opens a model manifest or deserializes an estimator. `validate_engine_history`
reports a nonfinite sensor or setting as `non_finite_reading`; the service
raises `PredictionInputError` and stops. This applies to the registered
XGBoost candidate as well as the other model services.

`tests/test_xgboost_artifact.py::test_nonfinite_xgboost_reading_is_rejected_before_model_load_or_predict`
uses the saved-candidate service route with one validation-prefix row changed
to NaN, positive infinity, or negative infinity. For each value it verifies
the error code and asserts that the model loader, estimator deserializer, and
estimator `predict` method are all untouched. This establishes that invalid
numeric readings do not reach the XGBoost model.

These are input-contract tests over synthetic histories and a temporary
synthetic candidate bundle. They do not modify the real model, measure model
accuracy, or use official test histories or answers. The test can be rerun with:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_xgboost_artifact.py::test_nonfinite_xgboost_reading_is_rejected_before_model_load_or_predict
```
