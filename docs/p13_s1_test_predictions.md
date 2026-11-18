# P13-S1: Frozen-model predictions on FD001/test prefixes

`python -m fleetpulse.evaluation.test` loads the original `data/raw/test_FD001.txt` sensor histories and the five checksum-verified P12 final bundles. It reads no `RUL_FD001.txt` answer file and performs no fitting or model selection. The P13-S2 scoring task consumes the resulting prediction artifact and handles label pairing separately.

## Prediction contract

- Partition: `FD001/test`; IDs must be exactly 1 through 100.
- One prediction per registered final model and test engine, using that engine's greatest observed cycle.
- Feature input: the existing ordered 96-feature `past-only-v1` contract, with the available observations in the inclusive 30-cycle window. Missing cycles are not filled.
- Output values are raw RUL predictions in cycles. They are not clipped or scored in this task.
- The report records model version, source-history checksum, engine/cutoff identity, and payload checksum. Different output at the same path is rejected; choose a new versioned path to preserve a prior result.

## Observed local run

Command: `.venv/Scripts/python.exe -m fleetpulse.evaluation.test`

- Official sensor-history file SHA-256: `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`.
- 100 distinct test engines; every engine had one last-cycle row and all five final models produced exactly 100 predictions. Data-quality warnings: none.
- Engine 1's last observed cycle was 31 with 31 observed rows and a 30-row feature window. Raw predictions: Decision Tree `164.96629213`, Random Forest `153.98205177`, XGBoost `145.28688049`, constant mean `107.80786196`, and final MLP `158.88244629` cycles.
- Prediction ranges across these 100 engines: Decision Tree `[2.53846154, 216.47755102]`; Random Forest `[5.07746515, 197.60937658]`; XGBoost `[8.29245186, 193.90315247]`; constant mean `[107.80786196, 107.80786196]`; final MLP `[2.83313751, 180.78842163]` cycles. These are output ranges, not accuracy metrics.
- Ignored local result: `artifacts/final/p13/official_test_predictions.json`. Payload SHA-256: `34a92a60d2a6af6d279f929210c29d52ef43172c2d2a7122bb249d35538ffc2d`; whole-file SHA-256: `8bf55e2bb529f52a846a0187c65f0e1360d9bfca83920acb3278bcfd0afB0FBD`.
- The output and model files are local/Git-ignored. They are not included in this PR; the tracked generator and tests reproduce the predictions only when the original data and verified P12 bundles are present.

## Verification and boundary

- `.venv/Scripts/python.exe -m pytest tests/test_official_test_predictions.py -q`: 4 passed.
- `.venv/Scripts/python.exe -m pytest -q`: 488 passed.
- `.venv/Scripts/ruff.exe check .`: passed. `git diff --check`: passed.
- The prediction generator accepts a test-history path and output path, but no label-file argument. This run did not open `RUL_FD001.txt`, calculate any scores, or retune a model. P13-S2 will map the separately stored answers to sorted engine IDs and calculate metrics.
