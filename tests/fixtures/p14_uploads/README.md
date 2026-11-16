# P14 synthetic upload examples

These whitespace-separated histories are invented test fixtures, not NASA
C-MAPSS observations. A valid example and one example for each required failure
case are included so a user or teammate can compare input shape and diagnostics.

| File | Expected result |
| --- | --- |
| `valid_synthetic_history.txt` | Loads three records and passes history validation. |
| `empty_history.txt` | Loader rejects a history with no records. |
| `duplicate_cycle.txt` | Loads, then validation reports a duplicate engine-cycle pair. |
| `missing_channel.txt` | Loader rejects a row with only 25 of the required 26 fields. |
| `text_sensor.txt` | Loader rejects a nonnumeric sensor token. |
| `nonfinite_sensor.txt` | Loads `nan` as a float, then validation rejects the nonfinite reading. |
| `shuffled_cycles.txt` | Loads in original order, then validation reports a non-increasing cycle. |

Run `python -m pytest -q tests/test_upload_examples.py` to verify the actual
loader and validator outcomes. The fixtures contain no labels or RUL answers.
