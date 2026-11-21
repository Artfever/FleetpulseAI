# P14 synthetic upload examples

All examples are invented synthetic fixtures, not NASA C-MAPSS observations.
They contain no labels or RUL answers. The `.txt` files exercise the existing
whitespace FD001 reader; the `.csv` files use the bounded P14 upload contract
with the exact 26 raw-history headers and no target column.

| CSV file | Expected result |
| --- | --- |
| `valid_synthetic_history.csv` | Parses three records with cycles 1, 2, 3 and passes quality checks. |
| `empty_history.csv` | Rejected because it has no header or rows. |
| `duplicate_cycle.csv` | Rejected for a duplicate engine-cycle pair. |
| `missing_channel.csv` | Rejected because the sensor_21 header is missing. |
| `text_sensor.csv` | Rejected because sensor_1 is not numeric. |
| `nonfinite_sensor.csv` | Rejected because sensor_1 is `nan`. |
| `shuffled_cycles.csv` | Rejected for non-increasing cycle order; rows are not sorted. |

Run `python -m pytest -q tests/test_upload_examples.py` to verify the actual
whitespace loader and CSV upload validator outcomes.
