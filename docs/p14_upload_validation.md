# P14 CSV history validation

Download the [synthetic upload example](examples/p14_synthetic_history_upload.csv)
to see the exact accepted header and row shape. It has one invented engine
with three ordered cycles, all 26 required fields, and no target/RUL column.
The readings are artificial and are only for understanding the format and
testing validation; they are not NASA observations or a model-quality example.

`fleetpulse.data.input_validation.parse_fd001_csv_upload` accepts UTF-8 CSV
bytes with a header and maps fields by name to the normalized raw FD001 history
schema (`unit_id`, `cycle`, three operating settings and 21 sensors). All 26
required names must appear exactly once; extra columns, including any RUL
target, are rejected. Header order may vary, but data row order is preserved.

The upload limit is 1 MiB. CSV syntax, UTF-8 decoding, header, row width and
numeric parsing errors use `UploadValidationError`, with stable finding codes
and row/field details where available. Parsed values are checked by the shared
`validate_engine_history` contract for finite readings, positive integral
engine/cycle identifiers, duplicate engine-cycle pairs and cycle order. Cycle
gaps do not reject parsing; callers can use `validate_engine_history` to show
its warning findings. Records are never silently sorted or dropped.

Run `python -m pytest -q tests/test_input_validation.py` for the synthetic
valid, reordered-header, malformed-header, nonnumeric, non-finite, invalid
cycle/identity, duplicate-pair, ragged-row and upload-size examples. No RUL
labels or official test data are used.
