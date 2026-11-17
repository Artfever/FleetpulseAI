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
Unexpected fields (including guessed aliases and RUL/target columns) are
rejected with instructions to remove them; the parser never infers column
meanings. Each engine's cycles must be in ascending row order. Out-of-order
input is rejected with a sorting instruction, and the parser does not reorder
rows automatically.

`fleetpulse.services.predict_uploaded_csv` is the prediction boundary for CSV
histories: it completes parsing and quality validation before calling the
shared saved-model predictor, and requires the caller to provide the selected
engine ID. Tests verify an invalid CSV never reaches that predictor and a clean
CSV made from the packaged engine 4 history matches the built-in service result
exactly. This service-level check does not claim the Streamlit upload screen is
implemented; that UI is covered by P14-T2.

Short valid histories keep the saved `FeatureConfig` schema unchanged (96
ordered columns by default). Feature windows use the observations available
within the configured cycle interval; they are not padded or extended. For a
single observation, population standard deviation and slope are zero. Latest,
mean and standard deviation retain the raw channel units; slope is channel
units per operating cycle. The published example is synthetic and does not
represent NASA measurements. A short-window prediction is software evidence,
not model-quality or maintenance evidence.

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
