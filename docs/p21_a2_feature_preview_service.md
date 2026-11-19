# P21-A2: keep Engine Explorer calculations in UI-free services

The Engine Explorer page now handles user selection and rendering only. It
calls `fleetpulse.services.build_engine_feature_preview` for the selected
engine/cycle and `fleetpulse.services.summarize_sensor_history` for its
descriptive trend text. The service modules do not import Streamlit. Numeric
feature engineering remains centralized in `fleetpulse.features.engineering`;
the preview service passes only rows through the requested cutoff and returns
both that visible history and its ordered `EngineFeatureVector`.

`EngineFeaturePreview` contains `visible_history` and `vector`. The vector
retains dataset/engine identity, selected cycle, observed/window row counts,
feature config, ordered names and values. `SensorHistorySummary` contains a
descriptive flat/varies label, observed min/max and strict direction-reversal
count. Neither is a fault diagnosis.

## Real packaged example

The saved packaged data at `data/demo/fd001_training_prefixes.txt` was used
without retraining. Its SHA-256 is
`953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`.
For `FD001/train` engine 4 through cycle 164, the service returned 164 visible
rows, 96 ordered features, and config `past-only-v1` (30-cycle window, 24
channels). In the window ending at cycle 164:

| Feature | Value | Units/meaning |
| --- | ---: | --- |
| `sensor_1_latest` | 518.67 | Recorded sensor value |
| `sensor_1_mean` | 518.67 | Mean of observed window readings |
| `sensor_1_std` | 0.0 | Population standard deviation of window readings |
| `sensor_1_slope` | 0.0 | Recorded sensor units per cycle |
| `sensor_4_latest` | 1411.81 | Recorded sensor value |
| `sensor_4_mean` | 1411.7286666666666 | Mean of observed window readings |
| `sensor_4_std` | 4.397945530459507 | Population standard deviation of window readings |
| `sensor_4_slope` | 0.2859310344827574 | Recorded sensor units per cycle |

This is a feature inspection from one simulated FD001 training prefix. It is
not a prediction, accuracy result, or physical fault explanation.
Reproduce and verify these values with
`.venv/Scripts/python.exe scripts/check_p21_a2_feature_preview.py`.

## Verification

- `tests/test_feature_preview.py` drives the Streamlit page with synthetic
  records and verifies the service passes only the selected engine prefix to
  feature engineering, preserving the feature order and values.
- `tests/test_feature_preview_service.py` calls the feature service directly
  with no Streamlit page and confirms a future `sensor_1=1e100` row does not
  enter the cycle-2 vector `(4, 3, 1, 2)`.
- `tests/test_sensor_history_summary.py` checks the trend-summary behavior.
- `tests/test_engine_selection.py` checks the user-visible trend descriptions.
