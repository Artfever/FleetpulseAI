# Example CSV files

`p14_synthetic_history_upload.csv` is a small, valid upload example with the
exact 26-column raw-history schema accepted by the P14 validator. It contains
three invented readings for one invented unit (`910001`), has increasing
cycles, and has no RUL or other target column.

The readings are synthetic and intentionally obvious; they are not NASA
C-MAPSS observations, representative measurements, training data, or evidence
of model quality. Use this file to understand the CSV shape and exercise input
validation only. Do not interpret its sensor values as real engine telemetry.
