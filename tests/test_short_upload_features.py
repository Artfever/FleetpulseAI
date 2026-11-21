"""Short valid CSV prefixes preserve the saved past-only feature layout."""

import csv
import io
import math

import pytest

from fleetpulse.data.input_validation import parse_fd001_csv_upload
from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.features import FeatureConfig, engineer_features


def _row(cycle):
    return {
        field: (
            930001
            if field == "unit_id"
            else cycle
            if field in {"cycle", "sensor_1"}
            else cycle * (index + 1)
        )
        for index, field in enumerate(HISTORY_COLUMNS)
    }


def _csv_bytes(records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


@pytest.mark.parametrize("prefix_length", [1, 3, 10, 29, 30])
def test_valid_short_upload_keeps_schema_order_and_uses_available_rows(prefix_length):
    config = FeatureConfig()
    records = parse_fd001_csv_upload(_csv_bytes([_row(cycle) for cycle in range(1, prefix_length + 1)]))

    vector = engineer_features(
        records,
        dataset_partition="synthetic/P14-S3",
        unit_id=930001,
        config=config,
    )

    assert vector.engine_key == ("synthetic/P14-S3", 930001)
    assert vector.selected_cycle == prefix_length
    assert vector.observed_row_count == vector.window_row_count == prefix_length
    assert len(vector.values) == len(config.output_schema) == 96
    assert tuple(vector.as_dict()) == config.output_schema
    assert all(math.isfinite(value) for value in vector.values)
    assert "rul" not in vector.as_dict()

    assert vector.as_dict()["sensor_1_latest"] == float(prefix_length)
    assert vector.as_dict()["sensor_1_mean"] == pytest.approx((prefix_length + 1) / 2)
    if prefix_length == 1:
        assert vector.as_dict()["sensor_1_std"] == 0.0
        assert vector.as_dict()["sensor_1_slope"] == 0.0
    else:
        assert vector.as_dict()["sensor_1_slope"] == pytest.approx(1.0)
