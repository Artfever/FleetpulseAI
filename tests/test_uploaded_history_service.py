"""The CSV boundary validates before shared model inference."""

import csv
import io

import pytest

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.input_validation import UploadValidationError
from fleetpulse.data.loader import HISTORY_COLUMNS, load_engine_history
from fleetpulse.services import RULPrediction, predict_uploaded_csv


def _to_csv_bytes(records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HISTORY_COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def test_invalid_upload_is_rejected_before_prediction_service(monkeypatch, tmp_path):
    from fleetpulse.services import uploaded_history

    calls = []

    def unexpected_prediction(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid CSV must not reach prediction")

    monkeypatch.setattr(uploaded_history, "predict_rul_with_context", unexpected_prediction)
    invalid_csv = (PROJECT_ROOT / "tests/fixtures/p14_uploads/text_sensor.csv").read_bytes()

    with pytest.raises(UploadValidationError, match="numeric value"):
        predict_uploaded_csv(
            invalid_csv,
            dataset_partition="FD001/train",
            unit_id=910001,
            manifest_path=tmp_path / "unused-manifest.json",
        )

    assert calls == []


def test_clean_csv_matches_builtin_history_prediction(monkeypatch):
    from fleetpulse.services import uploaded_history

    records = load_engine_history(PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt")
    engine_history = [row for row in records if row["unit_id"] == 4]
    uploaded_csv = _to_csv_bytes(engine_history)
    calls = []

    def controlled_predictor(history, **kwargs):
        calls.append((tuple(history), kwargs))
        final_cycle = int(history[-1]["cycle"])
        controlled_value = float(history[-1]["sensor_1"])
        return RULPrediction(controlled_value, controlled_value, "test-model", "test-v1", final_cycle, 30, ())

    monkeypatch.setattr(uploaded_history, "predict_rul_with_context", controlled_predictor)

    built_in = controlled_predictor(
        engine_history,
        dataset_partition="FD001/train",
        unit_id=4,
        manifest_path="test-only",
    )
    uploaded = predict_uploaded_csv(
        uploaded_csv,
        dataset_partition="FD001/train",
        unit_id=4,
        manifest_path="test-only",
    )

    assert uploaded == built_in
    assert calls[1] == calls[0]
    assert uploaded.raw_rul_cycles == float(engine_history[-1]["sensor_1"])
    assert uploaded.model_name == "test-model"
    assert uploaded.model_version == "test-v1"
    assert uploaded.observed_cycle == engine_history[-1]["cycle"]
    assert uploaded.input_window_rows == 30
