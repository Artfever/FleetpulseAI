"""The CSV boundary validates before shared model inference."""

import csv
import io

import pytest

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.input_validation import UploadValidationError
from fleetpulse.data.loader import HISTORY_COLUMNS, load_engine_history
from fleetpulse.services import predict_rul, predict_uploaded_csv


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


def test_clean_csv_matches_builtin_history_prediction():
    manifest = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    records = load_engine_history(PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt")
    engine_history = [row for row in records if row["unit_id"] == 4]
    uploaded_csv = _to_csv_bytes(engine_history)

    built_in = predict_rul(
        engine_history,
        dataset_partition="FD001/train",
        unit_id=4,
        manifest_path=manifest,
    )
    uploaded = predict_uploaded_csv(
        uploaded_csv,
        dataset_partition="FD001/train",
        unit_id=4,
        manifest_path=manifest,
    )

    assert uploaded.raw_rul_cycles == built_in.raw_rul_cycles
    assert uploaded.displayed_rul_cycles == built_in.raw_rul_cycles
    assert uploaded.model_name == "decision-tree-rul"
    assert uploaded.model_version == "decision-tree-rul-v1"
    assert uploaded.observed_cycle == 164
    assert uploaded.input_window_rows == 30
