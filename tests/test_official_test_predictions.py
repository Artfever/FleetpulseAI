"""Checks for the P13 official-test prediction stage, before label scoring."""

import pytest

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.evaluation.test import (
    FinalPredictor,
    _write_immutable_report,
    build_prediction_report,
)
from fleetpulse.features.config import FeatureConfig


def _rows():
    records = []
    for unit_id, cycles in ((1, (1, 2, 4)), (2, (1, 3))):
        for cycle in cycles:
            record = {"unit_id": unit_id, "cycle": cycle}
            record.update(
                {column: float(unit_id * 100 + cycle) for column in READING_COLUMNS}
            )
            records.append(record)
    return records


def _predictor(name, value):
    return FinalPredictor(
        name=name,
        version=f"{name}-frozen-v1",
        feature_config=FeatureConfig(),
        predict=lambda matrix: (value,),
    )


def test_report_predicts_once_at_each_engine_last_observed_cycle_without_labels():
    report = build_prediction_report(
        _rows(),
        {"tree": _predictor("tree", 12.5), "mlp": _predictor("mlp", 8.25)},
        source_sha256="a" * 64,
        source_name="test_FD001.txt",
        expected_engine_ids=(1, 2),
    )

    assert report["dataset_partition"] == "FD001/test"
    assert report["engine_count"] == 2
    assert report["labels_read"] is False
    assert report["engines"] == [
        {"unit_id": 1, "observed_cycle": 4, "observed_row_count": 3, "window_row_count": 3},
        {"unit_id": 2, "observed_cycle": 3, "observed_row_count": 2, "window_row_count": 2},
    ]
    assert [
        (row["unit_id"], row["observed_cycle"], row["raw_rul_cycles"])
        for row in report["models"][0]["predictions"]
    ] == [(1, 4, 12.5), (2, 3, 12.5)]
    assert all(model["prediction_count"] == 2 for model in report["models"])
    assert all("true_rul" not in row for model in report["models"] for row in model["predictions"])


def test_report_rejects_incomplete_official_engine_identity_set():
    with pytest.raises(ValueError, match="identity sequence"):
        build_prediction_report(
            _rows(),
            {"tree": _predictor("tree", 12.5)},
            source_sha256="a" * 64,
            source_name="test_FD001.txt",
            expected_engine_ids=(1, 2, 3),
        )


def test_report_rejects_predictor_returning_more_than_one_value_per_engine():
    invalid = FinalPredictor(
        name="invalid",
        version="invalid-v1",
        feature_config=FeatureConfig(),
        predict=lambda matrix: (1.0, 2.0),
    )
    with pytest.raises(ValueError, match="exactly one prediction"):
        build_prediction_report(
            _rows(),
            {"invalid": invalid},
            source_sha256="a" * 64,
            source_name="test_FD001.txt",
        )


def test_report_output_is_immutable(tmp_path):
    path = tmp_path / "predictions.json"
    first = {"schema_version": "test", "predictions": [1.0]}
    digest = _write_immutable_report(first, path)
    assert len(digest) == 64
    assert _write_immutable_report(first, path) == digest
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_immutable_report({"schema_version": "test", "predictions": [2.0]}, path)
