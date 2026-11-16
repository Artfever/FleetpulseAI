"""Focused integrity tests for the saved-forest prediction audit."""

import pytest

from scripts.audit_random_forest_validation import recompute_prediction_audit


def _inputs():
    validation = [
        {"key": ["FD001/train", 4, 164], "true_rul": 25},
        {"key": ["FD001/train", 5, 58], "true_rul": 211},
    ]
    predictions = [
        {
            "dataset_partition": "FD001/train",
            "unit_id": 4,
            "cutoff_cycle": 164,
            "prediction_cycles": 34.0,
            "true_rul_cycles": 25,
            "absolute_error_cycles": 9.0,
            "squared_error_cycles": 81.0,
        },
        {
            "dataset_partition": "FD001/train",
            "unit_id": 5,
            "cutoff_cycle": 58,
            "prediction_cycles": 159.0,
            "true_rul_cycles": 211,
            "absolute_error_cycles": 52.0,
            "squared_error_cycles": 2704.0,
        },
    ]
    return validation, predictions


def test_recompute_prediction_audit_checks_rows_and_aggregates():
    validation, predictions = _inputs()

    result = recompute_prediction_audit(validation, predictions)

    assert result["sample_count"] == 2
    assert result["engine_ids"] == [4, 5]
    assert result["mae_cycles"] == 30.5
    assert result["rmse_cycles"] == pytest.approx((1392.5) ** 0.5)
    assert [row["unit_id"] for row in result["rows"]] == [4, 5]


@pytest.mark.parametrize("fault", ["duplicate", "unknown"])
def test_recompute_prediction_audit_rejects_bad_identity_coverage(fault):
    validation, predictions = _inputs()
    if fault == "duplicate":
        predictions[1] = dict(predictions[0])
    else:
        predictions[1]["unit_id"] = 99

    with pytest.raises(ValueError, match="count|identity"):
        recompute_prediction_audit(validation, predictions)


def test_recompute_prediction_audit_rejects_changed_target_or_row_error():
    validation, predictions = _inputs()
    predictions[0]["true_rul_cycles"] = 24

    with pytest.raises(ValueError, match="target"):
        recompute_prediction_audit(validation, predictions)

    validation, predictions = _inputs()
    predictions[0]["squared_error_cycles"] = 80
    with pytest.raises(ValueError, match="squared error"):
        recompute_prediction_audit(validation, predictions)
