from __future__ import annotations

import hashlib
import json
import math

import pytest

from scripts.audit_p13_metrics import (
    EXPECTED_RUL_SHA256,
    EXPECTED_TEST_SHA256,
    recompute_and_compare,
)


def _seal(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"payload": payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _documents() -> tuple[dict[str, object], list[int], dict[str, object]]:
    engine_rows = [
        {"unit_id": 30, "observed_cycle": 7},
        {"unit_id": 10, "observed_cycle": 11},
        {"unit_id": 20, "observed_cycle": 9},
    ]
    definitions = {
        "tree": ("tree-v1", {10: 8.0, 20: 12.0, 30: 4.0}),
        "forest": ("forest-v1", {10: 9.0, 20: 6.0, 30: 15.0}),
    }
    source_models = []
    reported_models = []
    labels = {10: 3, 20: 5, 30: 10}
    for name, (version, predictions) in definitions.items():
        # Input prediction order is deliberately scrambled; alignment is by ID.
        prediction_rows = [
            {"unit_id": unit_id, "observed_cycle": next(e["observed_cycle"] for e in engine_rows if e["unit_id"] == unit_id), "raw_rul_cycles": predictions[unit_id]}
            for unit_id in (30, 10, 20)
        ]
        errors = [predictions[unit_id] - labels[unit_id] for unit_id in sorted(labels)]
        mae = math.fsum(abs(error) for error in errors) / len(errors)
        rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
        result_rows = [
            {
                "unit_id": unit_id,
                "observed_cycle": next(e["observed_cycle"] for e in engine_rows if e["unit_id"] == unit_id),
                "answer_line": index,
                "true_rul_cycles": labels[unit_id],
                "raw_prediction_cycles": predictions[unit_id],
            }
            for index, unit_id in enumerate(sorted(labels), start=1)
        ]
        source_models.append({"name": name, "version": version, "predictions": prediction_rows})
        reported_models.append(
            {"name": name, "version": version, "rows": result_rows, "metrics": {"mae_cycles": mae, "rmse_cycles": rmse}}
        )
    prediction_payload = {
        "schema_version": "p13-official-test-predictions-v1",
        "dataset_partition": "FD001/test",
        "labels_read": False,
        "source_sha256": EXPECTED_TEST_SHA256,
        "engine_count": 3,
        "engines": engine_rows,
        "models": source_models,
    }
    reported_payload = {
        "schema_version": "p13-final-test-results-v1",
        "dataset_partition": "FD001/test",
        "results_use": "final reporting only; not used to select, tune, or refit models",
        "models": reported_models,
    }
    return _seal(prediction_payload), [3, 5, 10], _seal(reported_payload)


def test_recomputes_metrics_and_joins_shuffled_predictions_by_engine_id() -> None:
    predictions, labels, reported = _documents()
    result = recompute_and_compare(
        predictions,
        labels,
        reported,
        test_history_sha256=EXPECTED_TEST_SHA256,
        rul_sha256=EXPECTED_RUL_SHA256,
        expected_engine_count=3,
        expected_model_count=2,
    )

    assert result["engine_count"] == 3
    assert result["model_count"] == 2
    assert result["tuning_performed"] is False
    assert all(model["mae_delta_cycles"] == 0 for model in result["models"])
    assert all(model["rmse_delta_cycles"] == 0 for model in result["models"])


def test_rejects_reported_metric_mismatch() -> None:
    predictions, labels, reported = _documents()
    payload = reported["payload"]
    payload["models"][0]["metrics"]["mae_cycles"] += 0.1
    reported = _seal(payload)

    with pytest.raises(ValueError, match="metrics do not match"):
        recompute_and_compare(
            predictions,
            labels,
            reported,
            test_history_sha256=EXPECTED_TEST_SHA256,
            rul_sha256=EXPECTED_RUL_SHA256,
            expected_engine_count=3,
            expected_model_count=2,
        )


def test_rejects_answer_count_mismatch() -> None:
    predictions, labels, reported = _documents()
    with pytest.raises(ValueError, match="counts differ"):
        recompute_and_compare(
            predictions,
            labels[:-1],
            reported,
            test_history_sha256=EXPECTED_TEST_SHA256,
            rul_sha256=EXPECTED_RUL_SHA256,
            expected_engine_count=3,
            expected_model_count=2,
        )


def test_rejects_wrong_rul_source_hash() -> None:
    predictions, labels, reported = _documents()
    with pytest.raises(ValueError, match="source hashes"):
        recompute_and_compare(
            predictions,
            labels,
            reported,
            test_history_sha256=EXPECTED_TEST_SHA256,
            rul_sha256="0" * 64,
            expected_engine_count=3,
            expected_model_count=2,
        )
