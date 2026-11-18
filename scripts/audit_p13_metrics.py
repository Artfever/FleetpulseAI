"""Independently recompute P13 final metrics from saved predictions and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PREDICTION_PATH = Path("artifacts/final/p13/official_test_predictions.json")
RUL_PATH = Path("data/raw/RUL_FD001.txt")
REPORTED_PATH = Path("artifacts/final/p13/final_test_results.json")
OUTPUT_PATH = Path("artifacts/final/p13/a2_independent_metric_audit.json")
EXPECTED_TEST_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
EXPECTED_RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
EXPECTED_ENGINE_COUNT = 100
EXPECTED_MODEL_COUNT = 5
AUDIT_SCHEMA = "p13-a2-independent-metric-audit-v1"
_METRIC_TOLERANCE = 1e-12


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def recompute_and_compare(
    prediction_document: Mapping[str, Any],
    rul_values: Sequence[int],
    reported_document: Mapping[str, Any],
    *,
    test_history_sha256: str,
    rul_sha256: str,
    expected_engine_count: int = EXPECTED_ENGINE_COUNT,
    expected_model_count: int = EXPECTED_MODEL_COUNT,
) -> dict[str, Any]:
    """Join by engine ID and recompute raw MAE/RMSE without P13 evaluator code."""
    prediction = prediction_document.get("payload")
    reported = reported_document.get("payload")
    if not isinstance(prediction, Mapping) or prediction_document.get("sha256") != _canonical_sha256(
        prediction
    ):
        raise ValueError("Saved prediction artifact checksum is invalid.")
    if not isinstance(reported, Mapping) or reported_document.get("sha256") != _canonical_sha256(
        reported
    ):
        raise ValueError("Reported result artifact checksum is invalid.")
    if (
        prediction.get("schema_version") != "p13-official-test-predictions-v1"
        or prediction.get("dataset_partition") != "FD001/test"
        or prediction.get("labels_read") is not False
        or prediction.get("source_sha256") != test_history_sha256
        or test_history_sha256 != EXPECTED_TEST_SHA256
        or rul_sha256 != EXPECTED_RUL_SHA256
    ):
        raise ValueError("Saved predictions or source hashes do not identify the pinned FD001/test inputs.")
    if (
        reported.get("schema_version") != "p13-final-test-results-v1"
        or reported.get("dataset_partition") != "FD001/test"
        or reported.get("results_use") != "final reporting only; not used to select, tune, or refit models"
    ):
        raise ValueError("Reported artifact is not the expected final FD001/test report.")

    engines = prediction.get("engines")
    engine_count = prediction.get("engine_count")
    if (
        not isinstance(engines, list)
        or type(engine_count) is not int
        or len(engines) != engine_count
        or engine_count != expected_engine_count
        or engine_count != len(rul_values)
    ):
        raise ValueError("Engine inventory and answer vector counts differ or are incomplete.")
    cutoffs: dict[int, int] = {}
    for engine in engines:
        if not isinstance(engine, Mapping):
            raise TypeError("Malformed engine inventory row.")
        unit_id, cycle = engine.get("unit_id"), engine.get("observed_cycle")
        if type(unit_id) is not int or unit_id < 1 or type(cycle) is not int or cycle < 1:
            raise ValueError("Invalid engine identity or observed cutoff.")
        if unit_id in cutoffs:
            raise ValueError(f"Duplicate engine ID in saved inventory: {unit_id}.")
        cutoffs[unit_id] = cycle
    ordered_ids = sorted(cutoffs)
    labels = {unit_id: value for unit_id, value in zip(ordered_ids, rul_values, strict=True)}
    answer_lines = {unit_id: line for line, unit_id in enumerate(ordered_ids, start=1)}
    if any(type(value) is not int or value < 0 for value in rul_values):
        raise ValueError("Every RUL answer must be a nonnegative integer.")

    source_models = prediction.get("models")
    result_models = reported.get("models")
    if (
        not isinstance(source_models, list)
        or not isinstance(result_models, list)
        or len(source_models) != expected_model_count
        or len(result_models) != expected_model_count
    ):
        raise TypeError("Prediction or reported model inventory is missing.")
    reported_by_name: dict[str, Mapping[str, Any]] = {}
    for model in result_models:
        if not isinstance(model, Mapping) or not isinstance(model.get("name"), str):
            raise TypeError("Malformed model in reported results.")
        if model["name"] in reported_by_name:
            raise ValueError(f"Duplicate reported model: {model['name']}.")
        reported_by_name[model["name"]] = model

    audited: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for model in source_models:
        if not isinstance(model, Mapping):
            raise TypeError("Malformed saved model prediction table.")
        name, version, predictions = model.get("name"), model.get("version"), model.get("predictions")
        if not isinstance(name, str) or name in seen_models or not isinstance(version, str):
            raise ValueError("Saved model names and versions must be unique and valid.")
        seen_models.add(name)
        if not isinstance(predictions, list) or len(predictions) != engine_count:
            raise ValueError(f"Prediction count is incomplete for {name}.")
        by_id: dict[int, Mapping[str, Any]] = {}
        for row in predictions:
            if not isinstance(row, Mapping):
                raise TypeError(f"Malformed prediction row for {name}.")
            unit_id, cycle, value = (
                row.get("unit_id"),
                row.get("observed_cycle"),
                row.get("raw_rul_cycles"),
            )
            if type(unit_id) is not int or unit_id in by_id:
                raise ValueError(f"Duplicate or invalid engine ID in {name} predictions.")
            if cycle != cutoffs.get(unit_id) or not _finite_number(value):
                raise ValueError(f"Cutoff or raw prediction is invalid for {name}, engine {unit_id}.")
            by_id[unit_id] = row
        if set(by_id) != set(ordered_ids):
            raise ValueError(f"Prediction identity coverage differs for {name}.")

        reported_model = reported_by_name.get(name)
        if reported_model is None or reported_model.get("version") != version:
            raise ValueError(f"Reported result model/version differs for {name}.")
        rows = reported_model.get("rows")
        metrics = reported_model.get("metrics")
        if not isinstance(rows, list) or len(rows) != engine_count or not isinstance(metrics, Mapping):
            raise ValueError(f"Reported row table or metrics are incomplete for {name}.")
        reported_rows: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"Malformed reported row for {name}.")
            unit_id = row.get("unit_id")
            if type(unit_id) is not int or unit_id in reported_rows:
                raise ValueError(f"Duplicate or invalid reported engine ID for {name}.")
            reported_rows[unit_id] = row
        if set(reported_rows) != set(ordered_ids):
            raise ValueError(f"Reported engine identity coverage differs for {name}.")

        absolute_errors: list[float] = []
        squared_errors: list[float] = []
        for unit_id in ordered_ids:
            prediction_row = by_id[unit_id]
            actual = float(prediction_row["raw_rul_cycles"])
            truth = labels[unit_id]
            error = actual - truth
            absolute_errors.append(abs(error))
            squared_errors.append(error * error)
            reported_row = reported_rows[unit_id]
            if (
                reported_row.get("observed_cycle") != cutoffs[unit_id]
                or reported_row.get("answer_line") != answer_lines[unit_id]
                or reported_row.get("true_rul_cycles") != truth
                or not math.isclose(
                    float(reported_row.get("raw_prediction_cycles", math.nan)),
                    actual,
                    rel_tol=0,
                    abs_tol=_METRIC_TOLERANCE,
                )
            ):
                raise ValueError(f"Reported per-engine row differs from independent source at {name}/{unit_id}.")

        mae = math.fsum(absolute_errors) / engine_count
        rmse = math.sqrt(math.fsum(squared_errors) / engine_count)
        saved_mae, saved_rmse = metrics.get("mae_cycles"), metrics.get("rmse_cycles")
        if not _finite_number(saved_mae) or not _finite_number(saved_rmse):
            raise ValueError(f"Reported aggregate metrics are invalid for {name}.")
        mae_delta = mae - float(saved_mae)
        rmse_delta = rmse - float(saved_rmse)
        if abs(mae_delta) > _METRIC_TOLERANCE or abs(rmse_delta) > _METRIC_TOLERANCE:
            raise ValueError(f"Independent metrics do not match reported values for {name}.")
        audited.append(
            {
                "name": name,
                "version": version,
                "engine_count": engine_count,
                "independent_mae_cycles": mae,
                "reported_mae_cycles": float(saved_mae),
                "mae_delta_cycles": mae_delta,
                "independent_rmse_cycles": rmse,
                "reported_rmse_cycles": float(saved_rmse),
                "rmse_delta_cycles": rmse_delta,
                "per_engine_rows_checked": engine_count,
            }
        )
    if seen_models != set(reported_by_name):
        raise ValueError("Saved prediction and reported model inventories differ.")
    return {
        "schema_version": AUDIT_SCHEMA,
        "dataset_partition": "FD001/test",
        "mapping": "RUL answer line i maps to sorted test unit_id i; joined predictions by unit_id",
        "engine_count": engine_count,
        "model_count": len(audited),
        "prediction_payload_sha256": prediction_document["sha256"],
        "test_history_sha256": test_history_sha256,
        "rul_sha256": rul_sha256,
        "reported_payload_sha256": reported_document["sha256"],
        "comparison_tolerance_cycles": _METRIC_TOLERANCE,
        "models": audited,
        "metrics_recomputed": True,
        "tuning_performed": False,
    }


def run_audit(
    prediction_path: Path = PREDICTION_PATH,
    rul_path: Path = RUL_PATH,
    reported_path: Path = REPORTED_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """Load pinned local files, independently compare scores, and save audit JSON."""
    prediction_bytes = prediction_path.read_bytes()
    rul_bytes = rul_path.read_bytes()
    reported_bytes = reported_path.read_bytes()
    test_path = Path("data/raw/test_FD001.txt")
    test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    rul_sha256 = hashlib.sha256(rul_bytes).hexdigest()
    rul_values = [int(value) for value in rul_bytes.split()]
    payload = recompute_and_compare(
        json.loads(prediction_bytes),
        rul_values,
        json.loads(reported_bytes),
        test_history_sha256=test_sha256,
        rul_sha256=rul_sha256,
    )
    document = {"payload": payload, "sha256": _canonical_sha256(payload)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **payload}, indent=2, sort_keys=True))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTION_PATH)
    parser.add_argument("--rul", type=Path, default=RUL_PATH)
    parser.add_argument("--reported", type=Path, default=REPORTED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    run_audit(args.predictions, args.rul, args.reported, args.output)


if __name__ == "__main__":
    main()
