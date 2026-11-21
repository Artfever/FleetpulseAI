from __future__ import annotations

import csv
import hashlib
import json

import pytest

from fleetpulse.evaluation.final_test_display import (
    EXPECTED_RUL_SHA256,
    EXPECTED_TEST_HISTORY_SHA256,
)
from fleetpulse.evaluation.final_test_export import (
    ENGINE_FILENAME,
    SUMMARY_FILENAME,
    build_final_test_csv_exports,
    write_final_test_csv_exports,
)


def _seal(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"payload": payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _write_report(path, *, residual_offset: float = 0.0, metric_offset: float = 0.0) -> None:
    versions = {
        "decision-tree-rul": ("Decision Tree", "decision-tree-rul-v1"),
        "random-forest-rul": ("Random Forest", "random-forest-rul-v1"),
        "xgboost-rul": ("XGBoost", "xgboost-rul-v1"),
        "constant-rul": ("Constant mean baseline", "constant-rul-v1"),
        "dense-rul-mlp-final": ("Dense MLP", "dense-rul-mlp-final-v1"),
    }
    models = []
    for name, (display, version) in versions.items():
        rows = []
        for unit_id in range(1, 101):
            error = 5.0 if unit_id <= 50 else 35.0
            residual = error + residual_offset
            rows.append(
                {
                    "unit_id": unit_id,
                    "observed_cycle": unit_id + 20,
                    "answer_line": unit_id,
                    "true_rul_cycles": 100,
                    "raw_prediction_cycles": 100.0 + error,
                    "residual_prediction_minus_true_cycles": residual,
                    "absolute_error_cycles": abs(residual),
                    "squared_error_cycles_squared": residual**2,
                }
            )
        models.append(
            {
                "name": name,
                "display_name": display,
                "version": version,
                "sample_count": 100,
                "metrics": {
                    "mae_cycles": 20.0 + metric_offset,
                    "rmse_cycles": 25.0 + metric_offset,
                },
                "rows": rows,
            }
        )
    payload = {
        "schema_version": "p13-final-test-results-v1",
        "dataset_partition": "FD001/test",
        "results_use": "final reporting only; not used to select, tune, or refit models",
        "sample_count_per_model": 100,
        "prediction_source": {
            "test_history_sha256": EXPECTED_TEST_HISTORY_SHA256,
            "rul_answer_file_sha256": EXPECTED_RUL_SHA256,
            "pairing_payload_sha256": "a" * 64,
        },
        "models": models,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seal(payload)), encoding="utf-8")


def test_exports_exact_summary_and_all_per_engine_rows(tmp_path) -> None:
    report = tmp_path / "final_test_results.json"
    _write_report(report)

    exports = build_final_test_csv_exports(report)
    summaries = list(csv.DictReader(exports.summary_csv.splitlines()))
    details = list(csv.DictReader(exports.per_engine_csv.splitlines()))

    assert exports.summary_row_count == 5
    assert exports.per_engine_row_count == 500
    assert len(summaries) == 5
    assert len(details) == 500
    assert all(row["Dataset partition"] == "FD001/test" for row in details)
    assert all(row["Sample count (engines)"] == "100" for row in summaries)
    assert all(row["MAE (cycles)"] == "20.0" for row in summaries)
    assert all(row["RMSE (cycles)"] == "25.0" for row in summaries)
    assert all(all(value != "" for value in row.values()) for row in summaries + details)
    assert details[0]["Engine ID"] == "1"
    assert details[99]["Engine ID"] == "100"
    assert details[100]["Model"] == "Random Forest"
    assert all(row["Test history SHA-256"] == EXPECTED_TEST_HISTORY_SHA256 for row in details)


def test_rejects_inconsistent_per_engine_error_instead_of_filling(tmp_path) -> None:
    report = tmp_path / "final_test_results.json"
    _write_report(report, residual_offset=0.5)

    with pytest.raises(ValueError, match="error fields are inconsistent"):
        build_final_test_csv_exports(report)


def test_rejects_missing_prediction_value_instead_of_exporting_blank(tmp_path) -> None:
    report = tmp_path / "final_test_results.json"
    _write_report(report)
    document = json.loads(report.read_text())
    del document["payload"]["models"][0]["rows"][0]["raw_prediction_cycles"]
    report.write_text(json.dumps(_seal(document["payload"])), encoding="utf-8")

    with pytest.raises(ValueError, match="Incomplete or invalid per-engine result data"):
        build_final_test_csv_exports(report)


def test_rejects_metrics_that_do_not_match_exported_rows(tmp_path) -> None:
    report = tmp_path / "final_test_results.json"
    _write_report(report, metric_offset=0.5)

    with pytest.raises(ValueError, match="do not reproduce summary scores"):
        build_final_test_csv_exports(report)


def test_write_export_files_creates_both_report_tables(tmp_path) -> None:
    report = tmp_path / "final_test_results.json"
    output = tmp_path / "exports"
    _write_report(report)

    exports = write_final_test_csv_exports(output, report)

    summary_path = output / SUMMARY_FILENAME
    detail_path = output / ENGINE_FILENAME
    assert summary_path.is_file()
    assert detail_path.is_file()
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == exports.summary_sha256
    assert hashlib.sha256(detail_path.read_bytes()).hexdigest() == exports.per_engine_sha256
