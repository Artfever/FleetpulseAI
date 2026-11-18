"""Create lossless, provenance-bearing CSV tables from the verified final report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetpulse.evaluation.final_test_display import (
    EXPECTED_RUL_SHA256,
    EXPECTED_TEST_HISTORY_SHA256,
    FINAL_TEST_REPORT_PATH,
    load_final_test_display,
)

DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/final/p13/exports")
SUMMARY_FILENAME = "p13_official_test_summary.csv"
ENGINE_FILENAME = "p13_official_test_per_engine.csv"
_TOLERANCE = 1e-12

SUMMARY_COLUMNS = (
    "Dataset partition",
    "Model",
    "Model version",
    "Sample count (engines)",
    "MAE (cycles)",
    "RMSE (cycles)",
    "Test history SHA-256",
    "RUL answer file SHA-256",
    "Prediction payload SHA-256",
    "P13 result payload SHA-256",
)
ENGINE_COLUMNS = (
    "Dataset partition",
    "Model",
    "Model version",
    "Engine ID",
    "Observed cycle",
    "RUL answer line",
    "True RUL (cycles)",
    "Raw prediction (cycles)",
    "Residual prediction minus true (cycles)",
    "Absolute error (cycles)",
    "Squared error (cycles squared)",
    "Test history SHA-256",
    "RUL answer file SHA-256",
    "Prediction payload SHA-256",
    "P13 result payload SHA-256",
)


@dataclass(frozen=True)
class FinalTestCsvExports:
    """CSV output and row counts produced only from validated report values."""

    summary_csv: str
    per_engine_csv: str
    summary_row_count: int
    per_engine_row_count: int
    summary_sha256: str
    per_engine_sha256: str


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _csv_text(columns: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _float_cell(value: float) -> str:
    return repr(float(value))


def build_final_test_csv_exports(
    report_path: Path = FINAL_TEST_REPORT_PATH,
) -> FinalTestCsvExports:
    """Validate every reported row/metric before serializing summary and detail CSVs."""
    display_rows = load_final_test_display(report_path)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    payload = document["payload"]
    prediction_source = payload["prediction_source"]
    if (
        prediction_source.get("test_history_sha256") != EXPECTED_TEST_HISTORY_SHA256
        or prediction_source.get("rul_answer_file_sha256") != EXPECTED_RUL_SHA256
        or document.get("sha256") != _canonical_sha256(payload)
    ):
        raise ValueError("Final result report provenance changed during export.")

    expected_by_name = {row.name: row for row in display_rows}
    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    models = payload["models"]
    if len(models) != len(expected_by_name):
        raise ValueError("Final result model inventory changed during export.")
    prediction_sha = prediction_source.get("pairing_payload_sha256")
    result_sha = document["sha256"]
    test_sha = prediction_source["test_history_sha256"]
    rul_sha = prediction_source["rul_answer_file_sha256"]
    if not isinstance(prediction_sha, str) or len(prediction_sha) != 64:
        raise ValueError("The verified pairing payload hash is missing from the final report.")

    for model in models:
        name, version = model.get("name"), model.get("version")
        summary = expected_by_name.get(name)
        if summary is None or summary.version != version:
            raise ValueError("A model row is missing from the validated display inventory.")
        rows = model.get("rows")
        if not isinstance(rows, list) or len(rows) != summary.sample_count:
            raise ValueError(f"Per-engine report rows are incomplete for {name}.")
        rows_by_id: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"Malformed per-engine result row for {name}.")
            unit_id = row.get("unit_id")
            if type(unit_id) is not int or unit_id in rows_by_id:
                raise ValueError(f"Duplicate or invalid engine ID in {name} report rows.")
            observed_cycle = row.get("observed_cycle")
            answer_line = row.get("answer_line")
            true_rul = row.get("true_rul_cycles")
            prediction = row.get("raw_prediction_cycles")
            residual = row.get("residual_prediction_minus_true_cycles")
            absolute_error = row.get("absolute_error_cycles")
            squared_error = row.get("squared_error_cycles_squared")
            if (
                type(observed_cycle) is not int
                or observed_cycle < 1
                or answer_line != unit_id
                or type(true_rul) is not int
                or true_rul < 0
                or type(prediction) not in (int, float)
                or not math.isfinite(prediction)
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in (residual, absolute_error, squared_error)
                )
            ):
                raise ValueError(f"Incomplete or invalid per-engine result data for {name}/{unit_id}.")
            expected_residual = float(prediction) - true_rul
            if (
                not math.isclose(float(residual), expected_residual, rel_tol=0, abs_tol=_TOLERANCE)
                or not math.isclose(
                    float(absolute_error), abs(expected_residual), rel_tol=0, abs_tol=_TOLERANCE
                )
                or not math.isclose(
                    float(squared_error), expected_residual**2, rel_tol=0, abs_tol=_TOLERANCE
                )
            ):
                raise ValueError(f"Per-engine error fields are inconsistent for {name}/{unit_id}.")
            rows_by_id[unit_id] = row

        expected_ids = list(range(1, summary.sample_count + 1))
        if sorted(rows_by_id) != expected_ids:
            raise ValueError(f"Expected one result for each official engine 1–100 in {name}.")
        ordered_rows = [rows_by_id[unit_id] for unit_id in expected_ids]
        mae = math.fsum(abs(float(row["residual_prediction_minus_true_cycles"])) for row in ordered_rows)
        mae /= summary.sample_count
        rmse = math.sqrt(
            math.fsum(float(row["squared_error_cycles_squared"]) for row in ordered_rows)
            / summary.sample_count
        )
        if (
            not math.isclose(mae, summary.mae_cycles, rel_tol=0, abs_tol=_TOLERANCE)
            or not math.isclose(rmse, summary.rmse_cycles, rel_tol=0, abs_tol=_TOLERANCE)
        ):
            raise ValueError(f"Per-engine rows do not reproduce summary scores for {name}.")

        summary_rows.append(
            {
                "Dataset partition": payload["dataset_partition"],
                "Model": summary.display_name,
                "Model version": summary.version,
                "Sample count (engines)": str(summary.sample_count),
                "MAE (cycles)": _float_cell(summary.mae_cycles),
                "RMSE (cycles)": _float_cell(summary.rmse_cycles),
                "Test history SHA-256": test_sha,
                "RUL answer file SHA-256": rul_sha,
                "Prediction payload SHA-256": prediction_sha,
                "P13 result payload SHA-256": result_sha,
            }
        )
        for unit_id in expected_ids:
            row = rows_by_id[unit_id]
            detail_rows.append(
                {
                    "Dataset partition": payload["dataset_partition"],
                    "Model": summary.display_name,
                    "Model version": summary.version,
                    "Engine ID": str(unit_id),
                    "Observed cycle": str(row["observed_cycle"]),
                    "RUL answer line": str(row["answer_line"]),
                    "True RUL (cycles)": str(row["true_rul_cycles"]),
                    "Raw prediction (cycles)": _float_cell(row["raw_prediction_cycles"]),
                    "Residual prediction minus true (cycles)": _float_cell(
                        row["residual_prediction_minus_true_cycles"]
                    ),
                    "Absolute error (cycles)": _float_cell(row["absolute_error_cycles"]),
                    "Squared error (cycles squared)": _float_cell(
                        row["squared_error_cycles_squared"]
                    ),
                    "Test history SHA-256": test_sha,
                    "RUL answer file SHA-256": rul_sha,
                    "Prediction payload SHA-256": prediction_sha,
                    "P13 result payload SHA-256": result_sha,
                }
            )

    summary_csv = _csv_text(SUMMARY_COLUMNS, summary_rows)
    per_engine_csv = _csv_text(ENGINE_COLUMNS, detail_rows)
    return FinalTestCsvExports(
        summary_csv=summary_csv,
        per_engine_csv=per_engine_csv,
        summary_row_count=len(summary_rows),
        per_engine_row_count=len(detail_rows),
        summary_sha256=hashlib.sha256(summary_csv.encode("utf-8")).hexdigest(),
        per_engine_sha256=hashlib.sha256(per_engine_csv.encode("utf-8")).hexdigest(),
    )


def write_final_test_csv_exports(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    report_path: Path = FINAL_TEST_REPORT_PATH,
) -> FinalTestCsvExports:
    """Write the validated CSV exports to an ignored local output directory."""
    exports = build_final_test_csv_exports(report_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / SUMMARY_FILENAME).write_text(
        exports.summary_csv, encoding="utf-8", newline="\n"
    )
    (output_directory / ENGINE_FILENAME).write_text(
        exports.per_engine_csv, encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "summary_file": SUMMARY_FILENAME,
                "summary_rows": exports.summary_row_count,
                "summary_sha256": exports.summary_sha256,
                "per_engine_file": ENGINE_FILENAME,
                "per_engine_rows": exports.per_engine_row_count,
                "per_engine_sha256": exports.per_engine_sha256,
            },
            sort_keys=True,
        )
    )
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--report", type=Path, default=FINAL_TEST_REPORT_PATH)
    args = parser.parse_args()
    write_final_test_csv_exports(args.output_directory, args.report)


if __name__ == "__main__":
    main()
