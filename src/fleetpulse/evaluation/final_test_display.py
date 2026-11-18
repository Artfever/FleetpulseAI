"""Load the verified, saved FD001/test report for a clearly separated UI view."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINAL_TEST_SCHEMA = "p13-final-test-results-v1"
FINAL_TEST_PARTITION = "FD001/test"
FINAL_TEST_REPORT_PATH = Path("artifacts/final/p13/final_test_results.json")
EXPECTED_TEST_HISTORY_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
EXPECTED_RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
EXPECTED_VERSIONS = {
    "decision-tree-rul": "decision-tree-rul-v1",
    "random-forest-rul": "random-forest-rul-v1",
    "xgboost-rul": "xgboost-rul-v1",
    "constant-rul": "constant-rul-v1",
    "dense-rul-mlp-final": "dense-rul-mlp-final-v1",
}


@dataclass(frozen=True)
class FinalTestDisplayRow:
    """A validated summary row from the saved final held-out test report."""

    name: str
    display_name: str
    version: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_final_test_display(path: Path = FINAL_TEST_REPORT_PATH) -> tuple[FinalTestDisplayRow, ...]:
    """Load the checksum-sealed P13 report, rejecting incomplete or wrong-split data."""
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("Saved official test report checksum is invalid.")
    source = payload.get("prediction_source")
    if (
        payload.get("schema_version") != FINAL_TEST_SCHEMA
        or payload.get("dataset_partition") != FINAL_TEST_PARTITION
        or payload.get("results_use")
        != "final reporting only; not used to select, tune, or refit models"
        or not isinstance(source, dict)
        or source.get("test_history_sha256") != EXPECTED_TEST_HISTORY_SHA256
        or source.get("rul_answer_file_sha256") != EXPECTED_RUL_SHA256
    ):
        raise ValueError("Saved results do not match the pinned final FD001/test report.")

    models = payload.get("models")
    if (
        not isinstance(models, list)
        or len(models) != len(EXPECTED_VERSIONS)
        or payload.get("sample_count_per_model") != 100
    ):
        raise ValueError("Saved final test report has an incomplete model/sample inventory.")
    rows: list[FinalTestDisplayRow] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("Saved final test model row is malformed.")
        name, version = model.get("name"), model.get("version")
        if not isinstance(name, str) or name in seen or EXPECTED_VERSIONS.get(name) != version:
            raise ValueError("Saved final test model identity/version is missing or unexpected.")
        metrics = model.get("metrics")
        mae = metrics.get("mae_cycles") if isinstance(metrics, dict) else None
        rmse = metrics.get("rmse_cycles") if isinstance(metrics, dict) else None
        count = model.get("sample_count")
        raw_rows = model.get("rows")
        display_name = model.get("display_name")
        if (
            type(count) is not int
            or count != 100
            or not isinstance(raw_rows, list)
            or len(raw_rows) != count
            or not isinstance(display_name, str)
            or not display_name.strip()
            or type(mae) not in (int, float)
            or not math.isfinite(mae)
            or type(rmse) not in (int, float)
            or not math.isfinite(rmse)
        ):
            raise ValueError(f"Saved final test summary is incomplete for {name}.")
        seen.add(name)
        rows.append(
            FinalTestDisplayRow(
                name=name,
                display_name=display_name,
                version=version,
                sample_count=count,
                mae_cycles=float(mae),
                rmse_cycles=float(rmse),
            )
        )
    if seen != set(EXPECTED_VERSIONS):
        raise ValueError("Saved final test model coverage is incomplete.")
    return tuple(rows)
