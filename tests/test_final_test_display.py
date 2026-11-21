from __future__ import annotations

import hashlib
import json

import pytest

from fleetpulse.evaluation.final_test_display import (
    EXPECTED_RUL_SHA256,
    EXPECTED_TEST_HISTORY_SHA256,
    load_final_test_display,
)


def _write_report(path, *, sample_count: int = 100, checksum_ok: bool = True) -> None:
    versions = {
        "decision-tree-rul": ("Decision Tree", "decision-tree-rul-v1"),
        "random-forest-rul": ("Random Forest", "random-forest-rul-v1"),
        "xgboost-rul": ("XGBoost", "xgboost-rul-v1"),
        "constant-rul": ("Constant mean baseline", "constant-rul-v1"),
        "dense-rul-mlp-final": ("Dense MLP", "dense-rul-mlp-final-v1"),
    }
    payload = {
        "schema_version": "p13-final-test-results-v1",
        "dataset_partition": "FD001/test",
        "results_use": "final reporting only; not used to select, tune, or refit models",
        "sample_count_per_model": 100,
        "prediction_source": {
            "test_history_sha256": EXPECTED_TEST_HISTORY_SHA256,
            "rul_answer_file_sha256": EXPECTED_RUL_SHA256,
        },
        "models": [
            {
                "name": name,
                "display_name": display,
                "version": version,
                "sample_count": sample_count,
                "metrics": {"mae_cycles": 12.5, "rmse_cycles": 16.75},
                "rows": [{} for _ in range(sample_count)],
            }
            for name, (display, version) in versions.items()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    if not checksum_ok:
        digest = "0" * 64
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "sha256": digest}), encoding="utf-8")


def test_loads_only_complete_verified_official_test_report(tmp_path) -> None:
    path = tmp_path / "final_test_results.json"
    _write_report(path)

    rows = load_final_test_display(path)

    assert len(rows) == 5
    assert rows[0].name == "decision-tree-rul"
    assert rows[0].display_name == "Decision Tree"
    assert rows[0].version == "decision-tree-rul-v1"
    assert rows[0].sample_count == 100
    assert rows[0].mae_cycles == 12.5
    assert rows[0].rmse_cycles == 16.75


def test_rejects_bad_report_checksum(tmp_path) -> None:
    path = tmp_path / "final_test_results.json"
    _write_report(path, checksum_ok=False)

    with pytest.raises(ValueError, match="checksum"):
        load_final_test_display(path)


def test_rejects_incomplete_engine_tables(tmp_path) -> None:
    path = tmp_path / "final_test_results.json"
    _write_report(path, sample_count=99)

    with pytest.raises(ValueError, match="incomplete"):
        load_final_test_display(path)
