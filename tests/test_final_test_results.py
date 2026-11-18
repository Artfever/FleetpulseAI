"""Checks for official test residual and MAE/RMSE reporting."""

import pytest

from fleetpulse.evaluation.final_test_results import summarize_paired_results
from fleetpulse.evaluation.official_test_pairing import PAIRING_SCHEMA, _canonical_sha256


def _paired_document():
    payload = {
        "schema_version": PAIRING_SCHEMA,
        "dataset_partition": "FD001/test",
        "scores_computed": False,
        "engine_count": 2,
        "test_history_sha256": "a" * 64,
        "answer_file": "RUL_FD001.txt",
        "answer_file_sha256": "b" * 64,
        "models": [
            {
                "name": "decision-tree-rul",
                "version": "decision-tree-rul-v1",
                "paired_count": 2,
                "rows": [
                    {
                        "unit_id": 1,
                        "observed_cycle": 31,
                        "answer_line": 1,
                        "true_rul_cycles": 10,
                        "raw_prediction_cycles": 12.0,
                    },
                    {
                        "unit_id": 2,
                        "observed_cycle": 49,
                        "answer_line": 2,
                        "true_rul_cycles": 20,
                        "raw_prediction_cycles": 17.0,
                    },
                ],
            }
        ],
    }
    return {"payload": payload, "sha256": _canonical_sha256(payload)}


def _artifacts():
    return {
        "decision-tree-rul": {
            "version": "decision-tree-rul-v1",
            "display_name": "Decision Tree",
            "manifest_payload_sha256": "c" * 64,
            "feature_config_sha256": "d" * 64,
        }
    }


def test_result_summary_saves_per_engine_residuals_and_raw_mae_rmse():
    results = summarize_paired_results(
        _paired_document(), _artifacts(), paired_file_sha256="e" * 64
    )

    model = results["models"][0]
    assert model["sample_count"] == 2
    assert model["metrics"]["mae_cycles"] == 2.5
    assert model["metrics"]["rmse_cycles"] == pytest.approx((6.5) ** 0.5)
    assert [
        row["residual_prediction_minus_true_cycles"] for row in model["rows"]
    ] == [2.0, -3.0]
    assert model["rows"][0]["absolute_error_cycles"] == 2.0
    assert model["rows"][1]["squared_error_cycles_squared"] == 9.0
    assert results["metric_definitions"]["predictions_clipped"] is False
    assert "not used" in results["results_use"]


def test_result_summary_rejects_changed_model_version_or_bad_order():
    artifacts = _artifacts()
    artifacts["decision-tree-rul"]["version"] = "unexpected-v2"
    with pytest.raises(ValueError, match="version differs"):
        summarize_paired_results(
            _paired_document(), artifacts, paired_file_sha256="e" * 64
        )

    artifact = _paired_document()
    artifact["payload"]["models"][0]["rows"].reverse()
    artifact["sha256"] = _canonical_sha256(artifact["payload"])
    with pytest.raises(ValueError, match="ascending engine-ID order"):
        summarize_paired_results(
            artifact, _artifacts(), paired_file_sha256="e" * 64
        )
