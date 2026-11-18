"""Explicit, one-to-one pairing of predictions to a separate RUL vector."""

import pytest

from fleetpulse.evaluation.official_test_pairing import (
    EXPECTED_TEST_HISTORY_SHA256,
    PAIRING_SCHEMA,
    _canonical_sha256,
    pair_test_predictions,
)
from fleetpulse.evaluation.test import PREDICTION_SCHEMA


def _document(*, engine_rows=None, prediction_rows=None):
    engines = engine_rows or [
        {"unit_id": 9, "observed_cycle": 7},
        {"unit_id": 2, "observed_cycle": 4},
    ]
    predictions = prediction_rows or [
        {"unit_id": 9, "observed_cycle": 7, "raw_rul_cycles": 18.5},
        {"unit_id": 2, "observed_cycle": 4, "raw_rul_cycles": 29.0},
    ]
    payload = {
        "schema_version": PREDICTION_SCHEMA,
        "dataset_partition": "FD001/test",
        "source_sha256": EXPECTED_TEST_HISTORY_SHA256,
        "labels_read": False,
        "engine_count": 2,
        "engines": engines,
        "models": [
            {
                "name": "decision-tree-rul",
                "version": "decision-tree-rul-v1",
                "prediction_count": 2,
                "predictions": predictions,
            }
        ],
    }
    return {"payload": payload, "sha256": _canonical_sha256(payload)}


def test_shuffled_engine_rows_pair_by_sorted_id_and_record_answer_line():
    paired = pair_test_predictions(
        _document(),
        (81, 17),
        rul_source_sha256="a" * 64,
        rul_source_name="RUL_FD001.txt",
        expected_engine_ids=(2, 9),
    )

    assert paired["schema_version"] == PAIRING_SCHEMA
    assert paired["scores_computed"] is False
    assert paired["models"][0]["rows"] == [
        {
            "unit_id": 2,
            "observed_cycle": 4,
            "answer_line": 1,
            "true_rul_cycles": 81,
            "raw_prediction_cycles": 29.0,
        },
        {
            "unit_id": 9,
            "observed_cycle": 7,
            "answer_line": 2,
            "true_rul_cycles": 17,
            "raw_prediction_cycles": 18.5,
        },
    ]


def test_pairing_rejects_wrong_answer_count():
    with pytest.raises(ValueError, match="does not match engine count"):
        pair_test_predictions(
            _document(),
            (81,),
            rul_source_sha256="a" * 64,
            rul_source_name="RUL_FD001.txt",
        )


def test_pairing_rejects_duplicate_or_missing_model_engine_rows():
    duplicated = [
        {"unit_id": 2, "observed_cycle": 4, "raw_rul_cycles": 29.0},
        {"unit_id": 2, "observed_cycle": 4, "raw_rul_cycles": 30.0},
    ]
    with pytest.raises(ValueError, match="duplicate prediction"):
        pair_test_predictions(
            _document(prediction_rows=duplicated),
            (81, 17),
            rul_source_sha256="a" * 64,
            rul_source_name="RUL_FD001.txt",
        )


def test_pairing_rejects_cutoff_or_prediction_hash_mismatch():
    wrong_cutoff = [
        {"unit_id": 9, "observed_cycle": 6, "raw_rul_cycles": 18.5},
        {"unit_id": 2, "observed_cycle": 4, "raw_rul_cycles": 29.0},
    ]
    with pytest.raises(ValueError, match="cutoff differs"):
        pair_test_predictions(
            _document(prediction_rows=wrong_cutoff),
            (81, 17),
            rul_source_sha256="a" * 64,
            rul_source_name="RUL_FD001.txt",
        )
    bad_hash = _document()
    bad_hash["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload hash"):
        pair_test_predictions(
            bad_hash,
            (81, 17),
            rul_source_sha256="a" * 64,
            rul_source_name="RUL_FD001.txt",
        )
