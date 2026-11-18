"""Independent P13-A1 tests for sorted test IDs and separate label lines."""

import pytest

from scripts.audit_p13_label_alignment import (
    sorted_id_line_mapping,
    verify_model_pair_rows,
)


def test_shuffled_toy_ids_are_paired_by_sorted_id_not_input_order():
    mapping = sorted_id_line_mapping((30, 10, 20), (300, 100, 200))
    assert mapping == [
        {"answer_line": 1, "unit_id": 10, "true_rul_cycles": 300},
        {"answer_line": 2, "unit_id": 20, "true_rul_cycles": 100},
        {"answer_line": 3, "unit_id": 30, "true_rul_cycles": 200},
    ]


def test_mapping_rejects_duplicate_engine_ids_or_wrong_answer_count():
    with pytest.raises(ValueError, match="duplicates"):
        sorted_id_line_mapping((2, 1, 2), (5, 6, 7))
    with pytest.raises(ValueError, match="does not match"):
        sorted_id_line_mapping((2, 1), (5,))


def test_pair_audit_joins_shuffled_rows_by_engine_id_and_checks_cutoffs():
    mapping = [
        {"answer_line": 1, "unit_id": 10, "true_rul_cycles": 111},
        {"answer_line": 2, "unit_id": 30, "true_rul_cycles": 222},
    ]
    shuffled_model_rows = [
        {
            "unit_id": 30,
            "answer_line": 2,
            "true_rul_cycles": 222,
            "observed_cycle": 7,
            "raw_prediction_cycles": 13.0,
        },
        {
            "unit_id": 10,
            "answer_line": 1,
            "true_rul_cycles": 111,
            "observed_cycle": 5,
            "raw_prediction_cycles": 14.0,
        },
    ]
    assert verify_model_pair_rows(shuffled_model_rows, mapping, {10: 5, 30: 7}) == 2

    shuffled_model_rows[0]["answer_line"] = 1
    with pytest.raises(ValueError, match="misaligned"):
        verify_model_pair_rows(shuffled_model_rows, mapping, {10: 5, 30: 7})


def test_pair_audit_rejects_duplicate_or_missing_engine_predictions():
    mapping = [
        {"answer_line": 1, "unit_id": 1, "true_rul_cycles": 4},
        {"answer_line": 2, "unit_id": 2, "true_rul_cycles": 7},
    ]
    rows = [
        {
            "unit_id": 1,
            "answer_line": 1,
            "true_rul_cycles": 4,
            "observed_cycle": 3,
            "raw_prediction_cycles": 8.0,
        },
        {
            "unit_id": 1,
            "answer_line": 1,
            "true_rul_cycles": 4,
            "observed_cycle": 3,
            "raw_prediction_cycles": 9.0,
        },
    ]
    with pytest.raises(ValueError, match="Duplicate paired row"):
        verify_model_pair_rows(rows, mapping, {1: 3, 2: 8})
