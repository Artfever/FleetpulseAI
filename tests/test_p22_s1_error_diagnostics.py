"""P22-S1 declared true-RUL bin and error-summary rules."""

import pytest

from scripts.run_p22_s1_error_diagnostics import rul_bin, summarize_model


@pytest.mark.parametrize(
    ("cycles", "expected"),
    [
        (0, "low"),
        (30, "low"),
        (31, "medium"),
        (100, "medium"),
        (101, "high"),
    ],
)
def test_true_rul_bin_boundaries_are_inclusive_and_explicit(cycles, expected):
    assert rul_bin(cycles) == expected


def test_summary_keeps_residual_sign_and_populations_separate():
    summary = summarize_model(
        [
            {"true_rul_cycles": 30, "raw_prediction_cycles": 34.0},
            {"true_rul_cycles": 31, "raw_prediction_cycles": 29.0},
            {"true_rul_cycles": 101, "raw_prediction_cycles": 106.0},
        ]
    )
    assert summary["sample_count"] == 3
    assert summary["overall"] == {
        "count": 3,
        "mae_cycles": 3.6666666666666665,
        "rmse_cycles": 3.872983346207417,
        "mean_residual_cycles": 2.3333333333333335,
    }
    assert summary["by_true_rul_bin"]["low"]["count"] == 1
    assert summary["by_true_rul_bin"]["low"]["mean_residual_cycles"] == 4.0
    assert summary["by_true_rul_bin"]["medium"]["mean_residual_cycles"] == -2.0
    assert summary["by_true_rul_bin"]["high"]["mean_residual_cycles"] == 5.0


@pytest.mark.parametrize("cycles", [-1, 1.0, True])
def test_true_rul_bins_reject_negative_or_noninteger_labels(cycles):
    with pytest.raises(ValueError, match="nonnegative integer"):
        rul_bin(cycles)
