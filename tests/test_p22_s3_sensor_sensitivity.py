from __future__ import annotations

import pytest

from scripts.run_p22_s3_sensor_sensitivity import perturb_sensor_history, summarize_deltas


def test_perturbation_changes_only_selected_sensor_inside_window() -> None:
    rows = [
        {"unit_id": 7, "cycle": 1, "sensor_2": 10.0, "sensor_3": -4.0},
        {"unit_id": 7, "cycle": 2, "sensor_2": 20.0, "sensor_3": -8.0},
        {"unit_id": 7, "cycle": 3, "sensor_2": 30.0, "sensor_3": -12.0},
    ]

    changed = perturb_sensor_history(
        rows,
        sensor="sensor_2",
        relative_change=0.05,
        first_affected_cycle=2,
    )

    assert changed == [
        rows[0],
        {"unit_id": 7, "cycle": 2, "sensor_2": 21.0, "sensor_3": -8.0},
        {"unit_id": 7, "cycle": 3, "sensor_2": 31.5, "sensor_3": -12.0},
    ]
    assert rows[1]["sensor_2"] == 20.0


@pytest.mark.parametrize("bad_sensor", ["setting_1", "sensor_22", "true_rul"])
def test_perturbation_rejects_non_sensor_channels(bad_sensor: str) -> None:
    with pytest.raises(ValueError, match="declared sensor"):
        perturb_sensor_history(
            [{"cycle": 1, bad_sensor: 1.0}],
            sensor=bad_sensor,
            relative_change=0.01,
            first_affected_cycle=1,
        )


@pytest.mark.parametrize("bad_change", [0.0, 0.06, -0.06, float("nan"), True])
def test_perturbation_rejects_unbounded_or_invalid_changes(bad_change: float) -> None:
    with pytest.raises(ValueError, match="within ±5%"):
        perturb_sensor_history(
            [{"cycle": 1, "sensor_1": 1.0}],
            sensor="sensor_1",
            relative_change=bad_change,
            first_affected_cycle=1,
        )


def test_delta_summary_reports_prediction_movement_not_error() -> None:
    assert summarize_deltas([0.0, 2.0, -4.0]) == {
        "engine_count": 3,
        "mean_absolute_prediction_change_cycles": 2.0,
        "median_absolute_prediction_change_cycles": 2.0,
        "max_absolute_prediction_change_cycles": 4.0,
        "engines_with_changed_prediction": 2,
    }
