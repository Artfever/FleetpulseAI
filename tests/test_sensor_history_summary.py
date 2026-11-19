"""The sensor trace summary is reusable outside the Streamlit page."""

import pytest

from fleetpulse.services.sensor_history import summarize_sensor_history


def test_sensor_history_summary_counts_strict_direction_reversals():
    result = summarize_sensor_history([1, 2, 3, 2, 1, 1, 2])

    assert result.pattern == "varies across observed cycles"
    assert (result.minimum, result.maximum) == (1.0, 3.0)
    assert result.direction_reversals == 1


def test_sensor_history_summary_reports_flat_trace_without_reversals():
    result = summarize_sensor_history([4.5, 4.5, 4.5])

    assert result.pattern == "flat across observed cycles"
    assert (result.minimum, result.maximum) == (4.5, 4.5)
    assert result.direction_reversals == 0


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_sensor_history_summary_rejects_empty_or_nonfinite_values(values):
    with pytest.raises(ValueError):
        summarize_sensor_history(values)
