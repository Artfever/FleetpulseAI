"""Exercise the planner's synthetic scenario form through Streamlit."""

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui


def test_planner_saves_reviewed_defaults_and_edits_without_a_fleet():
    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    assert not page.exception
    assert not page.dataframe
    page.button[0].click().run()
    assert not page.exception
    state = page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY]
    assert (state.team_count, state.buffer_cycles, state.horizon_cycles) == (1, 2, 6)
    assert page.dataframe[0].value["Priority weight"].tolist() == [1, 5, 1]
    assert page.dataframe[0].value["Duration (cycles)"].tolist() == [3, 1, 2]
    page.number_input[0].set_value(2)
    page.number_input[1].set_value(1.5)
    page.number_input[4].set_value(4)
    page.button[0].click().run()
    assert not page.exception
    state = page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY]
    assert (state.team_count, state.buffer_cycles, state.jobs[0].duration_cycles) == (2, 1.5, 4)
    assert page.dataframe[0].value["Assumed RUL (cycles)"].tolist() == [4.0, 5.0, 8.0]

    assert page.dataframe[1].value["Finish"].tolist() == [4, 1, 3]


def test_planner_edd_shows_reviewed_schedule_and_keeps_overflow_visible():
    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    page.number_input[2].set_value(2.0)
    page.button[0].click().run()
    assert not page.exception
    table = page.dataframe[1].value
    assert table["Engine"].tolist() == ["A", "B", "C"]
    assert table["Start"].tolist() == [0, 3, 4]
    assert table["Finish"].tolist() == [3, 4, 6]
    assert table["Deadline"].tolist() == [2, 3, 6]
    assert table["Past horizon"].tolist() == [1, 2, 4]
    assert [metric.value for metric in page.metric] == ["6", "6", "4"]
    assert len(page.get("plotly_chart")) == 1
