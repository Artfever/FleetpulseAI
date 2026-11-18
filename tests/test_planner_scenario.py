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
