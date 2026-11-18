"""Exercise the planner's synthetic scenario form through Streamlit."""

import json

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


def test_planner_submits_bounded_ga_controls_and_can_return_to_edd():
    from fleetpulse.optimization.ga import MAX_GENERATIONS, MAX_POPULATION_SIZE, MAX_SEED

    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    fields = {widget.label: widget for widget in page.number_input}
    assert fields["Population size"].max == MAX_POPULATION_SIZE
    assert fields["Generations"].max == MAX_GENERATIONS
    assert fields["Random seed"].max == MAX_SEED
    page.selectbox[0].select("Genetic Algorithm")
    fields["Population size"].set_value(2)
    fields["Generations"].set_value(12)
    fields["Random seed"].set_value(1)
    page.button[0].click().run()
    assert not page.exception
    result = page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY]
    assert (result.population_size, result.generations, result.seed) == (2, 12, 1)
    assert page.dataframe[1].value["Engine"].tolist() == ["B", "A", "C"]
    assert page.metric[0].value == "2"
    timeline, history = page.get("plotly_chart")
    assert timeline.proto.spec
    chart = json.loads(history.proto.spec)
    assert chart["data"][0]["x"] == list(range(13))
    assert chart["data"][0]["y"] == [6.0] + [2.0] * 12
    assert chart["layout"]["shapes"][0]["y0"] == 6
    assert any("Seed 1 ? Population 2 ? 12 generations" in item.value for item in page.markdown)
    page.run()
    assert page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY] == result
    page.selectbox[0].select("EDD")
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY] is None
    assert page.dataframe[1].value["Engine"].tolist() == ["A", "B", "C"]
    assert len(page.get("plotly_chart")) == 1


def test_planner_can_run_ga_with_one_selected_engine():
    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    page.multiselect[0].set_value(["B"]).run()
    page.selectbox[0].select("Genetic Algorithm")
    page.button[0].click().run()
    assert not page.exception
    assert page.dataframe[1].value["Engine"].tolist() == ["B"]
    assert page.dataframe[1].value["Finish"].tolist() == [1]
    result = page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY]
    assert result.evaluations == 1 and result.best_penalties == (0.0,) * 21


def test_planner_ties_follow_engine_id_and_invalid_capacity_keeps_previous_result():
    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    page.multiselect[0].set_value(["C", "B", "A"]).run()
    for field in page.number_input:
        if field.label.endswith("assumed RUL (cycles)"):
            field.set_value(4.0)
    page.button[0].click().run()
    assert not page.exception
    assert page.dataframe[1].value["Engine"].tolist() == ["A", "B", "C"]
    assert page.dataframe[1].value["Deadline"].tolist() == [2.0] * 3
    saved = page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY]
    page.number_input[0].set_value(0)
    page.button[0].click().run()
    assert not page.exception
    assert [error.value for error in page.error] == ["Enter at least one maintenance team."]
    assert page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY] == saved


def test_planner_rejects_an_empty_job_selection():
    page = AppTest.from_file(str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py")).run()
    page.multiselect[0].set_value([]).run()
    page.button[0].click().run()
    assert not page.exception
    assert [error.value for error in page.error] == ["Select at least one engine."]
    assert not page.dataframe
