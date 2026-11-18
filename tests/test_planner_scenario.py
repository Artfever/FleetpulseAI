"""Exercise both planner schedules from one saved synthetic scenario."""

import json

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui


def _page():
    return AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/4_Maintenance_Planner.py"), default_timeout=30
    ).run()


def test_planner_compares_both_methods_on_the_same_reviewed_scenario():
    page = _page()
    assert not page.exception
    assert not page.dataframe
    page.button[0].click().run()
    assert not page.exception

    scenario = page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY]
    assert (scenario.team_count, scenario.buffer_cycles, scenario.horizon_cycles) == (1, 2, 6)
    assert page.dataframe[0].value["Priority weight"].tolist() == [1, 5, 1]
    assert page.dataframe[0].value["Duration (cycles)"].tolist() == [3, 1, 2]
    edd_table = page.dataframe[1].value
    ga_table = page.dataframe[2].value
    assert edd_table["Engine"].tolist() == ["A", "B", "C"]
    assert ga_table["Engine"].tolist() == ["B", "A", "C"]
    edd_deadlines = dict(zip(edd_table["Engine"], edd_table["Deadline (cycles)"], strict=True))
    ga_deadlines = dict(zip(ga_table["Engine"], ga_table["Deadline (cycles)"], strict=True))
    assert edd_deadlines == ga_deadlines
    assert edd_table["Beyond horizon (cycles)"].tolist() == [0, 0, 0]
    assert ga_table["Beyond horizon (cycles)"].tolist() == [0, 0, 0]
    comparison = page.session_state[ui.P20_SYNTHETIC_COMPARISON_STATE_KEY]
    assert comparison["edd"].weighted_tardiness_cycles == 6
    assert comparison["ga"].schedule.weighted_tardiness_cycles == 2
    assert [metric.label for metric in page.metric] == [
        "Weighted tardiness (weighted cycles)",
        "Total tardiness (cycles)",
        "Makespan (cycles)",
        "Horizon overflow (cycles)",
        "Late jobs",
    ] * 2
    charts = page.get("plotly_chart")
    assert len(charts) == 3
    assert all(chart.proto.spec for chart in charts)


def test_planner_keeps_ga_controls_bounded_and_uses_them_for_the_comparison():
    from fleetpulse.optimization.ga import MAX_GENERATIONS, MAX_POPULATION_SIZE, MAX_SEED

    page = _page()
    fields = {widget.label: widget for widget in page.number_input}
    assert fields["Population size"].max == MAX_POPULATION_SIZE
    assert fields["Generations"].max == MAX_GENERATIONS
    assert fields["Random seed"].max == MAX_SEED
    fields["Population size"].set_value(2)
    fields["Generations"].set_value(12)
    fields["Random seed"].set_value(1)
    page.button[0].click().run()
    assert not page.exception
    result = page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY]
    assert (result.population_size, result.generations, result.seed) == (2, 12, 1)
    assert page.dataframe[1].value["Engine"].tolist() == ["A", "B", "C"]
    assert page.dataframe[2].value["Engine"].tolist() == ["B", "A", "C"]
    chart = json.loads(page.get("plotly_chart")[2].proto.spec)
    assert chart["data"][0]["x"] == list(range(13))
    assert chart["data"][0]["y"] == [6.0] + [2.0] * 12
    assert chart["layout"]["shapes"][0]["y0"] == 6
    assert any("seed 1, population 2, 12 generations" in item.value for item in page.caption)


def test_planner_can_compare_single_selected_engine():
    page = _page()
    page.multiselect[0].set_value(["B"]).run()
    page.button[0].click().run()
    assert not page.exception
    assert page.dataframe[1].value["Engine"].tolist() == ["B"]
    assert page.dataframe[2].value["Engine"].tolist() == ["B"]
    assert page.dataframe[1].value["Finish (cycles)"].tolist() == [1]
    result = page.session_state[ui.P19_GENETIC_SEARCH_STATE_KEY]
    assert result.evaluations == 1 and result.best_penalties == (0.0,) * 21


def test_planner_ties_follow_engine_id_and_invalid_capacity_keeps_both_results():
    page = _page()
    page.multiselect[0].set_value(["C", "B", "A"]).run()
    for field in page.number_input:
        if field.label.endswith("assumed RUL (cycles)"):
            field.set_value(4.0)
    page.button[0].click().run()
    assert not page.exception
    assert page.dataframe[1].value["Engine"].tolist() == ["A", "B", "C"]
    assert page.dataframe[1].value["Deadline (cycles)"].tolist() == [2.0] * 3
    assert page.dataframe[2].value["Deadline (cycles)"].tolist() == [2.0] * 3
    saved = page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY]
    comparison = page.session_state[ui.P20_SYNTHETIC_COMPARISON_STATE_KEY]
    page.number_input[0].set_value(0)
    page.button[0].click().run()
    assert not page.exception
    assert [error.value for error in page.error] == ["Enter at least one maintenance team."]
    assert page.session_state[ui.P18_SYNTHETIC_SCENARIO_STATE_KEY] == saved
    assert page.session_state[ui.P20_SYNTHETIC_COMPARISON_STATE_KEY] == comparison


def test_planner_rejects_an_empty_job_selection():
    page = _page()
    page.multiselect[0].set_value([]).run()
    page.button[0].click().run()
    assert not page.exception
    assert [error.value for error in page.error] == ["Select at least one engine."]
    assert not page.dataframe
