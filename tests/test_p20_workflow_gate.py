"""Exercise the real saved-model Fleet Overview to Maintenance Planner flow."""

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui


def test_real_saved_model_fleet_selection_reaches_both_schedulers():
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=60).run()
    fleet = app.switch_page("pages/1_Fleet_Overview.py").run()

    assert not fleet.exception
    predictions = fleet.dataframe[0].value
    assert predictions["Dataset"].tolist() == ["FD001/train"] * 3
    assert predictions["Engine ID"].tolist() == [4, 5, 12]
    assert predictions["Observed cycle"].tolist() == [164, 58, 7]
    assert predictions["Version"].tolist() == ["decision-tree-rul-v1"] * 3
    raw_rul_by_engine = dict(
        zip(predictions["Engine ID"], predictions["Raw RUL (cycles)"], strict=True)
    )
    assert raw_rul_by_engine == {
        4: 34.19607843137255,
        5: 158.16,
        12: 184.35714285714286,
    }

    fleet.multiselect[0].set_value([4, 5, 12]).run()
    fleet.button[0].click().run()
    assert not fleet.exception

    planner = app.switch_page("pages/4_Maintenance_Planner.py").run()
    assert not planner.exception
    planner.button[0].click().run()
    assert not planner.exception

    scenario = planner.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY]
    comparison = planner.session_state[ui.P20_FLEET_COMPARISON_STATE_KEY]
    artifact = planner.session_state[ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY]
    assert [job.engine.unit_id for job in scenario.jobs] == [4, 5, 12]
    assert [job.predicted_rul_cycles for job in scenario.jobs] == [
        raw_rul_by_engine[4],
        raw_rul_by_engine[5],
        raw_rul_by_engine[12],
    ]
    assert all((job.duration_cycles, job.priority_weight) == (1, 1) for job in scenario.jobs)
    assert (scenario.team_count, scenario.buffer_cycles, scenario.horizon_cycles) == (1, 2, 6)
    assert comparison.edd.feasible and comparison.ga.feasible
    assert comparison.scenario_sha256 == comparison.edd.report["scenario_sha256"]
    assert comparison.scenario_sha256 == comparison.ga.report["scenario_sha256"]
    assert comparison.edd.weighted_tardiness_cycles == 0
    assert comparison.ga.weighted_tardiness_cycles == 0
    assert comparison.edd.makespan_cycles == comparison.ga.makespan_cycles == 3
    assert comparison.edd.horizon_overflow_cycles == comparison.ga.horizon_overflow_cycles == 0
    assert artifact["scenario"]["scenario_sha256"] == comparison.scenario_sha256
    assert artifact["provenance"]["model"]["model_version"] == "decision-tree-rul-v1"
    assert [item.label for item in planner.get("download_button")] == [
        "Download paired fleet comparison JSON"
    ]
