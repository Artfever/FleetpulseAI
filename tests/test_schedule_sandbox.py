"""Exercise the live sandbox against the worked demo and a saved fleet."""

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.jobs import MaintenanceJob
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services.schedule_comparison import compare_schedules
from fleetpulse.ui_schedule_sandbox import DEMO_BASELINE


def _planner():
    app = AppTest.from_file(str(PROJECT_ROOT / "app/Home.py"), default_timeout=45).run()
    app.switch_page("pages/4_Maintenance_Planner.py")
    app.query_params["sandbox"] = "1"
    return app


def _scores(app):
    return {metric.label: metric.value for metric in app.metric}


def test_demo_compares_capacity_and_trial_weights_against_the_fixed_reference():
    app = _planner().run()

    assert not app.exception
    assert app.toggle(key="fp_schedule_sandbox").value
    assert _scores(app)["EDD weighted tardiness"] == "6 → 1"
    assert _scores(app)["Genetic algorithm weighted tardiness"] == "2 → 1"
    changes = next(frame.value for frame in app.dataframe if "Deadline outcome" in frame.value)
    job_b = changes.set_index("Job").loc["Job B"]
    assert (job_b["Team before"], job_b["Team after"], job_b["Moved"]) == (1, 2, "Yes")
    assert job_b["Deadline outcome"] == "Now on time"

    baseline_hash = compare_schedules(DEMO_BASELINE).scenario_sha256
    app.session_state[f"fp_sandbox_weights_{baseline_hash}"] = {
        "edited_rows": {0: {"Trial weight": 3}},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert not app.exception
    assert _scores(app)["EDD weighted tardiness"] == "6 → 3"


def test_saved_fleet_trial_updates_deadlines_without_overwriting_the_original():
    app = _planner()
    scenario = SchedulingScenario(
        (MaintenanceJob(EngineIdentity("FD001/train", 4), 34.2, 1, 2),),
        1,
        2,
        6,
    )
    app.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY] = scenario
    app.run()
    next(radio for radio in app.radio if radio.label == "Baseline scenario").set_value(
        "Saved fleet scenario"
    ).run()
    next(widget for widget in app.number_input if widget.label == "Trial buffer (cycles)").set_value(
        35.0
    ).run()

    assert not app.exception
    assert _scores(app)["EDD weighted tardiness"] == "0 → 2"
    assert app.session_state[ui.P20_FLEET_SCENARIO_STATE_KEY] == scenario
    changes = next(frame.value for frame in app.dataframe if "Deadline outcome" in frame.value)
    assert changes.iloc[0]["Deadline outcome"] == "Newly late"
