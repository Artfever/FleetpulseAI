"""Check the real EDD pipeline against the saved hand-calculated Appendix B schedule."""

import json
from pathlib import Path

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.jobs import MaintenanceJob, appendix_b_jobs
from fleetpulse.optimization.reporting import build_schedule_report
from fleetpulse.optimization.scoring import SchedulingScenario


def test_saved_appendix_b_scenario_matches_edd_and_planner_defaults():
    path = Path(__file__).with_name("fixtures") / "p18_appendix_b.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    jobs = tuple(
        MaintenanceJob(
            EngineIdentity(reference["dataset_partition"], row["unit_id"]),
            row["predicted_rul_cycles"],
            row["duration_cycles"],
            row["priority_weight"],
        )
        for row in reference["jobs"]
    )
    scenario = SchedulingScenario(
        jobs, reference["team_count"], reference["buffer_cycles"], reference["horizon_cycles"]
    )
    assert jobs == appendix_b_jobs()

    report = build_schedule_report(scenario, schedule_edd(scenario))
    expected = reference["expected_edd"]
    assert report["feasibility_checked"]
    assert len(report["jobs"]) == len(expected["jobs"])
    for row, oracle in zip(report["jobs"], expected["jobs"], strict=True):
        assert {key: row[key] for key in oracle} == oracle
    assert {key: report["summary"][key] for key in expected["summary"]} == expected["summary"]
    assert report == build_schedule_report(scenario, schedule_edd(scenario))
