"""UI-ready outputs must be complete, feasible and tied to their scenario."""

import json
from dataclasses import replace

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.reporting import build_schedule_report
from fleetpulse.optimization.scoring import ScheduleFeasibilityError, SchedulingScenario


def example():
    scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 2)
    return scenario, schedule_edd(scenario)


def test_report_contains_required_timing_input_units_and_verified_summary():
    scenario, schedule = example()
    report = build_schedule_report(scenario, schedule)
    assert report["schema_version"] == "p18-schedule-report-v1"
    assert report["feasibility_checked"] is True and report["time_unit"] == "relative_cycles"
    assert report["jobs"][0] == {
        "dataset_partition": "synthetic/appendix-b-v1",
        "unit_id": 1,
        "team_id": 1,
        "predicted_rul_cycles": 4.0,
        "duration_cycles": 3,
        "priority_weight": 1,
        "due_completion_cycle": 2.0,
        "start_cycle": 0,
        "completion_cycle": 3,
        "tardiness_cycles": 1.0,
        "horizon_overflow_cycles": 1.0,
    }
    assert [row["unit_id"] for row in report["jobs"]] == [1, 2, 3]
    assert [row["horizon_overflow_cycles"] for row in report["jobs"]] == [1.0, 2.0, 4.0]
    assert report["summary"] == {
        "team_count": 1,
        "buffer_cycles": 2.0,
        "horizon_cycles": 2.0,
        "scheduled_job_count": 3,
        "weighted_tardiness_cycles": 6.0,
        "total_tardiness_cycles": 2.0,
        "late_job_count": 2,
        "makespan_cycles": 6,
        "horizon_overflow_cycles": 4.0,
        "overflow_job_count": 3,
    }
    assert len(report["scenario_sha256"]) == 64
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report == build_schedule_report(scenario, schedule)


@pytest.mark.parametrize(
    "case",
    [
        "summary",
        "tardiness",
        "completion",
        "team",
        "overlap",
        "missing",
        "duplicate",
        "unknown",
        "source_job",
    ],
)
def test_forged_or_infeasible_result_is_rejected_before_presentation(case):
    scenario, schedule = example()
    rows = list(schedule.scheduled_jobs)
    if case == "summary":
        schedule = replace(schedule, weighted_tardiness_cycles=0.0)
    elif case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[1] = rows[0]
    elif case == "unknown":
        rows[0] = replace(rows[0], job=replace(rows[0].job, engine=EngineIdentity("unknown", 1)))
    elif case == "source_job":
        rows[0] = replace(rows[0], job=replace(rows[0].job, predicted_rul_cycles=99))
    elif case == "team":
        rows[0] = replace(rows[0], team_id=2)
    elif case == "overlap":
        rows[1] = replace(rows[1], timing=replace(rows[1].timing, start_cycle=2))
    elif case == "completion":
        rows[0] = replace(rows[0], timing=replace(rows[0].timing, completion_cycle=999))
    else:
        rows[0] = replace(rows[0], timing=replace(rows[0].timing, tardiness_cycles=0.0))
    if case != "summary":
        schedule = replace(schedule, scheduled_jobs=tuple(rows))
    with pytest.raises(ScheduleFeasibilityError):
        build_schedule_report(scenario, schedule)


def test_changed_scenario_cannot_reuse_stale_timing_score():
    scenario, schedule = example()
    changed = replace(scenario, buffer_cycles=3)
    with pytest.raises(ScheduleFeasibilityError):
        build_schedule_report(changed, schedule)
    fresh = build_schedule_report(changed, schedule_edd(changed))
    assert fresh["scenario_sha256"] != build_schedule_report(scenario, schedule)["scenario_sha256"]


def test_report_keeps_same_unit_in_different_partitions_distinct():
    jobs = appendix_b_jobs()
    second = replace(jobs[0], engine=EngineIdentity("another/partition", 1))
    scenario = SchedulingScenario((jobs[0], second), 2, 2, 6)
    report = build_schedule_report(scenario, schedule_edd(scenario))
    assert {(row["dataset_partition"], row["unit_id"]) for row in report["jobs"]} == {
        (jobs[0].engine.dataset_partition, 1),
        ("another/partition", 1),
    }


def test_untyped_schedule_is_rejected():
    scenario, _ = example()
    with pytest.raises(TypeError, match="ScheduleEvaluation"):
        build_schedule_report(scenario, {})


def test_numpy_integer_engine_ids_are_json_safe():
    import numpy as np

    from fleetpulse.optimization.jobs import MaintenanceJob

    job = MaintenanceJob(EngineIdentity("synthetic/table", np.int64(7)), 4, 1, 1)
    scenario = SchedulingScenario((job,), 1, 2, 6)
    report = build_schedule_report(scenario, schedule_edd(scenario))
    assert type(report["jobs"][0]["unit_id"]) is int
    assert json.loads(json.dumps(report, allow_nan=False))["jobs"][0]["unit_id"] == 7
