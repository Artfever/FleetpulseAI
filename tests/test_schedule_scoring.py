"""Independent Appendix B arithmetic and hard scheduling-rule checks."""

from dataclasses import replace

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.jobs import MaintenanceJob, appendix_b_jobs
from fleetpulse.optimization.scoring import (
    JobAssignment,
    ScheduleFeasibilityError,
    SchedulingScenario,
    evaluate_schedule,
)


def scenario(**changes):
    return SchedulingScenario(
        appendix_b_jobs(), **{"team_count": 1, "buffer_cycles": 2, "horizon_cycles": 6, **changes}
    )


def assignments(order=(0, 1, 2)):
    jobs = appendix_b_jobs()
    start = 0
    result = []
    for index in order:
        result.append(JobAssignment(jobs[index].engine, 1, start))
        start += jobs[index].duration_cycles
    return tuple(result)


@pytest.mark.parametrize(
    "order,penalty,tardiness", [((0, 1, 2), 6, (1, 1, 0)), ((1, 0, 2), 2, (0, 2, 0))]
)
def test_appendix_b_weighted_tardiness_is_hand_calculated(order, penalty, tardiness):
    result = evaluate_schedule(scenario(), assignments(order))
    assert result.weighted_tardiness_cycles == penalty
    assert tuple(entry.timing.tardiness_cycles for entry in result.scheduled_jobs) == tardiness
    assert result.total_tardiness_cycles == 2
    assert result.late_job_count == (2 if penalty == 6 else 1)
    assert result.makespan_cycles == 6
    assert result.horizon_overflow_cycles == 0 and result.overflow_job_count == 0
    assert len(result.scheduled_jobs) == 3


def test_soft_horizon_reports_overflow_without_dropping_or_changing_objective():
    result = evaluate_schedule(scenario(horizon_cycles=2), assignments())
    assert result.weighted_tardiness_cycles == 6
    assert result.makespan_cycles == 6 and result.horizon_overflow_cycles == 4
    assert result.overflow_job_count == 3
    assert [entry.horizon_overflow_cycles for entry in result.scheduled_jobs] == [1, 2, 4]
    assert len(result.scheduled_jobs) == 3


def test_identical_parallel_teams_allow_concurrency_and_touching_intervals():
    jobs = appendix_b_jobs()
    allocated = (
        JobAssignment(jobs[0].engine, 1, 0),
        JobAssignment(jobs[1].engine, 2, 0),
        JobAssignment(jobs[2].engine, 2, 1),
    )
    result = evaluate_schedule(scenario(team_count=2, horizon_cycles=2), allocated)
    assert result.weighted_tardiness_cycles == 1
    assert result.makespan_cycles == 3 and result.horizon_overflow_cycles == 1
    assert result.overflow_job_count == 2
    assert result == evaluate_schedule(scenario(team_count=2, horizon_cycles=2), allocated)
    assert (
        evaluate_schedule(scenario(team_count=2), reversed(allocated)).weighted_tardiness_cycles
        == 1
    )


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown", "capacity", "overlap"])
def test_infeasible_schedules_never_receive_an_objective(case):
    allocated = list(assignments())
    if case == "missing":
        allocated.pop()
    elif case == "duplicate":
        allocated[1] = allocated[0]
    elif case == "unknown":
        allocated[0] = replace(allocated[0], engine=EngineIdentity("other", 1))
    elif case == "capacity":
        allocated[0] = replace(allocated[0], team_id=2)
    else:
        allocated[1] = replace(allocated[1], start_cycle=2)
    with pytest.raises(ScheduleFeasibilityError):
        evaluate_schedule(scenario(), allocated)


@pytest.mark.parametrize(
    "changes",
    [
        {"team_count": 0},
        {"team_count": True},
        {"team_count": 1.5},
        {"buffer_cycles": -1},
        {"buffer_cycles": float("nan")},
        {"horizon_cycles": 0},
        {"horizon_cycles": float("inf")},
    ],
)
def test_invalid_scenario_parameters_fail(changes):
    with pytest.raises(ValueError):
        scenario(**changes)


def test_empty_and_duplicate_job_sets_are_rejected():
    with pytest.raises(ValueError):
        SchedulingScenario((), 1, 2, 6)
    jobs = appendix_b_jobs()
    with pytest.raises(ValueError):
        SchedulingScenario((jobs[0], jobs[0]), 1, 2, 6)


@pytest.mark.parametrize("team,start", [(0, 0), (True, 0), (1, -1), (1, 1.5), (1, True)])
def test_invalid_assignment_parameters_fail(team, start):
    with pytest.raises(ValueError):
        JobAssignment(appendix_b_jobs()[0].engine, team, start)


def test_same_numeric_unit_in_distinct_partitions_is_not_a_duplicate():
    first = MaintenanceJob(EngineIdentity("train", 1), 4, 1, 1)
    second = MaintenanceJob(EngineIdentity("test", 1), 4, 1, 1)
    example = SchedulingScenario((first, second), 1, 2, 6)
    result = evaluate_schedule(
        example, (JobAssignment(first.engine, 1, 0), JobAssignment(second.engine, 1, 1))
    )
    assert len(result.scheduled_jobs) == 2 and result.weighted_tardiness_cycles == 0


def test_fractional_due_times_keep_precision_and_priority_weight():
    job = MaintenanceJob(EngineIdentity("synthetic/fractional", 1), 2.75, 2, 3)
    result = evaluate_schedule(
        SchedulingScenario((job,), 1, 1.5, 3), (JobAssignment(job.engine, 1, 0),)
    )
    assert result.scheduled_jobs[0].timing.due_completion_cycle == 1.25
    assert result.total_tardiness_cycles == 0.75
    assert result.weighted_tardiness_cycles == 2.25


def test_objective_overflow_is_rejected_instead_of_reporting_infinite_score():
    job = MaintenanceJob(EngineIdentity("synthetic/overflow", 1), 0, 1, 10**400)
    with pytest.raises(ValueError, match="finite numeric range"):
        evaluate_schedule(SchedulingScenario((job,), 1, 0, 6), (JobAssignment(job.engine, 1, 0),))
