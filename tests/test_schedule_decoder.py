"""Independent tiny decoder schedules and deterministic permutation boundaries."""

from dataclasses import replace
from itertools import repeat

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import ScheduleFeasibilityError, SchedulingScenario


def scenario(teams=1, horizon=6):
    return SchedulingScenario(appendix_b_jobs(), teams, 2, horizon)


def order(example):
    return tuple(job.engine for job in example.jobs)


@pytest.mark.parametrize(
    "indices,completions,cost", [((0, 1, 2), (3, 4, 6), 6), ((1, 0, 2), (1, 4, 6), 2)]
)
def test_one_team_matches_independent_appendix_b_worked_orders(indices, completions, cost):
    example = scenario()
    selected = tuple(order(example)[i] for i in indices)
    result = decode_order(example, iter(selected))
    assert tuple(row.job.engine for row in result.scheduled_jobs) == selected
    assert tuple(row.timing.completion_cycle for row in result.scheduled_jobs) == completions
    assert tuple(row.team_id for row in result.scheduled_jobs) == (1, 1, 1)
    assert result.weighted_tardiness_cycles == cost
    assert result == decode_order(example, selected)
    assert example == scenario()  # No input mutation.


def test_two_teams_choose_next_available_team_and_keep_soft_horizon_overflow():
    example = scenario(teams=2, horizon=2)
    result = decode_order(example, order(example))
    assert [
        (row.team_id, row.timing.start_cycle, row.timing.completion_cycle)
        for row in result.scheduled_jobs
    ] == [(1, 0, 3), (2, 0, 1), (2, 1, 3)]
    assert result.weighted_tardiness_cycles == 1
    assert result.makespan_cycles == 3 and result.horizon_overflow_cycles == 1
    assert result.overflow_job_count == 2 and len(result.scheduled_jobs) == 3


def test_equal_availability_ties_use_lowest_team_id():
    jobs = tuple(replace(job, duration_cycles=1) for job in appendix_b_jobs())
    jobs = (*jobs, replace(jobs[0], engine=EngineIdentity("synthetic/tie", 4)))
    example = SchedulingScenario(jobs, 2, 2, 6)
    result = decode_order(example, order(example))
    assert [(row.team_id, row.timing.start_cycle) for row in result.scheduled_jobs] == [
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 1),
    ]


def test_capacity_bigger_than_job_count_does_not_allocate_unbounded_idle_teams():
    example = scenario(teams=10**9)
    result = decode_order(example, order(example))
    assert [row.team_id for row in result.scheduled_jobs] == [1, 2, 3]
    assert all(row.timing.start_cycle == 0 for row in result.scheduled_jobs)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "extra", "unknown", "numeric_ids", "infinite"]
)
def test_invalid_order_is_rejected_before_assignment(case):
    example = scenario()
    selected = list(order(example))
    if case == "missing":
        selected.pop()
    elif case == "duplicate":
        selected[1] = selected[0]
    elif case == "extra":
        selected.append(selected[0])
    elif case == "unknown":
        selected[0] = EngineIdentity("different/partition", 1)
    elif case == "numeric_ids":
        selected = [1, 2, 3]
    else:
        selected = repeat(selected[0])
    with pytest.raises(ScheduleFeasibilityError):
        decode_order(example, selected)


def test_same_unit_in_two_partitions_remains_two_distinct_jobs():
    jobs = appendix_b_jobs()
    second = replace(jobs[0], engine=EngineIdentity("different/partition", 1))
    example = SchedulingScenario((jobs[0], second), 1, 2, 6)
    result = decode_order(example, order(example))
    assert [row.job.engine for row in result.scheduled_jobs] == list(order(example))
    assert len(result.scheduled_jobs) == 2


def test_non_iterable_order_has_clear_error():
    with pytest.raises(TypeError, match="ordering must be an iterable"):
        decode_order(scenario(), None)
