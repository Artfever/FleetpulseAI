"""Paired planner service returns checked EDD/GA outputs for one snapshot."""

import math

import pytest

from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services import ScheduleComparisonResult, compare_schedules


def _appendix_b_scenario():
    return SchedulingScenario(appendix_b_jobs(), team_count=1, buffer_cycles=2, horizon_cycles=6)


def test_comparison_service_returns_same_scenario_feasibility_metrics_and_runtime():
    scenario = _appendix_b_scenario()
    comparison = compare_schedules(scenario)

    assert isinstance(comparison, ScheduleComparisonResult)
    assert comparison.scenario_sha256 == comparison.edd.report["scenario_sha256"]
    assert comparison.scenario_sha256 == comparison.ga.report["scenario_sha256"]
    assert comparison.edd.feasible is comparison.ga.feasible is True
    assert comparison.edd.weighted_tardiness_cycles == 6
    assert comparison.ga.weighted_tardiness_cycles == 2
    assert comparison.edd.late_jobs == (
        ("synthetic/appendix-b-v1", 1),
        ("synthetic/appendix-b-v1", 2),
    )
    assert comparison.ga.late_jobs == (("synthetic/appendix-b-v1", 1),)
    assert comparison.edd.makespan_cycles == comparison.ga.makespan_cycles == 6
    assert comparison.edd.horizon_overflow_cycles == comparison.ga.horizon_overflow_cycles == 0
    assert math.isfinite(comparison.edd.runtime_seconds) and comparison.edd.runtime_seconds >= 0
    assert math.isfinite(comparison.ga.runtime_seconds) and comparison.ga.runtime_seconds >= 0
    assert comparison.ga.search is not None
    assert (comparison.ga.search.population_size, comparison.ga.search.generations) == (16, 20)


def test_comparison_service_repeats_schedule_outputs_for_same_seed_and_snapshot():
    scenario = _appendix_b_scenario()
    first = compare_schedules(scenario, population_size=8, generations=5, seed=19)
    second = compare_schedules(scenario, population_size=8, generations=5, seed=19)

    assert first.scenario_sha256 == second.scenario_sha256
    assert first.edd.report == second.edd.report
    assert first.ga.report == second.ga.report
    assert first.ga.search.order == second.ga.search.order
    assert first.ga.search.best_penalties == second.ga.search.best_penalties


def test_comparison_service_rejects_invalid_scenario_or_ga_budget():
    with pytest.raises(TypeError, match="SchedulingScenario"):
        compare_schedules(None)
    with pytest.raises(ValueError, match="population_size"):
        compare_schedules(_appendix_b_scenario(), population_size=1)
