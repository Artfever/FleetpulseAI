"""Verify the bounded search through the shared scheduling interface."""

import pytest

from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario


def test_bounded_search_returns_feasible_schedule_and_records_budget():
    scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    result = schedule_ga(scenario)
    assert (result.seed, result.population_size, result.generations) == (42, 16, 20)
    assert len(result.best_penalties) == 21
    assert result.edd_penalty == 6
    assert result.schedule == decode_order(scenario, result.order)
    assert result.schedule.weighted_tardiness_cycles == result.best_penalties[-1] == 2
    assert 1 <= result.evaluations <= 6
    assert result.runtime_seconds >= 0


def test_initial_population_only_is_an_explicit_valid_budget():
    scenario = SchedulingScenario(appendix_b_jobs()[:1], 1, 2, 6)
    result = schedule_ga(scenario, population_size=2, generations=0, seed=0)
    assert result.order == (scenario.jobs[0].engine,)
    assert result.best_penalties == (result.edd_penalty,)
    assert result.evaluations == 1


@pytest.mark.parametrize(
    "options",
    [
        {"population_size": 1},
        {"population_size": 65},
        {"generations": -1},
        {"generations": 101},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
    ],
)
def test_budget_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        schedule_ga(SchedulingScenario(appendix_b_jobs(), 1, 2, 6), **options)
