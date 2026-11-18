"""Compare the bounded search with every order in the hand-checked tiny case."""

from itertools import permutations

import pytest

from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario


# Lexicographic orders are ABC, ACB, BAC, BCA, CAB, CBA.
@pytest.mark.parametrize(
    ("teams", "expected"), [(1, (6, 16, 2, 4, 18, 4)), (2, (1, 1, 1, 2, 1, 2))]
)
def test_exhaustive_appendix_b_reference(teams, expected):
    scenario = SchedulingScenario(appendix_b_jobs(), teams, 2, 6)
    orders = tuple(permutations(job.engine for job in scenario.jobs))
    scores = tuple(decode_order(scenario, order).weighted_tardiness_cycles for order in orders)
    assert scores == expected
    result = schedule_ga(scenario)
    exact_minimum = min(scores)
    assert result.schedule.weighted_tardiness_cycles == exact_minimum
    assert scores[orders.index(result.order)] == exact_minimum


def test_small_budget_can_finish_with_a_feasible_nonoptimal_order():
    scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    result = schedule_ga(scenario, population_size=2, generations=12, seed=3)
    assert result.schedule == decode_order(scenario, result.order)
    assert result.best_penalties[-1] == result.schedule.weighted_tardiness_cycles == 4
    assert 2 < result.schedule.weighted_tardiness_cycles < result.edd_penalty == 6
