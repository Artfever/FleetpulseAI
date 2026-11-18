"""Check reproducible search results and independently reconcile weighted tardiness."""

from itertools import pairwise
from math import fsum

import pytest

from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario


@pytest.mark.parametrize("teams", [1, 2])
def test_fixed_seed_retains_best_score_and_reconciles_returned_schedule(teams):
    scenario = SchedulingScenario(appendix_b_jobs(), teams, 2, 2)
    settings = {"population_size": 2, "generations": 12, "seed": 1}
    first = schedule_ga(scenario, **settings)
    second = schedule_ga(scenario, **settings)
    assert first.order == second.order
    assert first.schedule == second.schedule
    assert first.best_penalties == second.best_penalties
    assert first.evaluations == second.evaluations
    assert len(first.best_penalties) == 13
    assert first.best_penalties[0] <= first.edd_penalty
    assert all(new <= old for old, new in pairwise(first.best_penalties))

    by_engine = {job.engine: job for job in scenario.jobs}
    assert set(first.order) == set(by_engine)
    penalties = []
    for row in first.schedule.scheduled_jobs:
        job = by_engine[row.job.engine]
        deadline = max(0, job.predicted_rul_cycles - scenario.buffer_cycles)
        completion = row.timing.start_cycle + job.duration_cycles
        penalties.append(job.priority_weight * max(0, completion - deadline))
    assert fsum(penalties) == first.best_penalties[-1] == first.schedule.weighted_tardiness_cycles
    for team in range(1, teams + 1):
        intervals = sorted(
            (row.timing.start_cycle, row.timing.completion_cycle)
            for row in first.schedule.scheduled_jobs
            if row.team_id == team
        )
        assert all(left[1] <= right[0] for left, right in pairwise(intervals))

    if teams == 1:
        # A genuine improvement after initialization, retained for the remaining generations.
        assert first.best_penalties == (6.0, *(2.0 for _ in range(12)))
        assert first.order == (
            scenario.jobs[1].engine,
            scenario.jobs[0].engine,
            scenario.jobs[2].engine,
        )
