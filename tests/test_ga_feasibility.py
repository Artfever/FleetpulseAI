"""Check permutation and capacity invariants independently of scoring."""

from collections import Counter, defaultdict
from itertools import pairwise
from random import Random

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.ga import order_crossover, swap_mutation
from fleetpulse.optimization.jobs import MaintenanceJob
from fleetpulse.optimization.scoring import SchedulingScenario


@pytest.mark.parametrize(("job_count", "team_count"), [(1, 1), (2, 3), (7, 1), (12, 4)])
def test_crossover_and_mutation_keep_every_job_and_decode_within_capacity(job_count, team_count):
    jobs = tuple(
        MaintenanceJob(
            EngineIdentity(f"synthetic/{index % 2}", index // 2 + 1),
            index + 2,
            index % 4 + 1,
            index % 3 + 1,
        )
        for index in range(job_count)
    )
    scenario = SchedulingScenario(jobs, team_count, 2, 2)
    identities = tuple(job.engine for job in jobs)
    expected = Counter(identities)

    for seed in range(50):
        rng = Random(seed)
        parents = [tuple(rng.sample(identities, job_count)) for _ in range(2)]
        child = order_crossover(*parents, rng)
        mutated = swap_mutation(child, rng)
        for order in (child, mutated):
            assert Counter(order) == expected
            result = decode_order(scenario, order)
            assert Counter(row.job.engine for row in result.scheduled_jobs) == expected
            intervals = defaultdict(list)
            for row in result.scheduled_jobs:
                assert 1 <= row.team_id <= team_count
                start, end = row.timing.start_cycle, row.timing.completion_cycle
                assert end - start == row.job.duration_cycles
                intervals[row.team_id].append((start, end))
            for team_intervals in intervals.values():
                assert all(
                    previous[1] <= current[0]
                    for previous, current in pairwise(sorted(team_intervals))
                )
