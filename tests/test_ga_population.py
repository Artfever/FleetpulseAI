"""Check EDD seeding and initial candidate feasibility on the reviewed example."""

from dataclasses import replace
from random import Random

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.ga import initialize_population
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario


def test_seeded_population_keeps_edd_and_every_candidate_decodes():
    jobs = appendix_b_jobs()
    scenario = SchedulingScenario(jobs, 1, 2, 6)
    population = initialize_population(scenario, population_size=16, rng=Random(42))
    assert len(population) == 16
    assert population[0] == tuple(job.engine for job in jobs)
    assert len(set(population)) > 1
    scores = [decode_order(scenario, order) for order in population]
    assert scores[0].weighted_tardiness_cycles == 6
    assert all(len(result.scheduled_jobs) == 3 for result in scores)
    reordered = replace(scenario, jobs=tuple(reversed(jobs)))
    assert population == initialize_population(reordered, population_size=16, rng=Random(42))


def test_qualified_ids_and_small_fleets_are_supported():
    job = appendix_b_jobs()[0]
    same_unit = replace(job, engine=EngineIdentity("synthetic/other", job.engine.unit_id))
    scenario = SchedulingScenario((job, same_unit), 2, 2, 6)
    population = initialize_population(scenario, population_size=8, rng=Random(7))
    assert all(
        len(order) == 2 and set(order) == {job.engine, same_unit.engine} for order in population
    )
    singleton = replace(scenario, jobs=(job,))
    assert (
        initialize_population(singleton, population_size=8, rng=Random(7)) == ((job.engine,),) * 8
    )


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_population_size_is_rejected(size):
    scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    with pytest.raises(ValueError, match="positive integer"):
        initialize_population(scenario, population_size=size, rng=Random(42))
