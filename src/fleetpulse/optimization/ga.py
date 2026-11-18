"""Permutation-based genetic scheduling on the shared scheduling contract."""

from random import Random

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import edd_order
from fleetpulse.optimization.scoring import SchedulingScenario

EngineOrder = tuple[EngineIdentity, ...]


def initialize_population(
    scenario: SchedulingScenario, *, population_size: int, rng: Random
) -> tuple[EngineOrder, ...]:
    """Return the EDD seed followed by permutations sampled with the supplied RNG.

    Sampling is independent for each candidate. Repeated candidates are valid,
    which lets the population retain its requested size for very small fleets.
    """
    if type(population_size) is not int or population_size < 1:
        raise ValueError("population_size must be a positive integer.")
    baseline = edd_order(scenario)
    return (
        baseline,
        *(tuple(rng.sample(baseline, len(baseline))) for _ in range(population_size - 1)),
    )
