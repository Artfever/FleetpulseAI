"""Exercise genetic operators on independently specified orders and penalties."""

from random import Random

import pytest

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.ga import breed_population, order_crossover, swap_mutation

ORDERS = tuple(EngineIdentity("synthetic/operators", i) for i in range(1, 7))


class FixedSegment(Random):
    def sample(self, population, k, *, counts=None):
        return [1, 4]


def test_crossover_keeps_segment_and_second_parents_relative_order():
    first = ORDERS
    second = first[::-1]
    child = order_crossover(first, second, FixedSegment())
    assert [engine.unit_id for engine in child] == [6, 2, 3, 4, 5, 1]
    with pytest.raises(ValueError, match="same engine identities"):
        order_crossover(first, (first[0],) * len(first), Random(7))


def test_swap_mutation_changes_two_positions_without_changing_membership():
    mutated = swap_mutation(ORDERS, Random(42))
    assert sum(a != b for a, b in zip(mutated, ORDERS, strict=True)) == 2
    assert set(mutated) == set(ORDERS)
    assert swap_mutation(ORDERS[:1], Random(42)) == ORDERS[:1]


def test_generation_retains_elite_and_is_repeatable_with_population_of_two():
    population = (ORDERS, ORDERS[::-1])
    costs = (6.0, 2.0)
    result = breed_population(population, costs, rng=Random(42), mutation_rate=1)
    assert result[0] == population[1]
    assert len(result) == 2
    assert set(result[1]) == set(ORDERS) and result[1] != result[0]
    assert result == breed_population(population, costs, rng=Random(42), mutation_rate=1)
    assert (
        breed_population(population, costs, rng=Random(42), mutation_rate=0) == (population[1],) * 2
    )


@pytest.mark.parametrize(
    ("penalties", "rate"),
    [((0.0,), 0.2), ((0.0, float("nan")), 0.2), ((0.0, -1.0), 0.2), ((0.0, 1.0), 1.1)],
)
def test_generation_rejects_invalid_scores_or_mutation_rate(penalties, rate):
    with pytest.raises(ValueError):
        breed_population((ORDERS, ORDERS[::-1]), penalties, rng=Random(42), mutation_rate=rate)
