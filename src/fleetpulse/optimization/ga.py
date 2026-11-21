"""Permutation-based genetic scheduling on the shared scheduling contract."""

from dataclasses import dataclass
from math import isfinite
from random import Random
from time import perf_counter

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.edd import edd_order
from fleetpulse.optimization.scoring import ScheduleEvaluation, SchedulingScenario

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


def order_crossover(first: EngineOrder, second: EngineOrder, rng: Random) -> EngineOrder:
    """Keep a segment of the first parent and fill other slots in second-parent order."""
    if not first or len(first) != len(second) or len(set(first)) != len(first):
        raise ValueError("Parents must be nonempty permutations of the same engine identities.")
    identities = set(first)
    if set(second) != identities:
        raise ValueError("Parents must contain the same engine identities.")
    if len(first) == 1:
        return first
    start, stop = sorted(rng.sample(range(len(first) + 1), 2))
    segment = first[start:stop]
    retained = set(segment)
    remaining = tuple(engine for engine in second if engine not in retained)
    return remaining[:start] + segment + remaining[start:]


def swap_mutation(order: EngineOrder, rng: Random) -> EngineOrder:
    """Swap two distinct positions, or keep the sole position of a singleton."""
    if len(order) < 2:
        return order
    left, right = rng.sample(range(len(order)), 2)
    mutated = list(order)
    mutated[left], mutated[right] = mutated[right], mutated[left]
    return tuple(mutated)


def breed_population(
    population: tuple[EngineOrder, ...],
    penalties: tuple[float, ...],
    *,
    rng: Random,
    mutation_rate: float,
) -> tuple[EngineOrder, ...]:
    """Carry the best order forward, then fill the population by breeding."""
    if not population or len(population) != len(penalties):
        raise ValueError("Supply a nonempty population with one penalty per order.")
    if any(not isfinite(value) or value < 0 for value in penalties):
        raise ValueError("Penalties must be finite and nonnegative.")
    if isinstance(mutation_rate, bool) or not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between zero and one.")

    def select_parent() -> EngineOrder:
        competitors = rng.sample(range(len(population)), min(3, len(population)))
        return population[min(competitors, key=penalties.__getitem__)]

    elite_index = min(range(len(population)), key=penalties.__getitem__)
    next_population = [population[elite_index]]
    for _ in range(len(population) - 1):
        child = order_crossover(select_parent(), select_parent(), rng)
        if rng.random() < mutation_rate:
            child = swap_mutation(child, rng)
        next_population.append(child)
    return tuple(next_population)


DEFAULT_POPULATION_SIZE = 16
DEFAULT_GENERATIONS = 20
DEFAULT_SEED = 42
MAX_POPULATION_SIZE = 64
MAX_GENERATIONS = 100
MAX_SEED = 2**32 - 1
MUTATION_RATE = 0.2


@dataclass(frozen=True)
class GeneticSearchResult:
    order: EngineOrder
    schedule: ScheduleEvaluation
    edd_penalty: float
    best_penalties: tuple[float, ...]
    runtime_seconds: float
    evaluations: int
    population_size: int
    generations: int
    seed: int


def schedule_ga(
    scenario: SchedulingScenario,
    *,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    seed: int = DEFAULT_SEED,
) -> GeneticSearchResult:
    """Run the seeded search, recording the initial best and each generation's best.

    Repeated orders reuse their score within the call. Equal scores retain the
    earliest candidate, so an equally good alternative cannot displace the EDD
    seed unless the search first found an improvement.
    """
    for name, value, minimum, maximum in (
        ("population_size", population_size, 2, MAX_POPULATION_SIZE),
        ("generations", generations, 0, MAX_GENERATIONS),
        ("seed", seed, 0, MAX_SEED),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")

    start = perf_counter()
    rng = Random(seed)
    population = initialize_population(scenario, population_size=population_size, rng=rng)
    scores: dict[EngineOrder, ScheduleEvaluation] = {}
    best_order = population[0]
    edd_schedule = scores[best_order] = decode_order(scenario, best_order)
    best_schedule = edd_schedule
    history = []
    for generation in range(generations + 1):
        penalties = []
        for order in population:
            if order not in scores:
                scores[order] = decode_order(scenario, order)
            candidate = scores[order]
            penalties.append(candidate.weighted_tardiness_cycles)
            if candidate.weighted_tardiness_cycles < best_schedule.weighted_tardiness_cycles:
                best_order, best_schedule = order, candidate
        history.append(best_schedule.weighted_tardiness_cycles)
        if generation < generations:
            population = breed_population(
                population, tuple(penalties), rng=rng, mutation_rate=MUTATION_RATE
            )
    return GeneticSearchResult(
        order=best_order,
        schedule=best_schedule,
        edd_penalty=edd_schedule.weighted_tardiness_cycles,
        best_penalties=tuple(history),
        runtime_seconds=perf_counter() - start,
        evaluations=len(scores),
        population_size=population_size,
        generations=generations,
        seed=seed,
    )
