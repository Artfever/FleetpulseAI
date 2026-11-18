"""Deterministic earliest-available-team decoding shared by EDD and GA."""

from collections.abc import Iterable
from heapq import heapify, heappop, heappush
from itertools import islice

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.scoring import (
    JobAssignment,
    ScheduleEvaluation,
    ScheduleFeasibilityError,
    SchedulingScenario,
    evaluate_schedule,
)


def decode_order(
    scenario: SchedulingScenario,
    ordering: Iterable[EngineIdentity],
) -> ScheduleEvaluation:
    """Dispatch each job to earliest availability, tie-breaking by team ID.

    Ordering must contain every selected dataset-qualified engine exactly once.
    Teams begin at relative cycle zero. Jobs are contiguous and non-preemptive;
    none are discarded for missing a deadline or overflowing the soft horizon.
    Shared scoring validates the generated schedule before it is returned.
    """
    if not isinstance(scenario, SchedulingScenario):
        raise TypeError("scenario must be a SchedulingScenario.")
    try:
        # One extra value is sufficient to reject oversized/infinite orderings.
        order = tuple(islice(iter(ordering), len(scenario.jobs) + 1))
    except TypeError as error:
        raise TypeError(
            "ordering must be an iterable of dataset-qualified engine identities."
        ) from error
    if any(not isinstance(engine, EngineIdentity) for engine in order):
        raise ScheduleFeasibilityError(
            "Ordering must contain dataset-qualified EngineIdentity values."
        )
    by_engine = {job.engine: job for job in scenario.jobs}
    if len(order) != len(by_engine) or set(order) != set(by_engine):
        raise ScheduleFeasibilityError("Ordering must contain every selected engine exactly once.")
    # At most n teams can receive n jobs; larger IDs would remain idle.
    available = [(0, team) for team in range(1, min(scenario.team_count, len(order)) + 1)]
    heapify(available)
    assignments = []
    for engine in order:
        start, team = heappop(available)
        assignments.append(JobAssignment(engine, team, start))
        heappush(available, (start + by_engine[engine].duration_cycles, team))
    return evaluate_schedule(scenario, assignments)
