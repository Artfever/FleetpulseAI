"""Earliest-due-date baseline using the shared decoder and objective."""

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.decoder import decode_order
from fleetpulse.optimization.jobs import calculate_job_timing
from fleetpulse.optimization.scoring import ScheduleEvaluation, SchedulingScenario


def edd_order(scenario: SchedulingScenario) -> tuple[EngineIdentity, ...]:
    """Sort exact completion deadlines, then dataset-qualified engine identity.

    This deadline-only baseline does not optimize the weighted tardiness
    objective. It deliberately ignores priority/duration when choosing order.
    """
    if not isinstance(scenario, SchedulingScenario):
        raise TypeError("scenario must be a SchedulingScenario.")
    return tuple(
        job.engine
        for job in sorted(
            scenario.jobs,
            key=lambda job: (
                calculate_job_timing(
                    job, buffer_cycles=scenario.buffer_cycles
                ).due_completion_cycle,
                job.engine.dataset_partition,
                job.engine.unit_id,
            ),
        )
    )


def schedule_edd(scenario: SchedulingScenario) -> ScheduleEvaluation:
    """Decode the EDD ordering and return the same validated score/report as GA."""
    return decode_order(scenario, edd_order(scenario))
