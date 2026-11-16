"""Validated job inputs for explicitly synthetic maintenance scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from fleetpulse.data.split import EngineIdentity


@dataclass(frozen=True)
class MaintenanceJob:
    """One engine's estimated remaining life and scenario-only service inputs.

    RUL and duration are measured in relative operating cycles. Priority is a
    dimensionless positive weight. Duration and priority are scenario
    assumptions, not measurements from NASA C-MAPSS or an aircraft fleet.
    """

    engine: EngineIdentity
    predicted_rul_cycles: float
    duration_cycles: int
    priority_weight: int

    def __post_init__(self) -> None:
        if not isinstance(self.engine, EngineIdentity):
            raise TypeError("engine must be a dataset-qualified EngineIdentity.")
        partition = self.engine.dataset_partition
        if not isinstance(partition, str) or not partition.strip():
            raise ValueError("engine dataset_partition must be a nonempty string.")
        if (
            isinstance(self.engine.unit_id, bool)
            or not isinstance(self.engine.unit_id, Integral)
            or self.engine.unit_id <= 0
        ):
            raise ValueError("engine unit_id must be a positive integer.")

        rul = self.predicted_rul_cycles
        if isinstance(rul, bool) or not isinstance(rul, Real):
            raise TypeError("predicted_rul_cycles must be a real number.")
        if not math.isfinite(rul) or rul < 0:
            raise ValueError("predicted_rul_cycles must be finite and nonnegative.")
        object.__setattr__(self, "predicted_rul_cycles", float(rul))

        for name, value in (
            ("duration_cycles", self.duration_cycles),
            ("priority_weight", self.priority_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))


def appendix_b_jobs() -> tuple[MaintenanceJob, ...]:
    """Return the three explicitly synthetic job inputs from plan Appendix B."""
    partition = "synthetic/appendix-b-v1"
    return (
        MaintenanceJob(EngineIdentity(partition, 1), 4, 3, 1),
        MaintenanceJob(EngineIdentity(partition, 2), 5, 1, 5),
        MaintenanceJob(EngineIdentity(partition, 3), 8, 2, 1),
    )


@dataclass(frozen=True)
class JobTiming:
    """Timing for one job; the due cycle refers to service completion."""

    engine: EngineIdentity
    due_completion_cycle: float
    start_cycle: int
    completion_cycle: int
    tardiness_cycles: float


def calculate_job_timing(
    job: MaintenanceJob, *, buffer_cycles: float, start_cycle: int = 0
) -> JobTiming:
    """Calculate a completion deadline and lateness in relative cycle units.

    ``due=max(0, predicted_rul-buffer)``, ``completion=start+duration``, and
    ``tardiness=max(0, completion-due)``. Buffer is a nonnegative scenario
    assumption. The default start is relative cycle zero; a scheduler may pass
    the actual nonnegative start when evaluating an ordering.
    """
    if not isinstance(job, MaintenanceJob):
        raise TypeError("job must be a MaintenanceJob.")
    if isinstance(buffer_cycles, bool) or not isinstance(buffer_cycles, Real):
        raise TypeError("buffer_cycles must be a real number.")
    if not math.isfinite(buffer_cycles) or buffer_cycles < 0:
        raise ValueError("buffer_cycles must be finite and nonnegative.")
    if (
        isinstance(start_cycle, bool)
        or not isinstance(start_cycle, Integral)
        or start_cycle < 0
    ):
        raise ValueError("start_cycle must be a nonnegative integer.")

    start = int(start_cycle)
    due = max(0.0, job.predicted_rul_cycles - float(buffer_cycles))
    completion = start + job.duration_cycles
    tardiness = max(0.0, completion - due)
    return JobTiming(job.engine, due, start, completion, tardiness)
