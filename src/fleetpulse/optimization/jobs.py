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
