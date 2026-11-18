"""Build one immutable maintenance scenario from a fleet prediction snapshot.

The planner consumes the nonnegative, unrounded display estimate already
carried by each successful fleet result. The raw estimate remains on the
original result for model evaluation and is never overwritten. Durations,
weights, capacity, buffer, and horizon are separate user-supplied simulation
assumptions. Pass the returned scenario unchanged to each scheduling method.

This interface is proposed for P20-A1 review with Shaheer and Taha; team
agreement is not recorded in this module yet.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

from fleetpulse.data.split import EngineIdentity as SchedulingIdentity
from fleetpulse.optimization.jobs import MaintenanceJob
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services.fleet import EngineIdentity, FleetPredictionResult


@dataclass(frozen=True)
class PlannerJobAssumption:
    """Explicit positive synthetic service duration and priority for one engine."""

    duration_cycles: int
    priority_weight: int


def build_planning_scenario(
    predictions: Iterable[FleetPredictionResult],
    assumptions: Mapping[EngineIdentity, PlannerJobAssumption],
    *,
    team_count: int,
    buffer_cycles: float,
    horizon_cycles: float,
) -> SchedulingScenario:
    """Validate one fleet snapshot and pair it with declared scenario settings.

    Every prediction must be successful and belong to the same saved model
    version. Assumptions must cover exactly those dataset-qualified engine
    identities. Model and raw-estimate fields on the input results are left
    untouched; returned jobs contain only planner RUL and assumed duration and
    weight. The scenario is safe to share with both EDD and GA.
    """
    if isinstance(team_count, bool) or not isinstance(team_count, Integral) or team_count < 1:
        raise ValueError("team_count must be a positive integer.")
    try:
        results = tuple(predictions)
    except TypeError as error:
        raise TypeError("predictions must be an iterable of FleetPredictionResult values.") from error
    if not results:
        raise ValueError("At least one fleet prediction is required.")
    if any(not isinstance(result, FleetPredictionResult) for result in results):
        raise TypeError("Every prediction must be a FleetPredictionResult.")

    by_identity = {result.identity: result for result in results}
    if len(by_identity) != len(results):
        raise ValueError("Fleet prediction identities must be unique.")

    model_versions = set()
    jobs = []
    for identity, result in sorted(by_identity.items()):
        if result.error or result.error_codes:
            raise ValueError(f"Engine {identity} has no usable prediction: {result.error or result.error_codes}.")
        if not isinstance(result.model_name, str) or not result.model_name.strip():
            raise ValueError(f"Engine {identity} has no model name.")
        if not isinstance(result.model_version, str) or not result.model_version.strip():
            raise ValueError(f"Engine {identity} has no model version.")
        model_versions.add((result.model_name, result.model_version))

        rul = result.displayed_rul_cycles
        if isinstance(rul, bool) or not isinstance(rul, Real) or not math.isfinite(rul) or rul < 0:
            raise ValueError(f"Engine {identity} needs a finite nonnegative planner RUL estimate.")
        if isinstance(result.observed_cycle, bool) or not isinstance(result.observed_cycle, Integral) or result.observed_cycle < 1:
            raise ValueError(f"Engine {identity} needs a positive observed cycle.")
        setting = assumptions.get(identity)
        if not isinstance(setting, PlannerJobAssumption):
            raise TypeError(f"Engine {identity} needs explicit duration and priority assumptions.")
        jobs.append(
            MaintenanceJob(
                SchedulingIdentity(identity.dataset_partition, identity.unit_id),
                float(rul),
                setting.duration_cycles,
                setting.priority_weight,
            )
        )

    if len(model_versions) != 1:
        raise ValueError("A planning snapshot must use one saved model name and version.")
    if set(assumptions) != set(by_identity):
        raise ValueError("Assumptions must match the prediction snapshot identities exactly.")
    return SchedulingScenario(tuple(jobs), team_count, buffer_cycles, horizon_cycles)
