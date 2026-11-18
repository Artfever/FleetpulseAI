"""One synthetic scheduling objective and hard feasibility contract (P18-S3)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real

from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.jobs import JobTiming, MaintenanceJob, calculate_job_timing


class ScheduleFeasibilityError(ValueError):
    """The proposed schedule violates a hard assignment/capacity rule."""


def _integer(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer.")
    return int(value)


def _cycles(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return float(value)


@dataclass(frozen=True)
class SchedulingScenario:
    """Identical parallel teams and explicitly assumed service inputs.

    All time values use relative cycles. The horizon is a soft reporting
    target; duration/priority/buffer/capacity are simulation assumptions.
    """

    jobs: tuple[MaintenanceJob, ...]
    team_count: int
    buffer_cycles: float
    horizon_cycles: float

    def __post_init__(self):
        jobs = tuple(self.jobs)
        if not jobs or any(not isinstance(job, MaintenanceJob) for job in jobs):
            raise ValueError("jobs must contain at least one MaintenanceJob.")
        if len({job.engine for job in jobs}) != len(jobs):
            raise ValueError("Scenario jobs must have unique dataset-qualified engine identities.")
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "team_count", _integer(self.team_count, "team_count"))
        object.__setattr__(
            self, "buffer_cycles", _cycles(self.buffer_cycles, "buffer_cycles", zero=True)
        )
        object.__setattr__(self, "horizon_cycles", _cycles(self.horizon_cycles, "horizon_cycles"))


@dataclass(frozen=True)
class JobAssignment:
    """A non-preemptive job start on a one-based identical team ID."""

    engine: EngineIdentity
    team_id: int
    start_cycle: int

    def __post_init__(self):
        if not isinstance(self.engine, EngineIdentity):
            raise TypeError("engine must be a dataset-qualified EngineIdentity.")
        object.__setattr__(self, "team_id", _integer(self.team_id, "team_id"))
        object.__setattr__(
            self, "start_cycle", _integer(self.start_cycle, "start_cycle", zero=True)
        )


@dataclass(frozen=True)
class ScheduledJob:
    job: MaintenanceJob
    team_id: int
    timing: JobTiming
    horizon_overflow_cycles: float


@dataclass(frozen=True)
class ScheduleEvaluation:
    """Weighted tardiness objective with separately labelled overflow measures."""

    weighted_tardiness_cycles: float
    total_tardiness_cycles: float
    late_job_count: int
    makespan_cycles: int
    horizon_overflow_cycles: float
    overflow_job_count: int
    scheduled_jobs: tuple[ScheduledJob, ...]


def evaluate_schedule(
    scenario: SchedulingScenario,
    assignments: Iterable[JobAssignment],
) -> ScheduleEvaluation:
    """Require all selected jobs exactly once and no overlaps per team.

    Objective = sum(priority_weight * max(0, completion - due_completion)).
    Due/completion use the shared reviewed timing function. No horizon penalty
    is added to the objective and no overflow job is dropped. An infeasible
    schedule raises rather than receiving a misleading comparable score.
    Return order follows supplied assignments; overlap checks sort per team.
    """
    if not isinstance(scenario, SchedulingScenario):
        raise TypeError("scenario must be a SchedulingScenario.")
    assignments = tuple(assignments)
    if any(not isinstance(assignment, JobAssignment) for assignment in assignments):
        raise TypeError("assignments must contain JobAssignment values.")
    by_engine = {job.engine: job for job in scenario.jobs}
    engines = [assignment.engine for assignment in assignments]
    if len(set(engines)) != len(engines):
        raise ScheduleFeasibilityError(
            "Every selected job must be scheduled once; duplicate assignment."
        )
    if any(engine not in by_engine for engine in engines):
        raise ScheduleFeasibilityError("Schedule contains an unknown dataset-qualified engine.")
    if set(engines) != set(by_engine):
        raise ScheduleFeasibilityError(
            "Every selected job must be scheduled once; missing assignment."
        )
    entries = []
    per_team = {}
    for assignment in assignments:
        if assignment.team_id > scenario.team_count:
            raise ScheduleFeasibilityError("Assigned team_id exceeds scenario team capacity.")
        job = by_engine[assignment.engine]
        timing = calculate_job_timing(
            job, buffer_cycles=scenario.buffer_cycles, start_cycle=assignment.start_cycle
        )
        entry = ScheduledJob(
            job,
            assignment.team_id,
            timing,
            max(0.0, timing.completion_cycle - scenario.horizon_cycles),
        )
        entries.append(entry)
        per_team.setdefault(assignment.team_id, []).append(entry)
    for team_id, team_entries in per_team.items():
        ordered = sorted(team_entries, key=lambda entry: entry.timing.start_cycle)
        for previous, current in pairwise(ordered):
            if current.timing.start_cycle < previous.timing.completion_cycle:
                raise ScheduleFeasibilityError(f"Jobs overlap on team {team_id}.")
    try:
        weighted = math.fsum(
            entry.job.priority_weight * entry.timing.tardiness_cycles for entry in entries
        )
        total = math.fsum(entry.timing.tardiness_cycles for entry in entries)
    except (OverflowError, ValueError) as error:
        raise ValueError("Schedule objective exceeds finite numeric range.") from error
    if not math.isfinite(weighted) or not math.isfinite(total):
        raise ValueError("Schedule objective must be finite.")
    makespan = max(entry.timing.completion_cycle for entry in entries)
    return ScheduleEvaluation(
        weighted,
        total,
        sum(entry.timing.tardiness_cycles > 0 for entry in entries),
        makespan,
        max(0.0, makespan - scenario.horizon_cycles),
        sum(entry.horizon_overflow_cycles > 0 for entry in entries),
        tuple(entries),
    )
