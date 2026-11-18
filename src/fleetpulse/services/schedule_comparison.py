"""Run and report EDD and GA on one immutable scheduling scenario."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEED,
    GeneticSearchResult,
    schedule_ga,
)
from fleetpulse.optimization.reporting import build_schedule_report
from fleetpulse.optimization.scoring import ScheduleEvaluation, SchedulingScenario


@dataclass(frozen=True)
class ScheduleMethodResult:
    """A checked schedule report with measured algorithm runtime."""

    report: dict
    runtime_seconds: float
    search: GeneticSearchResult | None = None

    @property
    def feasible(self) -> bool:
        """True only because report creation revalidates every assignment."""
        return bool(self.report["feasibility_checked"])

    @property
    def weighted_tardiness_cycles(self) -> float:
        return self.report["summary"]["weighted_tardiness_cycles"]

    @property
    def late_jobs(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (row["dataset_partition"], row["unit_id"])
            for row in self.report["jobs"]
            if row["tardiness_cycles"] > 0
        )

    @property
    def makespan_cycles(self) -> int:
        return self.report["summary"]["makespan_cycles"]

    @property
    def horizon_overflow_cycles(self) -> float:
        return self.report["summary"]["horizon_overflow_cycles"]


@dataclass(frozen=True)
class ScheduleComparisonResult:
    """Paired outputs whose checked reports share one exact scenario hash."""

    scenario_sha256: str
    edd: ScheduleMethodResult
    ga: ScheduleMethodResult


def _method_result(
    scenario: SchedulingScenario,
    schedule: ScheduleEvaluation,
    runtime_seconds: float,
    *,
    search: GeneticSearchResult | None = None,
) -> ScheduleMethodResult:
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("Scheduler runtime must be finite and nonnegative.")
    return ScheduleMethodResult(
        report=build_schedule_report(scenario, schedule),
        runtime_seconds=float(runtime_seconds),
        search=search,
    )


def compare_schedules(
    scenario: SchedulingScenario,
    *,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    seed: int = DEFAULT_SEED,
) -> ScheduleComparisonResult:
    """Run both methods on the same scenario and return checked summaries.

    Runtime measures each scheduling algorithm call, excluding report
    validation/serialization. A successful return means both reports passed
    the shared feasibility validator. If either schedule is infeasible or a
    setting is invalid, the call raises and returns no partial comparison.
    """
    if not isinstance(scenario, SchedulingScenario):
        raise TypeError("scenario must be a SchedulingScenario.")

    started = perf_counter()
    edd_schedule = schedule_edd(scenario)
    edd_runtime = perf_counter() - started
    edd = _method_result(scenario, edd_schedule, edd_runtime)

    started = perf_counter()
    ga_search = schedule_ga(
        scenario,
        population_size=population_size,
        generations=generations,
        seed=seed,
    )
    ga_runtime = perf_counter() - started
    ga = _method_result(scenario, ga_search.schedule, ga_runtime, search=ga_search)

    if edd.report["scenario_sha256"] != ga.report["scenario_sha256"]:
        raise RuntimeError("Both scheduler reports must refer to the same scenario snapshot.")
    return ScheduleComparisonResult(edd.report["scenario_sha256"], edd, ga)
