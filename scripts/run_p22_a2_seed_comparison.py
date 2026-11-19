"""Compare EDD with the fixed genetic scheduler across declared seeds.

Reuses the P22-A1 fleet inputs and assumptions, varying only the GA seed. The
P13 loader consumes raw Decision Tree predictions but no true-RUL labels.
"""

from __future__ import annotations

import json
import os
import platform
from time import perf_counter

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.jobs import MaintenanceJob, appendix_b_jobs
from fleetpulse.optimization.scoring import (
    JobAssignment,
    SchedulingScenario,
    evaluate_schedule,
)

if __package__:
    from .run_p22_a1_fleet_scenarios import (
        BUFFER_LEVELS_CYCLES,
        FLEET_SIZES,
        HORIZON_CYCLES,
        PARTITION,
        TEAM_COUNTS,
        load_raw_predictions,
        synthetic_assumptions,
    )
else:  # Direct `python scripts/run_p22_a2_seed_comparison.py` invocation.
    from run_p22_a1_fleet_scenarios import (
        BUFFER_LEVELS_CYCLES,
        FLEET_SIZES,
        HORIZON_CYCLES,
        PARTITION,
        TEAM_COUNTS,
        load_raw_predictions,
        synthetic_assumptions,
    )

SEEDS = (0, 7, 42, 2026, 2027)


def build_scenario(predictions: dict[int, float], fleet_size: int, team_count: int, buffer: int):
    selected_ids = tuple(range(1, fleet_size + 1))
    missing = sorted(set(selected_ids).difference(predictions))
    if missing:
        raise ValueError(f"Missing raw predictions for required engine IDs: {missing}.")
    jobs = []
    for unit_id in selected_ids:
        duration, weight = synthetic_assumptions(unit_id)
        jobs.append(
            MaintenanceJob(
                EngineIdentity(PARTITION, unit_id),
                predictions[unit_id],
                duration,
                weight,
            )
        )
    return SchedulingScenario(tuple(jobs), team_count, buffer, HORIZON_CYCLES)


def check_feasibility(scenario: SchedulingScenario, schedule) -> bool:
    """Re-run the shared hard-feasibility validator on each returned schedule."""
    assignments = tuple(
        JobAssignment(entry.job.engine, entry.team_id, entry.timing.start_cycle)
        for entry in schedule.scheduled_jobs
    )
    return evaluate_schedule(scenario, assignments) == schedule


def summarize(schedule) -> dict:
    return {
        "weighted_tardiness_cycles": schedule.weighted_tardiness_cycles,
        "total_tardiness_cycles": schedule.total_tardiness_cycles,
        "late_job_count": schedule.late_job_count,
        "makespan_cycles": schedule.makespan_cycles,
        "horizon_overflow_cycles": schedule.horizon_overflow_cycles,
        "overflow_job_count": schedule.overflow_job_count,
        "feasible": True,
        "order": [entry.job.engine.unit_id for entry in schedule.scheduled_jobs],
    }


def appendix_b_reference() -> dict:
    """Check the hand-calculated three-job case retained from P18."""
    scenario = SchedulingScenario(appendix_b_jobs(), team_count=1, buffer_cycles=2, horizon_cycles=6)
    edd = schedule_edd(scenario)
    ga = schedule_ga(scenario, seed=42)
    if (edd.weighted_tardiness_cycles, ga.schedule.weighted_tardiness_cycles) != (6, 2):
        raise AssertionError("Appendix B hand-calculated reference scores changed.")
    if not check_feasibility(scenario, edd) or not check_feasibility(scenario, ga.schedule):
        raise AssertionError("Appendix B scheduler returned an infeasible result.")
    return {
        "assumptions": "synthetic Appendix B jobs; 1 team; buffer 2; horizon 6",
        "edd": summarize(edd),
        "ga_seed_42": summarize(ga.schedule),
    }


def run_comparison(predictions: dict[int, float]) -> dict:
    rows = []
    for fleet_size in FLEET_SIZES:
        for buffer in BUFFER_LEVELS_CYCLES:
            for team_count in TEAM_COUNTS:
                scenario = build_scenario(predictions, fleet_size, team_count, buffer)
                edd_started = perf_counter()
                edd = schedule_edd(scenario)
                edd_runtime = perf_counter() - edd_started
                if not check_feasibility(scenario, edd):
                    raise AssertionError("EDD schedule failed hard-feasibility revalidation.")
                seed_rows = []
                for seed in SEEDS:
                    first = schedule_ga(scenario, seed=seed)
                    second = schedule_ga(scenario, seed=seed)
                    if (
                        first.order != second.order
                        or first.schedule != second.schedule
                        or first.best_penalties != second.best_penalties
                    ):
                        raise AssertionError(f"GA was not repeatable for seed {seed}.")
                    if not check_feasibility(scenario, first.schedule):
                        raise AssertionError(f"GA schedule was infeasible for seed {seed}.")
                    score = first.schedule.weighted_tardiness_cycles
                    if score > edd.weighted_tardiness_cycles:
                        raise AssertionError(f"GA regressed from its EDD seed for seed {seed}.")
                    outcome = "equal" if score == edd.weighted_tardiness_cycles else "better"
                    seed_rows.append(
                        {
                            "seed": seed,
                            "ga": summarize(first.schedule),
                            "outcome_vs_edd": outcome,
                            "runtime_seconds": first.runtime_seconds,
                            "unique_schedule_evaluations": first.evaluations,
                            "repeatability_verified": True,
                        }
                    )
                rows.append(
                    {
                        "fleet_size": fleet_size,
                        "team_count": team_count,
                        "buffer_cycles": buffer,
                        "engine_ids": list(range(1, fleet_size + 1)),
                        "edd": summarize(edd),
                        "edd_runtime_seconds": edd_runtime,
                        "ga_by_seed": seed_rows,
                    }
                )

    all_seed_rows = [seed for row in rows for seed in row["ga_by_seed"]]
    return {
        "schema": "p22-a2-seed-comparison-v1",
        "dataset_partition": PARTITION,
        "ga_seeds": list(SEEDS),
        "fleet_sizes": list(FLEET_SIZES),
        "team_counts": list(TEAM_COUNTS),
        "buffer_levels_cycles": list(BUFFER_LEVELS_CYCLES),
        "soft_horizon_cycles": HORIZON_CYCLES,
        "duration_assumption_cycles": "1 + (engine_id mod 3)",
        "priority_weight_assumption": "1 + ((engine_id * 7) mod 5)",
        "case_count": len(rows),
        "ga_run_count": len(all_seed_rows),
        "outcome_counts": {
            outcome: sum(seed["outcome_vs_edd"] == outcome for seed in all_seed_rows)
            for outcome in ("equal", "better")
        },
        "all_schedules_feasible": all(
            row["edd"]["feasible"]
            and all(seed["ga"]["feasible"] for seed in row["ga_by_seed"])
            for row in rows
        ),
        "all_seed_repeats_match": all(
            seed["repeatability_verified"] for seed in all_seed_rows
        ),
        "appendix_b_hand_check": appendix_b_reference(),
        "scenarios": rows,
    }


def main() -> None:
    report_path = PROJECT_ROOT / "artifacts/final/p13/final_test_results.json"
    if not report_path.is_file():
        raise SystemExit(f"Required frozen P13 report is unavailable: {report_path}")
    predictions, provenance = load_raw_predictions(report_path)
    result = run_comparison(predictions)
    result["input_provenance"] = provenance
    result["benchmark_environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
