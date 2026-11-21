"""Run the fixed P22-A1 fleet-size/team-count scheduler experiment.

Inputs are raw Decision Tree estimates from the frozen P13 FD001/test report.
This script deliberately reads no true-RUL labels. Service durations and
priority weights are deterministic simulation assumptions, not dataset facts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from math import isfinite
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.split import EngineIdentity
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEED,
    schedule_ga,
)
from fleetpulse.optimization.jobs import MaintenanceJob
from fleetpulse.optimization.scoring import SchedulingScenario

MODEL_NAME = "decision-tree-rul"
PARTITION = "FD001/test"
FLEET_SIZES = (5, 10, 20)
TEAM_COUNTS = (1, 2, 3)
BUFFER_LEVELS_CYCLES = (2, 120)
HORIZON_CYCLES = 30


def load_raw_predictions(path: Path) -> tuple[dict[int, float], dict[str, str]]:
    """Read only unit IDs and raw prediction fields from the frozen P13 report."""
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    models = report["payload"]["models"]
    matches = [model for model in models if model.get("name") == MODEL_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {MODEL_NAME!r} model in {path}.")

    model = matches[0]
    rows = model.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"The {MODEL_NAME!r} model has no prediction rows in {path}.")

    predictions: dict[int, float] = {}
    for row in rows:
        # Do not access true_rul_cycles or any error/metric fields here.
        unit_id = int(row["unit_id"])
        prediction = float(row["raw_prediction_cycles"])
        if unit_id < 1 or not isfinite(prediction):
            raise ValueError("P13 raw predictions require positive engine IDs and finite values.")
        if unit_id in predictions:
            raise ValueError(f"Duplicate {PARTITION} engine ID {unit_id} in {path}.")
        predictions[unit_id] = prediction

    artifact = model["artifact"]
    provenance = {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "model_name": MODEL_NAME,
        "model_version": artifact["version"],
        "manifest_sha256": artifact["manifest_file_sha256"],
        "weights_sha256": artifact["weights_sha256"],
    }
    return predictions, provenance


def synthetic_assumptions(unit_id: int) -> tuple[int, int]:
    """Return stable varied (duration, priority) assumptions for an engine ID."""
    return 1 + (unit_id % 3), 1 + ((unit_id * 7) % 5)


def run_experiment(predictions: dict[int, float]) -> dict:
    """Evaluate EDD and seeded GA for all 18 fixed scenarios."""
    rows = []
    for fleet_size in FLEET_SIZES:
        selected_ids = tuple(range(1, fleet_size + 1))
        missing = sorted(set(selected_ids).difference(predictions))
        if missing:
            raise ValueError(f"Missing raw predictions for required engine IDs: {missing}.")
        for buffer_cycles in BUFFER_LEVELS_CYCLES:
            for team_count in TEAM_COUNTS:
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
                scenario = SchedulingScenario(
                    tuple(jobs), team_count, buffer_cycles, HORIZON_CYCLES
                )
                edd = schedule_edd(scenario)
                ga = schedule_ga(scenario)
                if ga.schedule.weighted_tardiness_cycles > edd.weighted_tardiness_cycles:
                    raise AssertionError(
                        "Seeded GA returned a worse score than its seeded EDD order."
                    )
                if ga.schedule.weighted_tardiness_cycles == edd.weighted_tardiness_cycles:
                    outcome = "equal"
                else:
                    outcome = "better"
                rows.append(
                    {
                        "fleet_size": fleet_size,
                        "team_count": team_count,
                        "buffer_cycles": buffer_cycles,
                        "engine_ids": list(selected_ids),
                        "edd": summarize(edd),
                        "ga": summarize(ga.schedule),
                        "ga_outcome_vs_edd": outcome,
                        "ga_runtime_seconds": ga.runtime_seconds,
                        "ga_unique_schedule_evaluations": ga.evaluations,
                        "ga_population_size": ga.population_size,
                        "ga_generations": ga.generations,
                        "ga_seed": ga.seed,
                    }
                )

    outcomes = {
        name: sum(row["ga_outcome_vs_edd"] == name for row in rows)
        for name in ("equal", "better")
    }
    return {
        "schema": "p22-a1-fixed-fleet-scenarios-v1",
        "dataset_partition": PARTITION,
        "selected_engine_ids": "ascending IDs 1 through each fleet size",
        "fleet_sizes": list(FLEET_SIZES),
        "team_counts": list(TEAM_COUNTS),
        "buffer_levels_cycles": list(BUFFER_LEVELS_CYCLES),
        "soft_horizon_cycles": HORIZON_CYCLES,
        "duration_assumption_cycles": "1 + (engine_id mod 3)",
        "priority_weight_assumption": "1 + ((engine_id * 7) mod 5)",
        "ga_configuration": {
            "population_size": DEFAULT_POPULATION_SIZE,
            "generations": DEFAULT_GENERATIONS,
            "seed": DEFAULT_SEED,
        },
        "ga_outcomes": outcomes,
        "scenarios": rows,
    }


def summarize(result) -> dict:
    return {
        "weighted_tardiness_cycles": result.weighted_tardiness_cycles,
        "total_tardiness_cycles": result.total_tardiness_cycles,
        "late_job_count": result.late_job_count,
        "makespan_cycles": result.makespan_cycles,
        "horizon_overflow_cycles": result.horizon_overflow_cycles,
        "overflow_job_count": result.overflow_job_count,
        "order": [job.job.engine.unit_id for job in result.scheduled_jobs],
    }


def main() -> None:
    report_path = PROJECT_ROOT / "artifacts/final/p13/final_test_results.json"
    if not report_path.is_file():
        raise SystemExit(f"Required frozen P13 report is unavailable: {report_path}")
    predictions, provenance = load_raw_predictions(report_path)
    result = run_experiment(predictions)
    result["input_provenance"] = provenance
    result["benchmark_environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
