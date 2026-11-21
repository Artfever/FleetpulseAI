"""Run a saved-model FD001/train subset through both P20 schedulers."""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.optimization.edd import schedule_edd
from fleetpulse.optimization.ga import schedule_ga
from fleetpulse.optimization.reporting import build_schedule_report
from fleetpulse.services import PlannerJobAssumption, build_planning_scenario, predict_fleet


def main() -> None:
    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    demo_path = PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt"
    for path in (manifest_path, demo_path):
        if not path.is_file():
            raise SystemExit(f"Required local evidence input is unavailable: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    histories = load_engine_history(demo_path)
    predictions = predict_fleet(
        histories,
        dataset_partition="FD001/train",
        manifest_path=manifest_path,
    )
    if not predictions or any(result.error for result in predictions):
        raise SystemExit("At least one packaged demo engine did not produce a saved-model estimate.")

    assumptions = {
        result.identity: PlannerJobAssumption(duration_cycles=1, priority_weight=1)
        for result in predictions
    }
    scenario = build_planning_scenario(
        predictions,
        assumptions,
        team_count=1,
        buffer_cycles=2,
        horizon_cycles=6,
    )
    edd = build_schedule_report(scenario, schedule_edd(scenario))
    ga_result = schedule_ga(scenario, population_size=16, generations=20, seed=42)
    ga = build_schedule_report(scenario, ga_result.schedule)
    if edd["scenario_sha256"] != ga["scenario_sha256"]:
        raise SystemExit("EDD and GA did not use the same scenario snapshot.")

    output = {
        "provenance": {
            "dataset": "FD001/train",
            "histories": len(histories),
            "source": "packaged training-prefix demo subset",
            "model": manifest["payload"]["model"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "scenario": {
            "assumptions": "illustrative only; not NASA measurements or operational recommendations",
            "team_count": scenario.team_count,
            "buffer_cycles": scenario.buffer_cycles,
            "horizon_cycles": scenario.horizon_cycles,
            "duration_cycles_per_job": 1,
            "priority_weight_per_job": 1,
            "scenario_sha256": edd["scenario_sha256"],
        },
        "predictions": [
            {
                "engine_id": result.identity.unit_id,
                "observed_cycle": result.observed_cycle,
                "raw_rul_cycles": result.raw_rul_cycles,
                "planner_rul_cycles": job.predicted_rul_cycles,
            }
            for result, job in zip(predictions, scenario.jobs, strict=True)
        ],
        "edd_summary": edd["summary"],
        "edd_jobs": edd["jobs"],
        "ga_summary": ga["summary"],
        "ga_jobs": ga["jobs"],
        "ga_search": {
            "seed": ga_result.seed,
            "population_size": ga_result.population_size,
            "generations": ga_result.generations,
            "evaluations": ga_result.evaluations,
        },
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
