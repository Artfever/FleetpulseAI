"""Print the observed saved-model values at the Fleet Overview/planner boundary.

Run locally after restoring the ignored development Decision Tree bundle.
This is a wiring/evidence check on the packaged three-prefix demo, not a model
quality evaluation or maintenance recommendation.
"""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.optimization.jobs import calculate_job_timing
from fleetpulse.services import PlannerJobAssumption, build_planning_scenario, predict_fleet


def main() -> None:
    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    demo_path = PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt"
    for path in (manifest_path, demo_path):
        if not path.is_file():
            raise SystemExit(f"Required local evidence input is unavailable: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest["payload"]
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
    results_by_identity = {result.identity.unit_id: result for result in predictions}
    rows = []
    for job in scenario.jobs:
        result = results_by_identity[job.engine.unit_id]
        timing = calculate_job_timing(job, buffer_cycles=scenario.buffer_cycles)
        expected_rul = max(0.0, result.raw_rul_cycles)
        expected_due = max(0.0, expected_rul - scenario.buffer_cycles)
        if job.predicted_rul_cycles != expected_rul or timing.due_completion_cycle != expected_due:
            raise SystemExit(f"Engine {job.engine.unit_id} changed at the planner boundary.")
        rows.append(
            {
                "dataset": job.engine.dataset_partition,
                "engine_id": job.engine.unit_id,
                "observed_cycle": result.observed_cycle,
                "raw_rul_cycles": result.raw_rul_cycles,
                "planner_rul_cycles": job.predicted_rul_cycles,
                "due_completion_cycle": timing.due_completion_cycle,
                "duration_cycles_assumed": job.duration_cycles,
                "priority_weight_assumed": job.priority_weight,
            }
        )

    output = {
        "model": payload["model"],
        "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training": {
            "random_seed": payload["training"]["random_seed"],
            "parameters": payload["training"]["parameters"],
            "fitting_engine_count": payload["training"]["fitting_engine_count"],
            "fitting_row_count": payload["training"]["fitting_row_count"],
            "feature_config": payload["training"]["feature_config"],
        },
        "scenario_assumptions": {
            "team_count": scenario.team_count,
            "buffer_cycles": scenario.buffer_cycles,
            "horizon_cycles": scenario.horizon_cycles,
            "duration_and_weight_are_synthetic": True,
        },
        "rows": rows,
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
