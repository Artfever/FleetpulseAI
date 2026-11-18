"""Verify the paired scheduling service on synthetic and saved-model inputs."""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services import PlannerJobAssumption, build_planning_scenario, compare_schedules
from fleetpulse.services.fleet import predict_fleet


def _method_output(result):
    return {
        "feasible": result.feasible,
        "penalty_weighted_tardiness_cycles": result.weighted_tardiness_cycles,
        "late_jobs": [list(identity) for identity in result.late_jobs],
        "makespan_cycles": result.makespan_cycles,
        "horizon_overflow_cycles": result.horizon_overflow_cycles,
        "runtime_seconds": result.runtime_seconds,
        "report": result.report,
        "search": (
            {
                "seed": result.search.seed,
                "population_size": result.search.population_size,
                "generations": result.search.generations,
                "evaluations": result.search.evaluations,
            }
            if result.search is not None
            else None
        ),
    }


def main() -> None:
    synthetic = SchedulingScenario(appendix_b_jobs(), 1, buffer_cycles=2, horizon_cycles=6)
    synthetic_pair = compare_schedules(synthetic)
    if synthetic_pair.edd.weighted_tardiness_cycles != 6:
        raise SystemExit("Appendix B EDD score did not match its hand-calculated 6 cycles.")
    if synthetic_pair.ga.weighted_tardiness_cycles != 2:
        raise SystemExit("Appendix B GA score did not match its hand-calculated 2 cycles.")

    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    demo_path = PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt"
    for path in (manifest_path, demo_path):
        if not path.is_file():
            raise SystemExit(f"Required local evidence input is unavailable: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    histories = load_engine_history(demo_path)
    predictions = predict_fleet(
        histories, dataset_partition="FD001/train", manifest_path=manifest_path
    )
    if not predictions or any(result.error for result in predictions):
        raise SystemExit("At least one packaged demo engine did not produce a saved-model estimate.")
    assumptions = {
        result.identity: PlannerJobAssumption(duration_cycles=1, priority_weight=1)
        for result in predictions
    }
    actual_scenario = build_planning_scenario(
        predictions,
        assumptions,
        team_count=1,
        buffer_cycles=2,
        horizon_cycles=6,
    )
    actual_pair = compare_schedules(actual_scenario)
    if actual_pair.scenario_sha256 != actual_pair.edd.report["scenario_sha256"]:
        raise SystemExit("Paired service returned an inconsistent scenario hash.")

    output = {
        "synthetic_provenance": "Appendix B toy scenario; durations/weights/capacity/buffer/horizon are assumptions",
        "synthetic_expected_penalties": {"edd": 6, "ga": 2},
        "synthetic": {
            "scenario_sha256": synthetic_pair.scenario_sha256,
            "edd": _method_output(synthetic_pair.edd),
            "ga": _method_output(synthetic_pair.ga),
        },
        "saved_model_provenance": {
            "dataset": "FD001/train",
            "source": "packaged training-prefix demo subset",
            "history_count": len(histories),
            "model": manifest["payload"]["model"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "actual_scenario_assumptions": {
            "duration_cycles_per_job": 1,
            "priority_weight_per_job": 1,
            "team_count": 1,
            "buffer_cycles": 2,
            "horizon_cycles": 6,
            "assumptions_are_illustrative": True,
        },
        "actual_predictions": [
            {
                "dataset_partition": result.identity.dataset_partition,
                "engine_id": result.identity.unit_id,
                "observed_cycle": result.observed_cycle,
                "raw_rul_cycles": result.raw_rul_cycles,
            }
            for result in predictions
        ],
        "actual_pair": {
            "scenario_sha256": actual_pair.scenario_sha256,
            "edd": _method_output(actual_pair.edd),
            "ga": _method_output(actual_pair.ga),
        },
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
