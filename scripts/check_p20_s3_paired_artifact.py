"""Build and validate a paired artifact from the packaged saved-model demo."""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import (
    PlannerJobAssumption,
    build_paired_comparison_artifact,
    build_planning_scenario,
    compare_schedules,
    model_provenance_from_manifest,
    predict_fleet,
)


def main() -> None:
    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    source_path = PROJECT_ROOT / "data/demo/fd001_training_prefixes.txt"
    for path in (manifest_path, source_path):
        if not path.is_file():
            raise SystemExit(f"Required local evidence input is unavailable: {path}")

    histories = load_engine_history(source_path)
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
    comparison = compare_schedules(scenario, population_size=16, generations=20, seed=42)
    canonical_records = json.dumps(
        histories, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source_provenance = {
        "source_name": "Packaged FD001/train prefix subset",
        "source_path": "data/demo/fd001_training_prefixes.txt",
        "dataset_partition": "FD001/train",
        "source_records_sha256": hashlib.sha256(canonical_records).hexdigest(),
        "source_record_count": len(histories),
        "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    artifact = build_paired_comparison_artifact(
        predictions,
        scenario,
        comparison,
        source_provenance=source_provenance,
        model_provenance=model_provenance_from_manifest(manifest_path),
    )
    serialized = json.dumps(artifact, indent=2, allow_nan=False)
    restored = json.loads(serialized)
    assert restored["comparisons"]["edd"]["schedule_report"]["scenario_sha256"] == restored[
        "comparisons"
    ]["genetic_algorithm"]["schedule_report"]["scenario_sha256"]
    assert len(restored["provenance"]["prediction_snapshot"]) == len(predictions)
    assert "true_rul" not in serialized.lower()
    print(
        json.dumps(
            {
                "schema_version": restored["schema_version"],
                "model_version": restored["provenance"]["model"]["model_version"],
                "model_sha256": restored["provenance"]["model"]["artifact_sha256"],
                "manifest_sha256": restored["provenance"]["model"]["manifest_sha256"],
                "source_file_sha256": source_provenance["source_file_sha256"],
                "source_records_sha256": source_provenance["source_records_sha256"],
                "scenario_sha256": restored["scenario"]["scenario_sha256"],
                "predictions": restored["provenance"]["prediction_snapshot"],
                "edd_weighted_tardiness_cycles": restored["comparisons"]["edd"][
                    "weighted_tardiness_cycles"
                ],
                "ga_weighted_tardiness_cycles": restored["comparisons"][
                    "genetic_algorithm"
                ]["weighted_tardiness_cycles"],
                "assumptions": "illustrative: one team, duration/weight 1, buffer 2, horizon 6",
                "artifact_round_trip": "PASS",
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
