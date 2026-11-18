"""Provenance binding and serialization for saved paired schedule comparisons."""

import hashlib
import json

import pytest

from fleetpulse.optimization.jobs import appendix_b_jobs
from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services import (
    EngineIdentity,
    FleetPredictionResult,
    build_paired_comparison_artifact,
    compare_schedules,
    model_provenance_from_manifest,
)


def _inputs():
    predictions = tuple(
        FleetPredictionResult(
            EngineIdentity(job.engine.dataset_partition, job.engine.unit_id),
            observed_cycle=job.engine.unit_id + 10,
            raw_rul_cycles=job.predicted_rul_cycles,
            displayed_rul_cycles=job.predicted_rul_cycles,
            model_name="decision-tree-rul",
            model_version="decision-tree-rul-v1",
            input_window_rows=2,
            warnings=(),
            error_codes=(),
            error=None,
        )
        for job in appendix_b_jobs()
    )
    scenario = SchedulingScenario(appendix_b_jobs(), 1, 2, 6)
    comparison = compare_schedules(scenario)
    source = {
        "source_name": "synthetic fixture for artifact contract test",
        "dataset_partition": "synthetic/appendix-b-v1",
        "source_records_sha256": "a" * 64,
    }
    model = {
        "model_name": "decision-tree-rul",
        "model_version": "decision-tree-rul-v1",
        "manifest_sha256": "b" * 64,
        "artifact_sha256": "c" * 64,
        "feature_config": {"channels": ["sensor_1"], "window_cycles": 30},
        "feature_schema": ["sensor_1_latest", "sensor_1_mean"],
    }
    return predictions, scenario, comparison, source, model


def test_paired_artifact_saves_prediction_model_scenario_and_both_methods():
    predictions, scenario, comparison, source, model = _inputs()
    artifact = build_paired_comparison_artifact(
        predictions,
        scenario,
        comparison,
        source_provenance=source,
        model_provenance=model,
    )
    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    decoded = json.loads(serialized)

    assert decoded["schema_version"] == "p20-paired-schedule-comparison-v1"
    assert decoded["provenance"]["source"] == source
    assert decoded["provenance"]["model"] == model
    assert decoded["scenario"]["scenario_sha256"] == comparison.scenario_sha256
    assert decoded["comparisons"]["edd"]["feasible"] is True
    assert decoded["comparisons"]["genetic_algorithm"]["feasible"] is True
    assert decoded["comparisons"]["edd"]["weighted_tardiness_cycles"] == 6
    assert decoded["comparisons"]["genetic_algorithm"]["weighted_tardiness_cycles"] == 2
    assert len(decoded["provenance"]["prediction_snapshot"]) == 3
    assert "true_rul" not in serialized.lower()
    assert "label" not in decoded["scenario"]


@pytest.mark.parametrize("provenance", [{}, {"source_records_sha256": "a" * 64}])
def test_paired_artifact_rejects_missing_source_or_model_provenance(provenance):
    predictions, scenario, comparison, source, model = _inputs()
    if not provenance:
        source = provenance
    else:
        model = provenance
    with pytest.raises(ValueError, match="provenance|hashes"):
        build_paired_comparison_artifact(
            predictions,
            scenario,
            comparison,
            source_provenance=source,
            model_provenance=model,
        )


def test_paired_artifact_rejects_reports_from_a_different_scenario():
    predictions, scenario, comparison, source, model = _inputs()
    changed_scenario = SchedulingScenario(scenario.jobs, 1, 2, 7)

    with pytest.raises(ValueError, match="supplied scenario"):
        build_paired_comparison_artifact(
            predictions,
            changed_scenario,
            comparison,
            source_provenance=source,
            model_provenance=model,
        )


def test_model_manifest_provenance_captures_manifest_model_and_features(tmp_path):
    payload = {
        "model": {
            "name": "decision-tree-rul",
            "version": "decision-tree-rul-v1",
            "filename": "decision_tree_model.joblib",
            "sha256": "c" * 64,
        },
        "training": {
            "random_seed": 42,
            "parameters": {"max_depth": 8},
            "scikit_learn_version": "1.9.1",
            "feature_config": {"channels": ["sensor_1"], "window_cycles": 30},
            "feature_names": ["sensor_1_latest", "sensor_1_mean"],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = {"sha256": hashlib.sha256(canonical.encode()).hexdigest(), "payload": payload}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    provenance = model_provenance_from_manifest(path)

    assert provenance["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["artifact_sha256"] == "c" * 64
    assert provenance["feature_schema"] == payload["training"]["feature_names"]
    assert provenance["training_seed"] == 42
