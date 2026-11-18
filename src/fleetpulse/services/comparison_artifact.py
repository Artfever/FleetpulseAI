"""Provenance-bound JSON artifact for a paired EDD/GA scenario comparison."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from numbers import Real
from pathlib import Path

from fleetpulse.optimization.scoring import SchedulingScenario
from fleetpulse.services.fleet import EngineIdentity, FleetPredictionResult
from fleetpulse.services.schedule_comparison import ScheduleComparisonResult

PAIRED_COMPARISON_SCHEMA = "p20-paired-schedule-comparison-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _scenario_sha256(scenario: SchedulingScenario) -> str:
    return hashlib.sha256(_canonical(asdict(scenario))).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def model_provenance_from_manifest(manifest_path: str | Path) -> dict:
    """Read model/schema identity and hashes from the already verified manifest."""
    path = Path(manifest_path)
    try:
        manifest_bytes = path.read_bytes()
        document = json.loads(manifest_bytes)
        payload = document["payload"]
        if document["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest():
            raise ValueError("manifest payload hash mismatch")
        model = payload["model"]
        training = payload["training"]
        feature_names = training["feature_names"]
        feature_config = training["feature_config"]
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("manifest feature schema is unavailable")
        if not isinstance(feature_config, dict):
            raise TypeError("manifest feature config is unavailable")
        return {
            "model_name": model["name"],
            "model_version": model["version"],
            "artifact_filename": model["filename"],
            "artifact_sha256": model["sha256"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_payload_sha256": document["sha256"],
            "training_seed": training.get("random_seed"),
            "training_parameters": training.get("parameters"),
            "scikit_learn_version": training.get("scikit_learn_version"),
            "feature_config": feature_config,
            "feature_schema": feature_names,
        }
    except OSError as error:
        raise ValueError(f"Cannot read model manifest for comparison provenance: {path}") from error
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Model manifest has incomplete comparison provenance: {path}") from error


def build_paired_comparison_artifact(
    predictions: Iterable[FleetPredictionResult],
    scenario: SchedulingScenario,
    comparison: ScheduleComparisonResult,
    *,
    source_provenance: Mapping[str, object],
    model_provenance: Mapping[str, object],
) -> dict:
    """Bind model outputs, exact scenario inputs, and both checked schedules.

    The caller supplies provenance captured at prediction time. This function
    rejects missing/failed predictions, mismatched engine sets, differing
    model versions, and schedule reports built from another scenario. True RUL
    and validation labels are never accepted or written.
    """
    if not isinstance(scenario, SchedulingScenario):
        raise TypeError("scenario must be a SchedulingScenario.")
    if not isinstance(comparison, ScheduleComparisonResult):
        raise TypeError("comparison must be a ScheduleComparisonResult.")
    if comparison.scenario_sha256 != comparison.edd.report["scenario_sha256"]:
        raise ValueError("EDD report does not match the paired scenario identity.")
    if comparison.scenario_sha256 != comparison.ga.report["scenario_sha256"]:
        raise ValueError("GA report does not match the paired scenario identity.")
    if comparison.scenario_sha256 != _scenario_sha256(scenario):
        raise ValueError("Paired schedule reports were not built from the supplied scenario.")
    if not isinstance(source_provenance, Mapping) or not source_provenance:
        raise ValueError("Source provenance is required to save the paired comparison.")
    if not isinstance(model_provenance, Mapping) or not model_provenance:
        raise ValueError("Model provenance is required to save the paired comparison.")
    if not _is_sha256(source_provenance.get("source_records_sha256")):
        raise ValueError("A valid SHA-256 hash of the prediction source records is required.")
    if not all(
        _is_sha256(model_provenance.get(key))
        for key in ("manifest_sha256", "artifact_sha256")
    ):
        raise ValueError("Valid model file and manifest SHA-256 hashes are required.")

    try:
        rows = tuple(predictions)
    except TypeError as error:
        raise TypeError("predictions must be an iterable of FleetPredictionResult values.") from error
    if not rows or any(not isinstance(row, FleetPredictionResult) for row in rows):
        raise ValueError("At least one typed fleet prediction is required.")
    by_identity = {row.identity: row for row in rows}
    if len(by_identity) != len(rows):
        raise ValueError("Fleet prediction identities must be unique.")

    job_by_identity = {
        EngineIdentity(job.engine.dataset_partition, job.engine.unit_id): job
        for job in scenario.jobs
    }
    if set(by_identity) != set(job_by_identity):
        raise ValueError("Prediction identities must exactly match the saved scenario jobs.")
    versions = {(row.model_name, row.model_version) for row in rows}
    if len(versions) != 1:
        raise ValueError("Paired comparison predictions must use exactly one model version.")
    model_name, model_version = next(iter(versions))
    if (
        model_provenance.get("model_name") != model_name
        or model_provenance.get("model_version") != model_version
    ):
        raise ValueError("Model manifest provenance does not match the prediction snapshot.")

    prediction_rows = []
    for identity, prediction in sorted(by_identity.items()):
        raw_rul = prediction.raw_rul_cycles
        if (
            prediction.error
            or prediction.error_codes
            or isinstance(raw_rul, bool)
            or not isinstance(raw_rul, Real)
            or not math.isfinite(raw_rul)
        ):
            raise ValueError(f"Engine {identity} has no successful raw prediction.")
        job = job_by_identity[identity]
        expected_rul = max(0.0, float(raw_rul))
        if job.predicted_rul_cycles != expected_rul:
            raise ValueError(f"Engine {identity} planner RUL differs from its raw prediction policy.")
        prediction_rows.append(
            {
                "dataset_partition": identity.dataset_partition,
                "engine_id": identity.unit_id,
                "observed_cycle": prediction.observed_cycle,
                "raw_rul_cycles": prediction.raw_rul_cycles,
                "planner_rul_cycles": job.predicted_rul_cycles,
                "feature_rows": prediction.input_window_rows,
                "warnings": list(prediction.warnings),
            }
        )

    def _method_payload(result):
        return {
            "feasible": result.feasible,
            "runtime_seconds": result.runtime_seconds,
            "weighted_tardiness_cycles": result.weighted_tardiness_cycles,
            "late_jobs": [
                {"dataset_partition": partition, "engine_id": unit_id}
                for partition, unit_id in result.late_jobs
            ],
            "makespan_cycles": result.makespan_cycles,
            "horizon_overflow_cycles": result.horizon_overflow_cycles,
            "schedule_report": result.report,
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

    return {
        "schema_version": PAIRED_COMPARISON_SCHEMA,
        "time_unit": "relative_cycles",
        "provenance": {
            "source": dict(source_provenance),
            "model": dict(model_provenance),
            "prediction_snapshot": prediction_rows,
        },
        "scenario": {
            "scenario_sha256": comparison.scenario_sha256,
            "dataset_partition": scenario.jobs[0].engine.dataset_partition,
            "team_count": scenario.team_count,
            "buffer_cycles": scenario.buffer_cycles,
            "horizon_cycles": scenario.horizon_cycles,
            "duration_priority_note": "durations and priority weights are user-supplied simulation assumptions",
            "jobs": [
                {
                    "dataset_partition": job.engine.dataset_partition,
                    "engine_id": job.engine.unit_id,
                    "planner_rul_cycles": job.predicted_rul_cycles,
                    "assumed_duration_cycles": job.duration_cycles,
                    "assumed_priority_weight": job.priority_weight,
                }
                for job in scenario.jobs
            ],
        },
        "comparisons": {
            "edd": _method_payload(comparison.edd),
            "genetic_algorithm": _method_payload(comparison.ga),
        },
    }
