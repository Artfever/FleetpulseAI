"""Load and validate saved P22 evidence for read-only application displays."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

P13_REPORT_SHA256 = "db420935dc14221c50f71f0ca3349b0667bea6b37df45b330026df3d3e8d40d4"
_EXPECTED_ERROR_MODELS = {
    "constant-rul",
    "decision-tree-rul",
    "dense-rul-mlp-final",
    "random-forest-rul",
    "xgboost-rul",
}
_EXPECTED_VALIDATION_MODELS = {
    "constant-rul",
    "decision-tree-rul",
    "dense-rul-mlp-dev",
    "random-forest-rul",
    "xgboost-rul",
}
_EXPECTED_SENSITIVITY_MODELS = {
    "constant-rul-v1",
    "decision-tree-rul-v1",
    "dense-rul-mlp-v1",
    "random-forest-rul-v1",
    "xgboost-rul-v1",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read saved P22 evidence at {path}: {error}") from error
    if not isinstance(document, dict):
        raise TypeError(f"Saved P22 evidence must be a JSON object: {path}")
    return document


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_error_analysis(path: str | Path) -> dict[str, Any]:
    """Validate the saved P22-S1 summary before showing tables or SVGs."""
    data = _read_json(path)
    if (
        data.get("schema_version") != "p22-s1-error-diagnostics-v1"
        or data.get("dataset_partition") != "FD001/test"
        or data.get("engine_count_per_model") != 100
        or data.get("model_count") != 5
        or set(data.get("models", {})) != _EXPECTED_ERROR_MODELS
    ):
        raise ValueError("P22-S1 summary does not match the verified five-model, 100-engine contract")
    bins = data.get("rul_bins")
    if not isinstance(bins, list) or len(bins) != 3:
        raise ValueError("P22-S1 true-RUL bins are missing or malformed")
    for model in data["models"].values():
        summary = model.get("summary", {})
        if summary.get("sample_count") != 100:
            raise ValueError("P22-S1 model summary has an unexpected engine count")
        if set(summary.get("by_true_rul_bin", {})) != {"low", "medium", "high"}:
            raise ValueError("P22-S1 model summary has unexpected bin names")
    return data


def load_validation_bootstrap(path: str | Path) -> dict[str, Any]:
    """Validate the P22-S2 engine-bootstrap summary and its interval semantics."""
    data = _read_json(path)
    method = data.get("method", {})
    if (
        data.get("schema_version") != "p22-s2-engine-bootstrap-v1"
        or data.get("provenance", {}).get("dataset_partition") != "FD001/train"
        or data.get("provenance", {}).get("validation_engine_count") != 20
        or method.get("replicates") != 10_000
        or method.get("seed") != 42
        or set(data.get("models", {})) != _EXPECTED_VALIDATION_MODELS
    ):
        raise ValueError("P22-S2 summary does not match its saved 20-engine bootstrap contract")
    for model in data["models"].values():
        interval = model.get("percentile_95_ci_mae_cycles")
        if (
            model.get("engine_count") != 20
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(_finite_number(value) for value in interval)
            or not _finite_number(model.get("observed_mae_cycles"))
        ):
            raise ValueError("P22-S2 model summary has invalid MAE or interval values")
    return data


def load_sensor_sensitivity(path: str | Path) -> dict[str, Any]:
    """Validate P22-S3 model movement summaries and experimental scope."""
    data = _read_json(path)
    if (
        data.get("schema_version") != 1
        or data.get("experiment") != "P22-S3 one-channel multiplicative sensitivity"
        or data.get("dataset_partition") != "FD001/train"
        or data.get("holdout_engine_count") != 20
        or data.get("feature_config", {}).get("version") != "past-only-v1"
        or data.get("feature_config", {}).get("window_cycles") != 30
        or set(data.get("protocol", {}).get("relative_changes", [])) != {0.01, 0.05}
        or len(data.get("aggregate_summary", [])) != 10
    ):
        raise ValueError("P22-S3 summary does not match its 20-engine sensitivity contract")
    for row in data["aggregate_summary"]:
        if (
            row.get("model_version") not in _EXPECTED_SENSITIVITY_MODELS
            or row.get("engine_case_count") != 840
            or not all(
                _finite_number(row.get(key))
                for key in (
                    "relative_change_magnitude",
                    "mean_absolute_prediction_change_cycles",
                    "p95_nearest_rank_absolute_prediction_change_cycles",
                    "max_absolute_prediction_change_cycles",
                )
            )
        ):
            raise ValueError("P22-S3 aggregate contains an invalid model or numeric summary")
    return data


def load_scheduler_study(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    """Verify a checksummed P22-A1/A2 snapshot sourced from frozen P13 predictions."""
    envelope = _read_json(path)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("sha256") != _canonical_sha256(payload):
        raise ValueError(f"Saved scheduler study checksum is invalid: {path}")
    if (
        payload.get("schema") != expected_schema
        or payload.get("dataset_partition") != "FD001/test"
        or payload.get("input_provenance", {}).get("report_sha256") != P13_REPORT_SHA256
        or payload.get("input_provenance", {}).get("model_version") != "decision-tree-rul-v1"
        or len(payload.get("scenarios", [])) != 18
    ):
        raise ValueError(f"Saved P22 scheduler study does not match its P13 input contract: {path}")
    if expected_schema == "p22-a1-fixed-fleet-scenarios-v1":
        if payload.get("ga_outcomes") != {"equal": 12, "better": 6}:
            raise ValueError("P22-A1 aggregate outcomes do not match its verified scenario report")
    elif expected_schema == "p22-a2-seed-comparison-v1":
        if (
            payload.get("case_count") != 18
            or payload.get("ga_run_count") != 90
            or payload.get("outcome_counts") != {"equal": 60, "better": 30}
            or payload.get("all_schedules_feasible") is not True
            or payload.get("all_seed_repeats_match") is not True
        ):
            raise ValueError("P22-A2 aggregate outcomes do not match its verified seed report")
    else:
        raise ValueError(f"Unsupported P22 scheduler study schema: {expected_schema}")
    return payload
