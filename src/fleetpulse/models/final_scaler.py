"""Fit and persist the MLP feature scaler on every original training cycle."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from fleetpulse.data.quality import validate_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.features.config import FeatureConfig
from fleetpulse.features.engineering import engineer_features

SCHEMA_VERSION = "p12-final-mlp-scaler-v1"


@dataclass(frozen=True)
class FinalTrainingFeatures:
    """All past-only cycle vectors plus their source-engine/cycle identities."""

    matrix: np.ndarray
    row_keys: tuple[tuple[str, int, int], ...]
    per_engine_row_counts: tuple[tuple[int, int], ...]


def build_final_training_features(
    records, *, config: FeatureConfig | None = None
) -> FinalTrainingFeatures:
    """Create one feature row for every observed FD001/train engine cycle."""
    feature_config = config or FeatureConfig()
    if not isinstance(feature_config, FeatureConfig):
        raise TypeError("config must be a FeatureConfig.")
    rows = tuple(records)
    quality = validate_engine_history(rows)
    if not quality.is_valid:
        raise ValueError("Final scaler source must be a complete valid training history.")
    by_unit: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(int(row["unit_id"]), []).append(row)

    matrix_rows = []
    keys = []
    row_counts = []
    for unit_id in sorted(by_unit):
        history = sorted(by_unit[unit_id], key=lambda row: int(row["cycle"]))
        for index, row in enumerate(history):
            cycle = int(row["cycle"])
            vector = engineer_features(
                history[: index + 1],
                dataset_partition=TRAIN_PARTITION,
                unit_id=unit_id,
                selected_cycle=cycle,
                config=feature_config,
            )
            matrix_rows.append(vector.values)
            keys.append((TRAIN_PARTITION, unit_id, cycle))
        row_counts.append((unit_id, len(history)))

    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_config.output_schema):
        raise ValueError("Final training feature matrix has an unexpected shape.")
    if not np.isfinite(matrix).all():
        raise ValueError("Final training features must all be finite.")
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise ValueError("Final training feature identities must be unique and canonical.")
    return FinalTrainingFeatures(matrix, tuple(keys), tuple(row_counts))


def fit_final_scaler(features: FinalTrainingFeatures) -> StandardScaler:
    """Fit a new StandardScaler on all original training-engine feature rows."""
    if not isinstance(features, FinalTrainingFeatures) or not len(features.row_keys):
        raise ValueError("features must contain all final training rows.")
    scaler = StandardScaler().fit(features.matrix)
    if not (
        np.isfinite(scaler.mean_).all()
        and np.isfinite(scaler.scale_).all()
        and np.isfinite(scaler.var_).all()
        and np.all(scaler.scale_ > 0)
        and scaler.n_features_in_ == features.matrix.shape[1]
        and scaler.n_samples_seen_ == len(features.row_keys)
    ):
        raise ValueError("Fresh final scaler has invalid fitted statistics.")
    return scaler


def save_final_scaler_manifest(
    scaler: StandardScaler,
    features: FinalTrainingFeatures,
    *,
    config: FeatureConfig,
    training_source_sha256: str,
    selection_manifest_sha256: str,
    path: str | Path,
) -> str:
    """Write a hashed scaler manifest separately from development artifacts."""
    for name, value in (
        ("training_source_sha256", training_source_sha256),
        ("selection_manifest_sha256", selection_manifest_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")
    if scaler.n_features_in_ != len(config.output_schema):
        raise ValueError("Scaler width and feature configuration differ.")
    if scaler.n_samples_seen_ != len(features.row_keys):
        raise ValueError("Scaler sample count and final training rows differ.")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "final-training-scaler",
        "dataset_partition": TRAIN_PARTITION,
        "training_source_sha256": training_source_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "feature_config": asdict(config),
        "feature_names": list(config.output_schema),
        "training_engine_count": len(features.per_engine_row_counts),
        "training_source_row_count": len(features.row_keys),
        "per_engine_feature_row_counts": [
            {"unit_id": unit_id, "row_count": count}
            for unit_id, count in features.per_engine_row_counts
        ],
        "scaler": {
            "type": "StandardScaler",
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "feature_count": int(scaler.n_features_in_),
            "fitting_row_count": int(scaler.n_samples_seen_),
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
            "variance": [float(value) for value in scaler.var_],
        },
    }
    encoded = _canonical(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    document = {"payload": payload, "sha256": digest}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return digest


def load_final_scaler_manifest(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_config: FeatureConfig | None = None,
) -> tuple[StandardScaler, dict[str, Any]]:
    """Reload the checked scaler and its metadata for the final MLP trainer."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document.get("sha256") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("final scaler manifest content hash mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("artifact_role") != (
        "final-training-scaler"
    ):
        raise ValueError("unsupported final scaler manifest")
    if payload.get("dataset_partition") != TRAIN_PARTITION:
        raise ValueError("final scaler partition mismatch")
    for field in ("training_source_sha256", "selection_manifest_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
            raise ValueError(f"final scaler {field} is invalid")
    if expected_source_sha256 is not None and payload.get("training_source_sha256") != (
        expected_source_sha256
    ):
        raise ValueError("final scaler training source mismatch")
    config_document = payload["feature_config"]
    config = FeatureConfig(
        channels=tuple(config_document["channels"]),
        window_cycles=config_document["window_cycles"],
    )
    if config_document != json.loads(_canonical(asdict(config))):
        raise ValueError("final scaler feature configuration is invalid")
    if expected_config is not None and config != expected_config:
        raise ValueError("final scaler feature configuration mismatch")
    if payload["feature_names"] != list(config.output_schema):
        raise ValueError("final scaler feature order mismatch")

    saved = payload["scaler"]
    count = saved["feature_count"]
    if (
        saved.get("type") != "StandardScaler"
        or saved.get("with_mean") is not True
        or saved.get("with_std") is not True
        or type(count) is not int
        or count != len(config.output_schema)
    ):
        raise ValueError("unsupported final scaler parameters")
    arrays = [np.asarray(saved[name], dtype=np.float64) for name in ("mean", "scale", "variance")]
    if any(array.shape != (count,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("final scaler statistics are invalid")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
        raise ValueError("final scaler scale/variance is invalid")
    sample_count = saved["fitting_row_count"]
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("final scaler fitting count is invalid")
    if payload["training_source_row_count"] != sample_count:
        raise ValueError("final scaler source-row/fitting-row count mismatch")
    engine_rows = payload["per_engine_feature_row_counts"]
    engine_ids = [row["unit_id"] for row in engine_rows]
    if (
        len(engine_ids) != payload["training_engine_count"]
        or len(engine_ids) != len(set(engine_ids))
        or any(type(unit_id) is not int or unit_id <= 0 for unit_id in engine_ids)
        or any(type(row["row_count"]) is not int or row["row_count"] <= 0 for row in engine_rows)
    ):
        raise ValueError("final scaler per-engine row counts are invalid")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays
    scaler.n_features_in_ = count
    scaler.n_samples_seen_ = sample_count
    if sum(row["row_count"] for row in engine_rows) != sample_count:
        raise ValueError("final scaler per-engine row-count total mismatch")
    return scaler, payload


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
