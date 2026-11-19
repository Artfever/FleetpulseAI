"""Run bounded sensor-perturbation checks on frozen development holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.quality import SENSOR_COLUMNS
from fleetpulse.data.split_audit import audit_split_manifest
from fleetpulse.features import engineer_features
from fleetpulse.models.constant_baseline import ConstantRULBaseline
from fleetpulse.services.predictor import _load_model

DEV_DIR = PROJECT_ROOT / "artifacts/development"
TRAIN_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
SPLIT_PATH = PROJECT_ROOT / "artifacts/split_manifest.json"
VALIDATION_FEATURES_PATH = DEV_DIR / "validation_features.json"
SOURCE_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
SPLIT_SHA256 = "6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92"
VALIDATION_FEATURES_SHA256 = "f843a342a3f8799a22a1d386936e9f4c2ab57b6f7cb30cc21885a2b989fcdff7"
FEATURE_VERSION = "past-only-v1"
WINDOW_CYCLES = 30
RELATIVE_CHANGES = (0.01, 0.05)
BASELINE_PARITY_ABS_TOL = 1e-4
MODEL_SPECS = (
    ("decision-tree-rul-v1", DEV_DIR / "decision_tree_validation.json"),
    ("random-forest-rul-v1", DEV_DIR / "random_forest_validation.json"),
    ("xgboost-rul-v1", DEV_DIR / "xgboost_validation.json"),
    ("dense-rul-mlp-v1", DEV_DIR / "mlp/mlp_training_manifest.json"),
)
CONSTANT_MANIFEST = DEV_DIR / "constant_rul_validation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def perturb_sensor_history(
    history: Sequence[Mapping[str, object]],
    *,
    sensor: str,
    relative_change: float,
    first_affected_cycle: int,
) -> list[dict[str, object]]:
    """Apply one multiplicative sensor-only change to observed window rows."""
    if sensor not in SENSOR_COLUMNS:
        raise ValueError("sensor must be one of the declared sensor channels")
    if (
        isinstance(relative_change, bool)
        or not isinstance(relative_change, (int, float))
        or not math.isfinite(relative_change)
        or not -0.05 <= relative_change <= 0.05
        or relative_change == 0
    ):
        raise ValueError("relative_change must be nonzero and within ±5%")
    if type(first_affected_cycle) is not int or first_affected_cycle < 1:
        raise ValueError("first_affected_cycle must be a positive integer")

    changed: list[dict[str, object]] = []
    for row in history:
        copy = dict(row)
        cycle = row.get("cycle")
        if type(cycle) is not int:
            raise ValueError("history cycle values must be integers")
        if cycle >= first_affected_cycle:
            value = row.get(sensor)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{sensor} values must be numeric")
            shifted = float(value) * (1.0 + relative_change)
            if not math.isfinite(shifted):
                raise ValueError("perturbation produced a non-finite reading")
            copy[sensor] = shifted
        changed.append(copy)
    return changed


def summarize_deltas(deltas: Sequence[float]) -> dict[str, float | int]:
    """Summarize raw prediction movement in cycles, never model accuracy."""
    if not deltas or any(not math.isfinite(value) for value in deltas):
        raise ValueError("deltas must be a nonempty sequence of finite values")
    absolute = [abs(value) for value in deltas]
    return {
        "engine_count": len(deltas),
        "mean_absolute_prediction_change_cycles": math.fsum(absolute) / len(absolute),
        "median_absolute_prediction_change_cycles": statistics.median(absolute),
        "max_absolute_prediction_change_cycles": max(absolute),
        "engines_with_changed_prediction": sum(value != 0 for value in deltas),
    }


def _predict_loaded(
    history: Sequence[Mapping[str, object]],
    *,
    unit_id: int,
    config: Any,
    model: Any,
) -> float:
    """Use the service-validated saved model/config without rehashing per case."""
    vector = engineer_features(
        history,
        dataset_partition="FD001/train",
        unit_id=unit_id,
        config=config,
    )
    predictions = model.predict((vector.values,))
    if len(predictions) != 1 or not math.isfinite(float(predictions[0])):
        raise ValueError("Saved development model returned an invalid raw RUL")
    return float(predictions[0])


def _load_inputs() -> tuple[
    list[dict[str, int | float]], list[dict[str, Any]], dict[str, str], dict[str, Any]
]:
    for path in (
        TRAIN_PATH,
        SPLIT_PATH,
        VALIDATION_FEATURES_PATH,
        *(manifest for _, manifest in MODEL_SPECS),
        CONSTANT_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required P06 development input is missing: {path}")
    source_hash = _sha256(TRAIN_PATH)
    if source_hash != SOURCE_SHA256:
        raise ValueError("FD001/train bytes differ from the verified P06 development source")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if _canonical_digest(split) != SPLIT_SHA256:
        raise ValueError("P06 split manifest differs from the verified development split")
    records = load_engine_history(TRAIN_PATH)
    audit_split_manifest(split, records, source_sha256=source_hash)
    validation_document = json.loads(VALIDATION_FEATURES_PATH.read_text(encoding="utf-8"))
    validation_features = validation_document.get("payload", {})
    if validation_document.get("sha256") != _canonical_digest(validation_features):
        raise ValueError("Saved P06 validation feature table checksum is invalid")
    if (
        validation_document["sha256"] != VALIDATION_FEATURES_SHA256
        or validation_features.get("role") != "validation"
        or validation_features.get("feature_config", {}).get("version") != FEATURE_VERSION
        or validation_features.get("feature_config", {}).get("window_cycles") != WINDOW_CYCLES
        or len(validation_features.get("rows", [])) != 20
    ):
        raise ValueError("Saved validation features differ from the verified P06 contract")
    prefixes = split.get("validation_prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != 20:
        raise ValueError("Expected the original 20-engine P06 validation holdout")
    checksums = {"train_source_sha256": source_hash, "split_manifest_sha256": SPLIT_SHA256}
    return records, prefixes, checksums, validation_features


def run_experiment() -> dict[str, Any]:
    """Compare baseline and one-sensor-at-a-time perturbed raw predictions."""
    records, prefixes, provenance, validation_features = _load_inputs()
    histories: dict[int, list[dict[str, int | float]]] = {}
    for row in records:
        histories.setdefault(int(row["unit_id"]), []).append(row)

    prediction_manifests: dict[str, dict[int, float]] = {}
    manifest_hashes: dict[str, str] = {}
    model_file_hashes: dict[str, str] = {}
    loaded_models: dict[str, tuple[Any, Any]] = {}
    for version, path in MODEL_SPECS:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document.get("payload", {})
        if document.get("sha256") != _canonical_digest(payload):
            raise ValueError(f"{version} manifest payload checksum is invalid")
        prediction_rows = payload.get("evaluation", {}).get("predictions", [])
        predictions = {int(row["unit_id"]): float(row["prediction_cycles"]) for row in prediction_rows}
        if len(predictions) != 20:
            raise ValueError(f"{version} must have one saved prediction for each holdout engine")
        prediction_manifests[version] = predictions
        manifest_hashes[version] = _sha256(path)
        model_file_hashes[version] = str(payload["model"]["sha256"])
        loaded_models[version] = _load_model(path)
        if (
            loaded_models[version][0].version != FEATURE_VERSION
            or loaded_models[version][0].window_cycles != WINDOW_CYCLES
        ):
            raise ValueError(f"{version} uses an unexpected feature configuration")

    constant_doc = json.loads(CONSTANT_MANIFEST.read_text(encoding="utf-8"))
    constant_payload = constant_doc.get("payload", {})
    if constant_doc.get("sha256") != _canonical_digest(constant_payload):
        raise ValueError("Constant baseline manifest payload checksum is invalid")
    constant = ConstantRULBaseline.from_artifact(constant_payload["baseline"])
    constant_rows = constant_payload["evaluation"]["predictions"]
    constant_predictions = {int(row["unit_id"]): float(row["prediction_cycles"]) for row in constant_rows}
    if len(constant_predictions) != 20:
        raise ValueError("Constant baseline must have one saved prediction for each holdout engine")
    prediction_manifests["constant-rul-v1"] = constant_predictions
    manifest_hashes["constant-rul-v1"] = _sha256(CONSTANT_MANIFEST)
    model_file_hashes["constant-rul-v1"] = "not_applicable_constant_baseline"

    baseline_by_model: dict[str, dict[int, float]] = {version: {} for version in prediction_manifests}
    for entry in prefixes:
        unit_id = int(entry["unit_id"])
        cutoff = int(entry["cutoff_cycle"])
        prefix = [row for row in histories[unit_id] if int(row["cycle"]) <= cutoff]
        if len(prefix) != int(entry["prefix_row_count"]) or int(prefix[-1]["cycle"]) != cutoff:
            raise ValueError(f"Validation cutoff/prefix mismatch for engine {unit_id}")
        for version, _ in MODEL_SPECS:
            config, model, _, _ = loaded_models[version]
            predicted = _predict_loaded(prefix, unit_id=unit_id, config=config, model=model)
            if version == MODEL_SPECS[0][0]:
                validation_row = next(
                    (
                        row
                        for row in validation_features["rows"]
                        if row.get("key") == ["FD001/train", unit_id, cutoff]
                    ),
                    None,
                )
                vector = engineer_features(
                    prefix,
                    dataset_partition="FD001/train",
                    unit_id=unit_id,
                    config=config,
                )
                if validation_row is None or tuple(validation_row["features"]) != vector.values:
                    raise ValueError(f"Rebuilt P06 features differ for validation engine {unit_id}")
            expected = prediction_manifests[version][unit_id]
            if not math.isclose(
                predicted, expected, rel_tol=0.0, abs_tol=BASELINE_PARITY_ABS_TOL
            ):
                raise ValueError(f"{version} baseline does not reproduce saved P06 output for {unit_id}")
            baseline_by_model[version][unit_id] = predicted
        baseline_by_model["constant-rul-v1"][unit_id] = constant.predict(sample_count=1)[0]
        if not math.isclose(
            baseline_by_model["constant-rul-v1"][unit_id],
            prediction_manifests["constant-rul-v1"][unit_id],
            rel_tol=0.0,
            abs_tol=BASELINE_PARITY_ABS_TOL,
        ):
            raise ValueError(f"Constant baseline does not reproduce saved output for {unit_id}")

    model_results: list[dict[str, Any]] = []
    for version, _ in (*MODEL_SPECS, ("constant-rul-v1", CONSTANT_MANIFEST)):
        baseline = baseline_by_model[version]
        for sensor in SENSOR_COLUMNS:
            for magnitude in RELATIVE_CHANGES:
                for direction in (-1, 1):
                    relative_change = direction * magnitude
                    per_engine_delta_cycles = []
                    for entry in prefixes:
                        unit_id = int(entry["unit_id"])
                        cutoff = int(entry["cutoff_cycle"])
                        prefix = [row for row in histories[unit_id] if int(row["cycle"]) <= cutoff]
                        first_window_cycle = max(1, cutoff - WINDOW_CYCLES + 1)
                        changed = perturb_sensor_history(
                            prefix,
                            sensor=sensor,
                            relative_change=relative_change,
                            first_affected_cycle=first_window_cycle,
                        )
                        if version == "constant-rul-v1":
                            prediction = constant.predict(sample_count=1)[0]
                        else:
                            config, model, _, _ = loaded_models[version]
                            prediction = _predict_loaded(
                                changed, unit_id=unit_id, config=config, model=model
                            )
                        delta = prediction - baseline[unit_id]
                        per_engine_delta_cycles.append([unit_id, cutoff, delta])
                    deltas = [item[2] for item in per_engine_delta_cycles]
                    model_results.append(
                        {
                            "model_version": version,
                            "manifest_sha256": manifest_hashes[version],
                            "sensor": sensor,
                            "relative_change": relative_change,
                            "summary": summarize_deltas(deltas),
                            "per_engine_delta_cycles": per_engine_delta_cycles,
                        }
                    )

    aggregate_summary = []
    for version in prediction_manifests:
        for magnitude in RELATIVE_CHANGES:
            absolute_deltas = sorted(
                abs(engine[2])
                for result in model_results
                if result["model_version"] == version
                and abs(result["relative_change"]) == magnitude
                for engine in result["per_engine_delta_cycles"]
            )
            aggregate_summary.append(
                {
                    "model_version": version,
                    "relative_change_magnitude": magnitude,
                    "engine_case_count": len(absolute_deltas),
                    "mean_absolute_prediction_change_cycles": math.fsum(absolute_deltas)
                    / len(absolute_deltas),
                    "p95_nearest_rank_absolute_prediction_change_cycles": absolute_deltas[
                        math.ceil(0.95 * len(absolute_deltas)) - 1
                    ],
                    "max_absolute_prediction_change_cycles": absolute_deltas[-1],
                    "changed_output_count": sum(value != 0.0 for value in absolute_deltas),
                }
            )

    return {
        "schema_version": 1,
        "experiment": "P22-S3 one-channel multiplicative sensitivity",
        "dataset_partition": "FD001/train",
        "holdout_engine_count": 20,
        "feature_config": {
            "version": FEATURE_VERSION,
            "window_cycles": WINDOW_CYCLES,
            "window_rule": "observed rows with cycles in [max(1, cutoff-29), cutoff]",
            "channels_perturbed": list(SENSOR_COLUMNS),
            "setting_channels_perturbed": [],
        },
        "protocol": {
            "relative_changes": list(RELATIVE_CHANGES),
            "direction": "both negative and positive for each magnitude",
            "application": "multiply one sensor channel by 1+relative_change on observed feature-window rows only",
            "evaluation_value": "raw, unclipped RUL prediction change in cycles",
            "candidate_scope": "frozen P06 development models; no fitting, tuning, or official-test labels",
            "constant_baseline_role": "unchanged-input control; it has no sensor-dependent inputs",
        },
        "provenance": {
            **provenance,
            "model_manifest_sha256": manifest_hashes,
            "model_file_sha256": model_file_hashes,
            "feature_validation_table_payload_sha256": VALIDATION_FEATURES_SHA256,
        },
        "baseline_predictions": [
            {
                "unit_id": int(entry["unit_id"]),
                "cutoff_cycle": int(entry["cutoff_cycle"]),
                "raw_rul_cycles_by_model": {
                    version: baseline_by_model[version][int(entry["unit_id"])]
                    for version in baseline_by_model
                },
            }
            for entry in prefixes
        ],
        "aggregate_summary": aggregate_summary,
        "results": model_results,
        "limitations": [
            "FD001 is simulated data and these 20 engines are development validation holdouts.",
            "Changes are artificial relative shifts, not measurement-noise estimates or plausible failure interventions.",
            "Prediction movement is conditional on frozen model fits; it is not accuracy, confidence, a causal effect, or a maintenance recommendation.",
            "The constant-mean baseline is included only as an unchanged-input control.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/p22_s3_sensor_sensitivity.json",
        help="JSON path for verified predictions and perturbation summaries",
    )
    args = parser.parse_args()
    result = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['results'])} sensitivity cases for {result['holdout_engine_count']} engines to {args.output}")


if __name__ == "__main__":
    main()
