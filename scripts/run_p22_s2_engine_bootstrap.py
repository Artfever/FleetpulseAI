"""Estimate P22-S2 validation MAE variability by resampling whole engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetpulse import PROJECT_ROOT
from fleetpulse.features.build_dataset import manifest_digest

DEV_DIR = PROJECT_ROOT / "artifacts/development"
SPLIT_PATH = PROJECT_ROOT / "artifacts/split_manifest.json"
TRAIN_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
VALIDATION_FEATURES_PATH = DEV_DIR / "validation_features.json"
EXPECTED_TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
EXPECTED_SPLIT_SHA256 = "6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
MODEL_INPUTS = (
    {
        "key": "decision-tree-rul",
        "display_name": "Decision Tree",
        "manifest_path": DEV_DIR / "decision_tree_validation.json",
        "weights_path": DEV_DIR / "decision_tree_model.joblib",
        "version": "decision-tree-rul-v1",
        "prediction_key": "prediction_cycles",
    },
    {
        "key": "random-forest-rul",
        "display_name": "Random Forest",
        "manifest_path": DEV_DIR / "random_forest_validation.json",
        "weights_path": DEV_DIR / "random_forest_model.joblib",
        "version": "random-forest-rul-v1",
        "prediction_key": "prediction_cycles",
    },
    {
        "key": "xgboost-rul",
        "display_name": "XGBoost",
        "manifest_path": DEV_DIR / "xgboost_validation.json",
        "weights_path": DEV_DIR / "xgboost_model.joblib",
        "version": "xgboost-rul-v1",
        "prediction_key": "prediction_cycles",
    },
    {
        "key": "constant-rul",
        "display_name": "Constant mean baseline",
        "manifest_path": DEV_DIR / "constant_rul_validation.json",
        "weights_path": None,
        "version": "constant-rul-v1",
        "prediction_key": "prediction_cycles",
    },
    {
        "key": "dense-rul-mlp-dev",
        "display_name": "Dense RUL MLP development",
        "manifest_path": DEV_DIR / "mlp/mlp_training_manifest.json",
        "weights_path": DEV_DIR / "mlp/mlp_best_state.pt",
        "version": "dense-rul-mlp-v1",
        "prediction_key": "prediction_cycles",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_verified_document(path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _canonical_sha256(payload):
        raise ValueError(f"Invalid artifact payload checksum: {path}")
    return payload, document["sha256"]


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile, equivalent to the common type-7 rule."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mae_distributions(
    absolute_errors: Mapping[str, Sequence[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    """Resample common whole-engine indices to keep candidate rows paired."""
    if not absolute_errors:
        raise ValueError("at least one model is required")
    model_names = tuple(absolute_errors)
    sample_count = len(absolute_errors[model_names[0]])
    if sample_count < 2:
        raise ValueError("at least two validation engines are required")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    for name, values in absolute_errors.items():
        if len(values) != sample_count:
            raise ValueError(f"model {name} does not have the shared engine count")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"model {name} contains an invalid absolute error")

    generator = random.Random(seed)
    distributions = {name: [] for name in model_names}
    for _ in range(replicates):
        indices = [generator.randrange(sample_count) for _ in range(sample_count)]
        for name, values in absolute_errors.items():
            distributions[name].append(sum(values[index] for index in indices) / sample_count)
    return distributions


def _load_validation_rows() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    for path in (TRAIN_PATH, SPLIT_PATH, VALIDATION_FEATURES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required local validation input is missing: {path}")
    train_sha256 = _sha256(TRAIN_PATH)
    split_document = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_sha256 = manifest_digest(split_document)
    if train_sha256 != EXPECTED_TRAIN_SHA256 or split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("FD001/train source or split manifest differs from the frozen P06 inputs")

    feature_payload, feature_sha256 = _read_verified_document(VALIDATION_FEATURES_PATH)
    if (
        feature_payload.get("role") != "validation"
        or feature_payload.get("feature_config", {}).get("version") != "past-only-v1"
        or feature_payload.get("feature_config", {}).get("window_cycles") != 30
        or feature_payload.get("input_hashes", {}).get("source_sha256") != train_sha256
        or feature_payload.get("input_hashes", {}).get("split_manifest_sha256") != split_sha256
    ):
        raise ValueError("Saved validation features do not match the audited source/split contract")
    feature_rows = feature_payload.get("rows")
    split_rows = split_document.get("validation_prefixes")
    if not isinstance(feature_rows, list) or len(feature_rows) != 20:
        raise ValueError("Expected exactly 20 saved validation feature rows")
    if not isinstance(split_rows, list) or len(split_rows) != 20:
        raise ValueError("Expected exactly 20 validation prefixes in the split manifest")
    expected_by_id = {row["unit_id"]: row for row in split_rows}
    if len(expected_by_id) != 20:
        raise ValueError("Split manifest contains duplicate validation engine IDs")
    feature_by_id = {row["metadata"]["unit_id"]: row for row in feature_rows}
    if set(feature_by_id) != set(expected_by_id):
        raise ValueError("Saved validation features do not match split engine IDs")
    for unit_id, row in feature_by_id.items():
        split_row = expected_by_id[unit_id]
        if (
            row["key"] != ["FD001/train", unit_id, split_row["cutoff_cycle"]]
            or row["true_rul"] != split_row["target_rul"]
            or row["metadata"]["selected_cycle"] != split_row["cutoff_cycle"]
            or row["metadata"]["observed_row_count"] != split_row["prefix_row_count"]
        ):
            raise ValueError(f"Validation feature key/label mismatch for engine {unit_id}")

    validation_rows = {}
    artifact_records = []
    for spec in MODEL_INPUTS:
        if not spec["manifest_path"].is_file():
            raise FileNotFoundError(f"Missing saved validation manifest: {spec['manifest_path']}")
        payload, manifest_payload_sha256 = _read_verified_document(spec["manifest_path"])
        is_constant = spec["key"] == "constant-rul"
        if is_constant:
            model_identity = payload.get("baseline", {})
            evaluation = payload.get("evaluation", {})
            training = evaluation
            model_file_sha256 = None
        else:
            model_identity = payload.get("model", {})
            training = payload.get("training", {})
            evaluation = payload.get("evaluation", {})
            if not spec["weights_path"].is_file():
                raise FileNotFoundError(f"Missing saved development model: {spec['weights_path']}")
            model_file_sha256 = _sha256(spec["weights_path"])
            if model_file_sha256 != model_identity.get("sha256"):
                raise ValueError(f"Saved model bytes do not match manifest for {spec['key']}")
        model_version = model_identity.get("version") or model_identity.get("artifact_version")
        if model_version != spec["version"]:
            raise ValueError(f"Unexpected development model version for {spec['key']}: {model_version}")
        if (
            training.get("source_sha256") != train_sha256
            or training.get("split_manifest_sha256") != split_sha256
            or training.get("validation_dataset_sha256") != feature_sha256
        ):
            raise ValueError(f"Source, split or validation feature hash mismatch for {spec['key']}")
        if not is_constant:
            config = training.get("feature_config", {})
            if config.get("version") != "past-only-v1" or config.get("window_cycles") != 30:
                raise ValueError(f"Unexpected feature configuration for {spec['key']}")
        raw_rows = evaluation.get("predictions")
        if not isinstance(raw_rows, list) or len(raw_rows) != 20:
            raise ValueError(f"Expected 20 saved validation predictions for {spec['key']}")
        rows_by_id = {}
        for row in raw_rows:
            unit_id = row.get("unit_id")
            if unit_id in rows_by_id:
                raise ValueError(f"Duplicate validation prediction for {spec['key']} engine {unit_id}")
            split_row = expected_by_id.get(unit_id)
            prediction = row.get(spec["prediction_key"])
            if (
                split_row is None
                or row.get("dataset_partition", row.get("partition")) != "FD001/train"
                or row.get("cutoff_cycle") != split_row["cutoff_cycle"]
                or row.get("true_rul_cycles") != split_row["target_rul"]
                or type(prediction) not in (int, float)
                or not math.isfinite(prediction)
            ):
                raise ValueError(f"Validation identity/label/prediction mismatch for {spec['key']} engine {unit_id}")
            rows_by_id[unit_id] = {
                "unit_id": unit_id,
                "dataset_partition": "FD001/train",
                "cutoff_cycle": row["cutoff_cycle"],
                "true_rul_cycles": row["true_rul_cycles"],
                "raw_prediction_cycles": float(prediction),
                "absolute_error_cycles": abs(float(prediction) - row["true_rul_cycles"]),
            }
        if set(rows_by_id) != set(expected_by_id):
            raise ValueError(f"Incomplete validation engine inventory for {spec['key']}")
        validation_rows[spec["key"]] = [rows_by_id[unit_id] for unit_id in sorted(rows_by_id)]
        artifact_records.append(
            {
                "name": spec["key"],
                "display_name": spec["display_name"],
                "version": model_version,
                "manifest_file": spec["manifest_path"].relative_to(PROJECT_ROOT).as_posix(),
                "manifest_file_sha256": _sha256(spec["manifest_path"]),
                "manifest_payload_sha256": manifest_payload_sha256,
                "model_file": spec["weights_path"].relative_to(PROJECT_ROOT).as_posix()
                if spec["weights_path"] is not None
                else None,
                "model_file_sha256": model_file_sha256,
                "random_seed": training.get("random_seed"),
                "training_parameters": training.get("parameters"),
                "saved_validation_mae_cycles": evaluation.get("mae_cycles"),
            }
        )

    provenance = {
        "dataset_partition": "FD001/train",
        "validation_engine_count": len(expected_by_id),
        "train_source_sha256": train_sha256,
        "split_manifest_sha256": split_sha256,
        "validation_features_file_sha256": _sha256(VALIDATION_FEATURES_PATH),
        "validation_features_payload_sha256": feature_sha256,
        "validation_feature_config": feature_payload["feature_config"],
        "models": artifact_records,
    }
    return provenance, validation_rows


def _summarize(
    provenance: Mapping[str, Any],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    errors = {
        name: [float(row["absolute_error_cycles"]) for row in rows]
        for name, rows in rows_by_model.items()
    }
    distributions = bootstrap_mae_distributions(errors, replicates=replicates, seed=seed)
    model_inputs = {item["name"]: item for item in provenance["models"]}
    summaries = {}
    for name, sample_errors in errors.items():
        boot = distributions[name]
        saved_mae = model_inputs[name]["saved_validation_mae_cycles"]
        observed_mae = statistics.fmean(sample_errors)
        if saved_mae is not None and not math.isclose(
            observed_mae, float(saved_mae), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Recomputed validation MAE differs from saved result for {name}")
        summaries[name] = {
            "display_name": model_inputs[name]["display_name"],
            "model_version": model_inputs[name]["version"],
            "engine_count": len(sample_errors),
            "observed_mae_cycles": observed_mae,
            "saved_validation_mae_cycles": saved_mae,
            "observed_rmse_cycles": math.sqrt(
                statistics.fmean(row["absolute_error_cycles"] ** 2 for row in rows_by_model[name])
            ),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_mean_mae_cycles": statistics.fmean(boot),
            "bootstrap_standard_error_cycles": statistics.stdev(boot),
            "percentile_95_ci_mae_cycles": [percentile(boot, 0.025), percentile(boot, 0.975)],
        }
    return {
        "schema_version": "p22-s2-engine-bootstrap-v1",
        "method": {
            "unit_of_resampling": "whole validation engine prefix",
            "resampling": "sample 20 engine indices with replacement for each replicate",
            "paired_across_models": True,
            "replicates": replicates,
            "seed": seed,
            "python_version": platform.python_version(),
            "interval": "95% percentile interval, linear-interpolated quantiles",
            "metric": "mean absolute raw RUL error in cycles",
        },
        "provenance": dict(provenance),
        "models": summaries,
        "limits": [
            "These are five saved development-model validation results on 20 FD001/train engine prefixes.",
            "The interval summarizes engine-level validation MAE sampling variability; it is not an individual prediction interval or a probability that a specific engine will fail.",
            "All five candidates share the same held-out engine sample; paired resampling keeps model rows aligned.",
            "This resampling study is descriptive and is not a model selection, tuning, official-test or maintenance step.",
            "NASA C-MAPSS is simulated data; these results do not establish real fleet performance.",
        ],
    }


def run_bootstrap(
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    provenance, rows = _load_validation_rows()
    return _summarize(provenance, rows, replicates=replicates, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/p22_s2_validation_bootstrap.json",
    )
    args = parser.parse_args()
    result = run_bootstrap(replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
