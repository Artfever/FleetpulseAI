"""Versioned final bundles for the three tree estimators and mean baseline."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.constant_baseline import ConstantRULBaseline

FINAL_TREE_BUNDLE_SCHEMA = "p12-final-tree-bundle-v1"
FINAL_TREE_INDEX_SCHEMA = "p12-final-tree-bundle-index-v1"
FINAL_TREE_MODELS = {
    "decision-tree-rul": ("decision-tree-rul-v1", "decision_tree_final_v1.joblib"),
    "random-forest-rul": ("random-forest-rul-v1", "random_forest_final_v1.joblib"),
    "xgboost-rul": ("xgboost-rul-v1", "xgboost_final_v1.joblib"),
    "constant-rul": ("constant-rul-v1", None),
}
FINAL_TREE_INDEX_FILENAME = "final_tree_bundle_index.json"
_ENGINE_MODEL_TYPES = {
    "decision-tree-rul": DecisionTreeRegressor,
    "random-forest-rul": RandomForestRegressor,
    "xgboost-rul": XGBRegressor,
}


@dataclass(frozen=True)
class FinalTreeBundle:
    """Loaded final estimator or constant predictor with verified metadata."""

    name: str
    version: str
    model: Any | None
    baseline: ConstantRULBaseline | None
    feature_config: FeatureConfig
    feature_names: tuple[str, ...]
    payload: dict[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_document(payload: dict[str, Any], path: Path) -> str:
    digest = _sha256_bytes(_canonical(payload))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"payload": payload, "sha256": digest}, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return digest


def load_final_tree_bundle(manifest_path: str | Path) -> FinalTreeBundle:
    """Verify checksums, provenance, feature order, and estimator class on CPU."""
    path = Path(manifest_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _sha256_bytes(_canonical(payload)):
        raise ValueError("Final tree bundle manifest hash mismatch.")
    if payload.get("schema_version") != FINAL_TREE_BUNDLE_SCHEMA:
        raise ValueError("Unsupported final tree bundle schema.")
    name = payload.get("name")
    if name not in FINAL_TREE_MODELS or payload.get("artifact_role") != "final-training-model-bundle":
        raise ValueError("Unsupported final tree bundle identity or role.")
    version, expected_filename = FINAL_TREE_MODELS[name]
    model_info = payload.get("model", {})
    training = payload.get("training", {})
    if model_info.get("version") != version:
        raise ValueError("Final tree bundle model version mismatch.")
    config_doc = training.get("feature_config", {})
    config = FeatureConfig(
        channels=tuple(config_doc["channels"]), window_cycles=config_doc["window_cycles"]
    )
    names = tuple(training.get("feature_names", ()))
    if (
        config_doc.get("version") != config.version
        or names != config.output_schema
        or len(names) != 96
        or training.get("dataset_partition") != "train"
        or training.get("training_engine_count") != 100
        or training.get("training_row_count") != 20631
    ):
        raise ValueError("Final tree bundle feature schema or training provenance is invalid.")
    for key in ("training_source_sha256", "selection_manifest_sha256", "refit_manifest_sha256"):
        value = training.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Final tree bundle {key} is invalid.")
    rows = training.get("per_engine_row_counts")
    if (
        not isinstance(rows, list)
        or len(rows) != 100
        or len({item.get("unit_id") for item in rows}) != 100
        or any(type(item.get("unit_id")) is not int or item["unit_id"] < 1 for item in rows)
        or any(type(item.get("row_count")) is not int or item["row_count"] < 1 for item in rows)
        or {item["unit_id"] for item in rows} != set(range(1, 101))
        or sum(item["row_count"] for item in rows) != 20631
    ):
        raise ValueError("Final tree bundle per-engine training row counts are invalid.")
    parameters = training.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("Final tree bundle frozen parameters are missing.")

    if name == "constant-rul":
        if model_info.get("weights_file") is not None or model_info.get("weights_sha256") is not None:
            raise ValueError("Constant baseline must not declare a weights file.")
        baseline = ConstantRULBaseline.from_artifact(model_info["baseline"])
        if baseline.statistic != "mean" or baseline.fitting_label_count != 20631:
            raise ValueError("Final constant baseline does not match the frozen full-train mean.")
        return FinalTreeBundle(name, version, None, baseline, config, names, payload)

    if model_info.get("weights_file") != expected_filename:
        raise ValueError("Final tree bundle weights filename mismatch.")
    weights_path = path.parent / expected_filename
    expected_hash = model_info.get("weights_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(weights_path) != expected_hash:
        raise ValueError("Final tree bundle weights hash mismatch.")
    model = joblib.load(weights_path)
    expected_type = _ENGINE_MODEL_TYPES[name]
    if not isinstance(model, expected_type):
        raise TypeError(f"Final {name} bundle has an unexpected estimator class.")
    if getattr(model, "n_features_in_", None) != len(names):
        raise ValueError("Final tree estimator feature count differs from the saved schema.")
    return FinalTreeBundle(name, version, model, None, config, names, payload)


def predict_final_tree_bundle(bundle: FinalTreeBundle, features: np.ndarray) -> tuple[float, ...]:
    """Predict using the saved ordered raw feature schema; no training occurs."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(bundle.feature_names):
        raise ValueError("features must be a 2D matrix in the saved feature order.")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values.")
    if bundle.baseline is not None:
        return bundle.baseline.predict(sample_count=len(matrix))
    values = np.asarray(bundle.model.predict(matrix), dtype=np.float64)
    if values.shape != (len(matrix),) or not np.isfinite(values).all():
        raise FloatingPointError("Final tree model returned invalid predictions.")
    return tuple(float(value) for value in values)


def package_final_tree_bundles(
    refit_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, FinalTreeBundle]:
    """Create immutable, hash-checked bundles from verified P12-A1 refits."""
    source_manifest = Path(refit_manifest_path)
    source_dir = source_manifest.parent
    refit_document = json.loads(source_manifest.read_text(encoding="utf-8"))
    refit = refit_document.get("payload")
    if not isinstance(refit, dict) or refit_document.get("sha256") != _sha256_bytes(
        _canonical(refit)
    ):
        raise ValueError("P12-A1 refit manifest hash mismatch.")
    selection_doc = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    selection = load_selection_manifest(selection_manifest_path)
    if (
        refit["selection_manifest_sha256"] != selection_doc.get("sha256")
        or refit["training_source_sha256"]
        != selection_doc.get("payload", {}).get("training_source_sha256")
        or selection["decision"]["selected_candidate"] != "XGBoost, depth 4"
    ):
        raise ValueError("P12-A1 refit source/selection lineage differs from P11.")

    candidates = selection_doc["payload"]["candidates"]
    expected_candidates = {
        "decision-tree-rul": "Decision Tree, 30-cycle window",
        "random-forest-rul": "Random Forest, depth 16",
        "xgboost-rul": "XGBoost, depth 4",
    }
    models = refit["models"]
    for name, candidate_name in expected_candidates.items():
        matches = [item for item in candidates if item.get("name") == candidate_name]
        if len(matches) != 1:
            raise ValueError(f"P11 selection must include exactly one {candidate_name!r} candidate.")
        candidate = matches[0]
        model_record = models.get(name)
        expected_parameters = (
            selection["decision"]["selected_configuration"]["model_parameters"]
            if name == "xgboost-rul"
            else candidate["configuration"]
        )
        if (
            model_record is None
            or model_record.get("parameters") != expected_parameters
            or model_record.get("random_seed") != candidate.get("seed")
            or model_record.get("version") != FINAL_TREE_MODELS[name][0]
            or model_record.get("fit_rows") != 20631
        ):
            raise ValueError(f"P12-A1 {name} differs from its frozen P11 configuration.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing final tree bundle directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    feature_config = refit["feature_config"]
    selected_feature_config = selection["decision"]["selected_configuration"]["feature_config"]
    if feature_config != selected_feature_config:
        raise ValueError("P12-A1 feature config differs from the selected P11 feature contract.")
    entries: dict[str, dict[str, str]] = {}
    source_models = refit["models"]
    for name, (version, filename) in FINAL_TREE_MODELS.items():
        source_record = source_models.get(name)
        if source_record is None:
            if name != "constant-rul":
                raise ValueError(f"P12-A1 refit output is missing {name}.")
            baseline = refit["baseline"]
            baseline_file = source_dir / baseline["filename"]
            if _sha256_file(baseline_file) != baseline["sha256"]:
                raise ValueError("P12-A1 constant baseline artifact hash mismatch.")
            baseline_doc = json.loads(baseline_file.read_text(encoding="utf-8"))
            baseline_payload = baseline_doc.get("payload")
            if baseline_doc.get("sha256") != _sha256_bytes(_canonical(baseline_payload)):
                raise ValueError("P12-A1 constant baseline payload hash mismatch.")
            constant = ConstantRULBaseline.from_artifact(baseline_payload)
            if constant.statistic != "mean" or constant.fitting_label_count != 20631:
                raise ValueError("P12-A1 constant baseline is not the selected full-train mean.")
            model_info = {
                "version": version,
                "weights_file": None,
                "weights_sha256": None,
                "baseline": constant.to_artifact(),
            }
            parameters = {"statistic": "mean"}
            seed = None
        else:
            source_file = source_dir / source_record["filename"]
            if _sha256_file(source_file) != source_record["sha256"]:
                raise ValueError(f"P12-A1 {name} refit file hash mismatch.")
            if filename is None:
                raise ValueError(f"P12-A1 estimator {name} has no final weights filename.")
            final_file = output / filename
            shutil.copyfile(source_file, final_file)
            model_info = {
                "version": version,
                "weights_file": filename,
                "weights_sha256": _sha256_file(final_file),
                "baseline": None,
            }
            parameters = source_record["parameters"]
            seed = source_record["random_seed"]

        payload = {
            "schema_version": FINAL_TREE_BUNDLE_SCHEMA,
            "artifact_role": "final-training-model-bundle",
            "name": name,
            "model": model_info,
            "training": {
                "dataset_partition": "train",
                "training_source_sha256": refit["training_source_sha256"],
                "selection_manifest_sha256": refit["selection_manifest_sha256"],
                "refit_manifest_sha256": refit_document["sha256"],
                "feature_config": feature_config,
                "feature_names": refit["feature_names"],
                "training_engine_count": refit["training_engine_count"],
                "training_row_count": refit["training_row_count"],
                "per_engine_row_counts": refit["per_engine_row_counts"],
                "parameters": parameters,
                "random_seed": seed,
                "preprocessing": None,
                "target": refit["target"],
                "python_version": refit["python_version"],
                "scikit_learn_version": refit["scikit_learn_version"],
                "xgboost_version": refit["xgboost_version"],
                "official_test_data_opened": False,
                "official_test_labels_used": False,
            },
        }
        manifest_path = output / f"{name}_manifest.json"
        manifest_hash = _write_document(payload, manifest_path)
        load_final_tree_bundle(manifest_path)
        entries[name] = {"manifest_file": manifest_path.name, "manifest_sha256": manifest_hash}

    index_payload = {
        "schema_version": FINAL_TREE_INDEX_SCHEMA,
        "artifact_role": "final-tree-model-inventory",
        "training_source_sha256": refit["training_source_sha256"],
        "selection_manifest_sha256": refit["selection_manifest_sha256"],
        "refit_manifest_sha256": refit_document["sha256"],
        "bundles": entries,
    }
    _write_document(index_payload, output / FINAL_TREE_INDEX_FILENAME)
    return {name: load_final_tree_bundle(output / record["manifest_file"]) for name, record in entries.items()}
