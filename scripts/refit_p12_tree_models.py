"""Refit frozen tree candidates and the selected mean baseline on all FD001/train."""

from __future__ import annotations

import hashlib
import json
import platform
from time import perf_counter

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.data.targets import calculate_training_rul
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.constant_baseline import ConstantRULBaseline
from fleetpulse.models.final_scaler import build_final_training_features

SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
SELECTION_PATH = PROJECT_ROOT / "artifacts/selection_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/final/tree/refit"
BASELINE_STATISTIC = "mean"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _candidate(selection_doc: dict, name: str) -> dict:
    candidates = selection_doc["payload"]["candidates"]
    matches = [row for row in candidates if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"P11 selection must contain exactly one candidate named {name!r}.")
    return matches[0]


def main() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    selection = load_selection_manifest(SELECTION_PATH)
    selection_doc = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection["training_source_sha256"] != source_sha256:
        raise ValueError("P11 selection and original FD001/train source hashes differ.")
    if selection["decision"]["selected_candidate"] != "XGBoost, depth 4":
        raise ValueError("P11 tree-model choice differs from the frozen depth-4 XGBoost decision.")
    selected = selection["decision"]["selected_configuration"]
    config_doc = selected["feature_config"]
    config = FeatureConfig(
        channels=tuple(config_doc["channels"]), window_cycles=config_doc["window_cycles"]
    )

    tree_candidate = _candidate(selection_doc, "Decision Tree, 30-cycle window")
    forest_candidate = _candidate(selection_doc, "Random Forest, depth 16")
    xgb_candidate = _candidate(selection_doc, "XGBoost, depth 4")
    tree_config = tree_candidate["configuration"]
    forest_config = forest_candidate["configuration"]
    xgb_config = selected["model_parameters"]
    if xgb_candidate["configuration"] != xgb_config:
        raise ValueError("P11 XGBoost candidate differs from its frozen selected configuration.")
    for candidate in (tree_candidate, forest_candidate, xgb_candidate):
        if candidate["seed"] != 42:
            raise ValueError("A P11 tree candidate has an unexpected random seed.")

    histories = load_engine_history(SOURCE_PATH)
    features = build_final_training_features(histories, config=config)
    labels = calculate_training_rul(histories)
    labels_by_key = {
        (TRAIN_PARTITION, int(row["unit_id"]), int(row["cycle"])): float(label)
        for row, label in zip(histories, labels, strict=True)
    }
    if set(labels_by_key) != set(features.row_keys):
        raise ValueError("Full-history RUL labels do not align one-to-one with the final feature rows.")
    targets = np.asarray([labels_by_key[key] for key in features.row_keys], dtype=np.float64)
    if not np.isfinite(features.matrix).all() or not np.isfinite(targets).all():
        raise ValueError("Final tree training rows and targets must be finite.")

    baseline = ConstantRULBaseline.fit(targets, statistic=BASELINE_STATISTIC)
    models = {
        "decision-tree-rul": DecisionTreeRegressor(
            random_state=tree_candidate["seed"], **tree_config
        ),
        "random-forest-rul": RandomForestRegressor(
            random_state=forest_candidate["seed"], **forest_config
        ),
        "xgboost-rul": XGBRegressor(**xgb_config),
    }
    reference_key = (TRAIN_PARTITION, 4, 164)
    try:
        reference_row = features.row_keys.index(reference_key)
    except ValueError as error:
        raise ValueError(f"Training reference row {reference_key} is missing.") from error
    sample = features.matrix[[reference_row]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_records: dict[str, dict[str, object]] = {}
    frozen_parameters = {
        "decision-tree-rul": tree_config,
        "random-forest-rul": forest_config,
        "xgboost-rul": xgb_config,
    }
    for name, model in models.items():
        started = perf_counter()
        model.fit(features.matrix, targets)
        runtime_seconds = perf_counter() - started
        prediction = float(model.predict(sample)[0])
        if not np.isfinite(prediction):
            raise FloatingPointError(f"{name} produced a non-finite training-row sanity prediction.")
        artifact_path = OUTPUT_DIR / f"{name}_refit.joblib"
        joblib.dump(model, artifact_path)
        reloaded = joblib.load(artifact_path)
        reloaded_prediction = float(reloaded.predict(sample)[0])
        if prediction != reloaded_prediction:
            raise ValueError(f"{name} did not reproduce its saved training-row prediction exactly.")
        artifact_records[name] = {
            "version": {
                "decision-tree-rul": "decision-tree-rul-v1",
                "random-forest-rul": "random-forest-rul-v1",
                "xgboost-rul": "xgboost-rul-v1",
            }[name],
            "filename": artifact_path.name,
            "sha256": _sha256_bytes(artifact_path.read_bytes()),
            "parameters": frozen_parameters[name],
            "random_seed": tree_candidate["seed"] if name == "decision-tree-rul" else (
                forest_candidate["seed"] if name == "random-forest-rul" else xgb_config["random_state"]
            ),
            "fit_rows": len(targets),
            "runtime_seconds": runtime_seconds,
            "training_row_sanity_prediction_cycles": prediction,
            "save_reload_exact_prediction": True,
        }
        print(
            f"{name}: version={artifact_records[name]['version']} rows={len(targets)} "
            f"prediction={prediction:.8f} cycles reload_exact=yes "
            f"sha256={artifact_records[name]['sha256']}"
        )

    baseline_artifact = baseline.to_artifact()
    baseline_path = OUTPUT_DIR / "constant_mean_baseline.json"
    baseline_document = {"payload": baseline_artifact, "sha256": _sha256_bytes(_canonical(baseline_artifact))}
    baseline_path.write_text(
        json.dumps(baseline_document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    reloaded_baseline = ConstantRULBaseline.from_artifact(baseline_doc["payload"])
    if baseline_doc["sha256"] != _sha256_bytes(_canonical(baseline_doc["payload"])):
        raise ValueError("Final constant baseline manifest hash did not reload correctly.")
    if reloaded_baseline != baseline:
        raise ValueError("Final constant mean baseline did not reload exactly.")
    print(
        f"constant-rul: statistic={baseline.statistic} value={baseline.value_cycles:.8f} "
        f"cycles labels={baseline.fitting_label_count} reload_exact=yes "
        f"sha256={_sha256_bytes(baseline_path.read_bytes())}"
    )

    manifest_payload = {
        "schema_version": "p12-final-tree-refit-v1",
        "artifact_role": "final-training-refit-inputs-for-p12-a2-a3",
        "dataset_partition": TRAIN_PARTITION,
        "training_source_sha256": source_sha256,
        "selection_manifest_sha256": selection_doc["sha256"],
        "feature_config": {
            "version": config.version,
            "channels": list(config.channels),
            "window_cycles": config.window_cycles,
        },
        "feature_names": list(config.output_schema),
        "training_engine_count": len(features.per_engine_row_counts),
        "training_row_count": len(features.row_keys),
        "per_engine_row_counts": [
            {"unit_id": unit_id, "row_count": count}
            for unit_id, count in features.per_engine_row_counts
        ],
        "target": "uncapped remaining cycles from complete original training lifetime",
        "baseline": {
            "filename": baseline_path.name,
            "sha256": _sha256_bytes(baseline_path.read_bytes()),
            "artifact": baseline_artifact,
            "prediction_cycles": baseline.value_cycles,
        },
        "models": artifact_records,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "official_test_data_opened": False,
        "official_test_labels_used": False,
    }
    report_path = OUTPUT_DIR / "refit_manifest.json"
    report_document = {
        "payload": manifest_payload,
        "sha256": _sha256_bytes(_canonical(manifest_payload)),
    }
    report_path.write_text(
        json.dumps(report_document, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"source_sha256: {source_sha256}")
    print(f"training_engines: {manifest_payload['training_engine_count']}")
    print(f"training_rows: {manifest_payload['training_row_count']}")
    print(f"feature_count: {len(config.output_schema)}")
    print(f"selection_manifest_sha256: {selection_doc['sha256']}")
    print(f"manifest_sha256: {report_document['sha256']}")
    print(f"output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
