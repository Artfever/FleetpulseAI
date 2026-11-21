"""Audit and index every saved P11-S1/A1/A2 run, including weaker runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.dataset_io import load_feature_dataset

RUN_GROUPS = (
    ("P11-S1", Path("artifacts/experiments/P11_S1_window_comparison"), 3),
    ("P11-A1", Path("artifacts/experiments/P11_A1_tree_search"), 6),
    ("P11-A2", Path("artifacts/experiments/P11_A2_feature_ablation"), 3),
)
DEV_VALIDATION = PROJECT_ROOT / "artifacts/development/validation_features.json"
EXPECTED_TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
OUTPUT = PROJECT_ROOT / "artifacts/experiments/P11_A3_run_index.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_value(row: dict) -> float:
    if "prediction" in row:
        return float(row["prediction"])
    return float(row["prediction_cycles"])


def _row_key(row: dict) -> tuple[str, int, int]:
    return (
        row["dataset_partition"],
        int(row["unit_id"]),
        int(row.get("cutoff_cycle", row.get("selected_cycle"))),
    )


def audit_run(group: str, results_path: Path) -> dict:
    document = json.loads(results_path.read_text(encoding="utf-8"))
    model_path = results_path.parent / (
        "decision_tree_model.joblib"
        if (results_path.parent / "decision_tree_model.joblib").exists()
        else "model.joblib"
    )
    if not model_path.is_file() or _sha256(model_path) != document["model_sha256"]:
        raise ValueError(f"Missing or hash-mismatched model for {results_path}.")

    feature_config_data = document.get("feature_config")
    if feature_config_data is None:
        raise ValueError(f"Missing feature configuration in {results_path}.")
    config = FeatureConfig(
        channels=tuple(feature_config_data["channels"]),
        window_cycles=feature_config_data["window_cycles"],
    )
    per_run_validation = results_path.parent / "validation_features.json"
    validation_path = per_run_validation if per_run_validation.exists() else DEV_VALIDATION
    validation = load_feature_dataset(validation_path, expected_config=config)
    model = joblib.load(model_path)
    replay = model.predict(validation.X)
    prediction_rows = document["predictions"]
    if len(prediction_rows) != len(replay) or len(replay) != 20:
        raise ValueError(f"Expected 20 saved predictions for {results_path}.")
    result_keys = [_row_key(row) for row in prediction_rows]
    validation_keys = list(validation.row_keys)
    if result_keys != validation_keys or len(set(result_keys)) != 20:
        raise ValueError(f"Saved validation engine/cutoff identities changed for {results_path}.")
    if any(abs(float(actual) - _prediction_value(saved)) > 1e-12
           for actual, saved in zip(replay, prediction_rows, strict=True)):
        raise ValueError(f"Reloaded model predictions differ for {results_path}.")

    protocol = document.get("protocol", {})
    if protocol.get("training_source_sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Unexpected training source for {results_path}.")
    duration = document["metrics"].get(
        "fit_and_predict_runtime_seconds",
        document["metrics"].get("training_and_scoring_runtime_seconds"),
    )
    if duration is None or float(duration) < 0:
        raise ValueError(f"Missing or invalid fit/predict duration for {results_path}.")
    return {
        "task_group": group,
        "name": results_path.parent.name,
        "results_path": results_path.relative_to(PROJECT_ROOT).as_posix(),
        "model_path": model_path.relative_to(PROJECT_ROOT).as_posix(),
        "model_sha256": document["model_sha256"],
        "family": document.get("family", document.get("model", {}).get("name")),
        "model_version": document.get("model_version", document.get("model", {}).get("version")),
        "model_package_version": document.get(
            "model_package_version", document.get("xgboost_version")
        ),
        "seed": document.get("random_seed", document.get("seed")),
        "parameters": document.get("parameters", document.get("model_parameters")),
        "feature_config": config.__dict__,
        "validation_engine_count": len(validation.rows),
        "validation_keys": [list(key) for key in validation_keys],
        "mae_cycles": document["metrics"]["mae_cycles"],
        "rmse_cycles": document["metrics"]["rmse_cycles"],
        "fit_and_predict_seconds": float(duration),
        "saved_model_reload_predictions": "20/20 exact",
        "results_sha256": _sha256(results_path),
    }


def main() -> None:
    runs = []
    for group, relative_root, expected_count in RUN_GROUPS:
        root = PROJECT_ROOT / relative_root
        results = sorted(root.glob("*/results.json"))
        if len(results) != expected_count:
            raise ValueError(
                f"{group}: expected {expected_count} saved runs at {relative_root}, found {len(results)}."
            )
        runs.extend(audit_run(group, path) for path in results)

    document = {
        "schema_version": "p11-a3-run-index-v1",
        "training_source_sha256": EXPECTED_TRAIN_SHA256,
        "run_count": len(runs),
        "failed_run_count": 0,
        "note": "All 12 predeclared P11-S1/A1/A2 runs completed. Weaker configurations are retained; no execution failure occurred in these runs.",
        "runs": runs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"audited_runs: {len(runs)}")
    print("reload_predictions: 20/20 exact for every saved run")
    print(f"failed_runs: {document['failed_run_count']}")
    print(f"run_index: {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
