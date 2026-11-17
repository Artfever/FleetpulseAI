"""Run the predeclared small RF/XGBoost search for P11-A1."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.train_forest import train_random_forest
from fleetpulse.models.train_xgboost import train_xgboost

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
FEATURE_CONFIG = FeatureConfig(window_cycles=30)
RANDOM_FOREST_CONFIGS = (
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
)
XGBOOST_CONFIGS = (
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05},
)
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/experiments/P11_A1_tree_search"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_one(family, config, fitting, validation, output_dir):
    if family == "random_forest":
        result = train_random_forest(fitting, validation, **config)
        model_version = "random-forest-rul-v1"
        package_version = version("scikit-learn")
        parameters = result.parameters
    else:
        result = train_xgboost(fitting, validation, **config)
        model_version = "xgboost-rul-v1"
        package_version = version("xgboost")
        parameters = result.parameters

    model_path = output_dir / "model.joblib"
    joblib.dump(result.model, model_path)
    report = {
        "schema_version": "p11-a1-tree-search-v1",
        "family": family,
        "model_version": model_version,
        "python_version": platform.python_version(),
        "model_package_version": package_version,
        "random_seed": result.random_seed,
        "parameters": parameters,
        "feature_config": asdict(FEATURE_CONFIG),
        "protocol": {
            "partition": "FD001/train",
            "fitting_rows": len(fitting.rows),
            "fitting_engines": len({row.vector.engine_key for row in fitting.rows}),
            "validation_engines": result.metrics.engine_count,
            "training_source_sha256": TRAIN_SHA256,
            "split_manifest_sha256": fitting.split_manifest_sha256,
            "fitting_table_sha256": _sha256(
                PROJECT_ROOT / "artifacts/development/fitting_features.json"
            ),
            "validation_table_sha256": _sha256(
                PROJECT_ROOT / "artifacts/development/validation_features.json"
            ),
            "official_test_data_opened": False,
        },
        "metrics": {
            "mae_cycles": result.metrics.mae,
            "rmse_cycles": result.metrics.rmse,
            "fit_and_predict_runtime_seconds": result.runtime_seconds,
        },
        "model_sha256": _sha256(model_path),
        "predictions": [asdict(prediction) for prediction in result.predictions],
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts/development"
    manifest_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("P04 manifest does not identify the reviewed FD001/train source.")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment output: {OUTPUT_ROOT}")

    expected_manifest_sha256 = manifest_digest(manifest)
    tables = {
        role: load_feature_dataset(
            artifact_dir / f"{role}_features.json",
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_config=FEATURE_CONFIG,
        )
        for role in ("fitting", "validation")
    }
    fitting, validation = tables["fitting"], tables["validation"]
    results = []
    configurations = (
        *(('random_forest', config) for config in RANDOM_FOREST_CONFIGS),
        *(('xgboost', config) for config in XGBOOST_CONFIGS),
    )
    for index, (family, config) in enumerate(configurations, start=1):
        output_dir = OUTPUT_ROOT / f"{index:02d}_{family}_depth_{config['max_depth']}"
        output_dir.mkdir(parents=True, exist_ok=False)
        report = _run_one(family, config, fitting, validation, output_dir)
        results.append(report)
        print(
            f"{family} depth={config['max_depth']} MAE={report['metrics']['mae_cycles']:.6f} "
            f"RMSE={report['metrics']['rmse_cycles']:.6f} "
            f"runtime={report['metrics']['fit_and_predict_runtime_seconds']:.3f}s "
            f"output={output_dir.relative_to(PROJECT_ROOT)}"
        )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("P04 split manifest changed during the experiments.")
    print(f"completed_configurations: {len(results)}")
    print(f"output_root: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
