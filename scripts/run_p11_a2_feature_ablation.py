"""Run three predeclared channel ablations with a fixed XGBoost model."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.quality import SENSOR_COLUMNS, SETTING_COLUMNS
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import (
    build_fitting_dataset,
    build_validation_dataset,
    manifest_digest,
)
from fleetpulse.features.dataset_io import save_feature_dataset
from fleetpulse.models.train_xgboost import train_xgboost

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
WINDOW_CYCLES = 30
SEED = 42
XGBOOST_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
FEATURE_CONFIGS = (
    ("all_channels", (*SETTING_COLUMNS, *SENSOR_COLUMNS)),
    ("sensors_only", SENSOR_COLUMNS),
    ("settings_only", SETTING_COLUMNS),
)
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/experiments/P11_A2_feature_ablation"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = PROJECT_ROOT / "data/raw/train_FD001.txt"
    manifest_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    if _sha256(source) != TRAIN_SHA256:
        raise ValueError("Requires the reviewed original FD001 training source.")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment output: {OUTPUT_ROOT}")
    source_bytes = source.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("P04 manifest does not identify the reviewed FD001/train source.")
    manifest_sha256 = manifest_digest(manifest)
    records = load_engine_history(source)
    expected_keys = None

    for name, channels in FEATURE_CONFIGS:
        config = FeatureConfig(channels=tuple(channels), window_cycles=WINDOW_CYCLES)
        fitting = build_fitting_dataset(
            records, manifest=manifest, source_sha256=TRAIN_SHA256, config=config
        )
        validation = build_validation_dataset(
            records, manifest=manifest, source_sha256=TRAIN_SHA256, config=config
        )
        keys = validation.row_keys
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("Feature ablations changed validation engine/cutoff membership.")
        if source.read_bytes() != source_bytes or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Training source or P04 manifest changed during feature rebuilding.")

        result = train_xgboost(
            fitting,
            validation,
            random_seed=SEED,
            **XGBOOST_PARAMETERS,
        )
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=False)
        fitting_sha256 = save_feature_dataset(fitting, output_dir / "fitting_features.json")
        validation_sha256 = save_feature_dataset(validation, output_dir / "validation_features.json")
        model_path = output_dir / "model.joblib"
        joblib.dump(result.model, model_path)
        document = {
            "schema_version": "p11-a2-feature-ablation-v1",
            "model": {"name": "xgboost-rul", "version": "xgboost-rul-v1"},
            "python_version": platform.python_version(),
            "xgboost_version": version("xgboost"),
            "seed": result.random_seed,
            "model_parameters": result.parameters,
            "feature_config": asdict(config),
            "protocol": {
                "partition": "FD001/train",
                "fitting_rows": len(fitting.rows),
                "fitting_engines": len({row.vector.engine_key for row in fitting.rows}),
                "validation_engines": result.metrics.engine_count,
                "validation_keys": validation.row_keys,
                "training_source_sha256": TRAIN_SHA256,
                "split_manifest_sha256": manifest_sha256,
                "fitting_table_sha256": fitting_sha256,
                "validation_table_sha256": validation_sha256,
                "official_test_data_opened": False,
            },
            "metrics": {
                "mae_cycles": result.metrics.mae,
                "rmse_cycles": result.metrics.rmse,
                "fit_and_predict_runtime_seconds": result.runtime_seconds,
            },
            "model_sha256": _sha256(model_path),
            "predictions": [asdict(row) for row in result.predictions],
        }
        (output_dir / "results.json").write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"{name}: channels={len(channels)} features={len(config.output_schema)} "
            f"MAE={result.metrics.mae:.6f} RMSE={result.metrics.rmse:.6f} "
            f"validation_engines={result.metrics.engine_count} "
            f"output={output_dir.relative_to(PROJECT_ROOT)}"
        )

    if source.read_bytes() != source_bytes or manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Pinned input changed before feature ablation completed.")
    print(f"split_manifest_sha256: {manifest_sha256}")
    print(f"results_root: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
