"""Compare fixed Decision Tree settings across past-only feature windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import joblib

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import (
    build_fitting_dataset,
    build_validation_dataset,
    manifest_digest,
)
from fleetpulse.features.dataset_io import save_feature_dataset
from fleetpulse.models.train_tree import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_RANDOM_SEED,
    MODEL_VERSION,
    train_decision_tree,
)

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
WINDOWS = (10, 20, 30)
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/experiments/P11_S1_window_comparison"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = PROJECT_ROOT / "data/raw/train_FD001.txt"
    manifest_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    if _sha256(source) != TRAIN_SHA256:
        raise ValueError("Requires the reviewed original FD001 training source.")
    if OUTPUT_ROOT.exists() and any(path.name != ".gitkeep" for path in OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {OUTPUT_ROOT}")

    source_bytes = source.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = manifest_digest(manifest)
    records = load_engine_history(source)
    results = []

    for window in WINDOWS:
        config = FeatureConfig(window_cycles=window)
        fitting = build_fitting_dataset(
            records, manifest=manifest, source_sha256=TRAIN_SHA256, config=config
        )
        validation = build_validation_dataset(
            records, manifest=manifest, source_sha256=TRAIN_SHA256, config=config
        )
        if _sha256(source) != TRAIN_SHA256 or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Training source or split manifest changed during experiment.")

        result = train_decision_tree(
            fitting,
            validation,
            random_seed=DEFAULT_RANDOM_SEED,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        )
        output_dir = OUTPUT_ROOT / f"window_{window}"
        output_dir.mkdir(parents=True, exist_ok=False)
        fitting_sha256 = save_feature_dataset(fitting, output_dir / "fitting_features.json")
        validation_sha256 = save_feature_dataset(validation, output_dir / "validation_features.json")
        model_path = output_dir / "decision_tree_model.joblib"
        joblib.dump(result.model, model_path)
        prediction_rows = [
            {
                "dataset_partition": row.dataset_partition,
                "unit_id": row.unit_id,
                "cutoff_cycle": row.cutoff_cycle,
                "prediction_cycles": row.prediction,
                "true_rul_cycles": row.true_rul,
                "absolute_error_cycles": row.absolute_error,
            }
            for row in result.metrics.per_engine
        ]
        metrics = {
            "schema_version": "p11-s1-window-comparison-v1",
            "model": {"name": "decision-tree-rul", "version": MODEL_VERSION},
            "feature_config": asdict(config),
            "random_seed": result.random_seed,
            "parameters": {
                "max_depth": result.max_depth,
                "min_samples_leaf": result.min_samples_leaf,
            },
            "protocol": {
                "partition": "FD001/train",
                "fitting_engine_count": len({row.vector.engine_key for row in fitting.rows}),
                "fitting_row_count": len(fitting.rows),
                "validation_engine_count": result.metrics.engine_count,
                "split_manifest_sha256": manifest_sha256,
                "training_source_sha256": TRAIN_SHA256,
                "fitting_dataset_sha256": fitting_sha256,
                "validation_dataset_sha256": validation_sha256,
                "official_test_data_opened": False,
            },
            "metrics": {
                "mae_cycles": result.metrics.mae,
                "rmse_cycles": result.metrics.rmse,
                "fit_and_predict_runtime_seconds": result.runtime_seconds,
            },
            "model_sha256": _sha256(model_path),
            "predictions": prediction_rows,
        }
        (output_dir / "results.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        results.append(metrics)
        print(
            f"window={window} model={MODEL_VERSION} seed={result.random_seed} "
            f"MAE={result.metrics.mae:.6f} RMSE={result.metrics.rmse:.6f} "
            f"validation_engines={result.metrics.engine_count} output={output_dir.relative_to(PROJECT_ROOT)}"
        )

    if source.read_bytes() != source_bytes or manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Pinned inputs changed before experiment completion.")
    print(f"manifest_sha256: {manifest_sha256}")
    print(f"results_root: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
