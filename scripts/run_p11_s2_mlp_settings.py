"""Compare three predeclared MLP widths on the unchanged P10 split."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import sklearn
import torch

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation import ValidationPrediction, calculate_validation_metrics
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.mlp import train_mlp
from fleetpulse.models.mlp_artifact import save_mlp_artifact
from fleetpulse.models.mlp_contract import MODEL_VERSION

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
HIDDEN_DIMS = ((32, 16), (64, 32), (128, 64))
SEED = 42
MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/experiments/P11_S2_mlp_settings"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts/development"
    manifest_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["source"]["sha256"] != TRAIN_SHA256:
        raise ValueError("P04 manifest does not identify the reviewed FD001/train source.")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing MLP experiment: {OUTPUT_ROOT}")

    expected_manifest_sha256 = manifest_digest(manifest)
    table_paths = {
        role: artifact_dir / f"{role}_features.json" for role in ("fitting", "validation")
    }
    tables = {
        role: load_feature_dataset(
            path,
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_config=FeatureConfig(),
        )
        for role, path in table_paths.items()
    }
    table_hashes = {
        role: json.loads(path.read_text(encoding="utf-8"))["sha256"]
        for role, path in table_paths.items()
    }

    for hidden_dims in HIDDEN_DIMS:
        name = "width_" + "_".join(map(str, hidden_dims))
        output_dir = OUTPUT_ROOT / name
        result = train_mlp(
            tables["fitting"],
            tables["validation"],
            hidden_dims=hidden_dims,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            min_delta=0.0,
            learning_rate=LEARNING_RATE,
            random_seed=SEED,
        )
        saved = save_mlp_artifact(
            result,
            tables["fitting"],
            tables["validation"],
            output_dir,
            fitting_dataset_sha256=table_hashes["fitting"],
            validation_dataset_sha256=table_hashes["validation"],
        )
        verification_process = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/verify_mlp_artifact.py"),
                str(saved["manifest_path"]),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        verification = json.loads(verification_process.stdout)
        if verification["prediction_count"] != 20 or verification["max_prediction_difference_cycles"] != 0.0:
            raise ValueError(f"Fresh-process MLP replay failed for {name}.")

        predictions = tuple(
            ValidationPrediction(
                row.vector.dataset_partition,
                row.vector.unit_id,
                row.vector.selected_cycle,
                predicted,
                row.true_rul,
            )
            for row, predicted in zip(
                tables["validation"].rows, result.validation_predictions, strict=True
            )
        )
        metrics = calculate_validation_metrics(predictions)
        selected_mse = result.validation_losses[result.best_epoch - 1]
        document = {
            "schema_version": "p11-s2-mlp-settings-v1",
            "model_version": MODEL_VERSION,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "scikit_learn_version": sklearn.__version__,
            "hidden_dims": list(hidden_dims),
            "random_seed": result.random_seed,
            "best_epoch": result.best_epoch,
            "epochs_trained": len(result.training_losses),
            "selected_validation_mse_cycles_squared": selected_mse,
            "mae_cycles": metrics.mae,
            "rmse_cycles": metrics.rmse,
            "runtime_seconds": result.runtime_seconds,
            "parameters": result.parameters,
            "source_sha256": TRAIN_SHA256,
            "split_manifest_sha256": expected_manifest_sha256,
            "fitting_table_sha256": table_hashes["fitting"],
            "validation_table_sha256": table_hashes["validation"],
            "weights_sha256": saved["payload"]["model"]["sha256"],
            "manifest_sha256": json.loads(saved["manifest_path"].read_text(encoding="utf-8"))["sha256"],
            "fresh_process_verification": verification,
            "predictions": [
                {
                    "dataset_partition": row.dataset_partition,
                    "unit_id": row.unit_id,
                    "cutoff_cycle": row.cutoff_cycle,
                    "true_rul_cycles": row.true_rul,
                    "prediction_cycles": row.prediction,
                }
                for row in predictions
            ],
        }
        (output_dir / "p11_s2_results.json").write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"hidden_dims={hidden_dims} seed={result.random_seed} "
            f"best_epoch={result.best_epoch}/{len(result.training_losses)} "
            f"validation_mse={selected_mse:.6f} MAE={metrics.mae:.6f} "
            f"RMSE={metrics.rmse:.6f} runtime={result.runtime_seconds:.3f}s "
            f"replay={verification['prediction_count']}/20 exact output={output_dir.relative_to(PROJECT_ROOT)}"
        )

    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("P04 split manifest changed during MLP experiments.")


if __name__ == "__main__":
    main()
