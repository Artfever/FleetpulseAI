"""Refit the frozen P11 MLP for eight epochs on all FD001/train cycle rows."""

from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import sklearn
import torch

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.data.targets import calculate_training_rul
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_mlp import train_final_mlp
from fleetpulse.models.final_scaler import (
    build_final_training_features,
    load_final_scaler_manifest,
)

SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
SELECTION_PATH = PROJECT_ROOT / "artifacts/selection_manifest.json"
SCALER_PATH = PROJECT_ROOT / "artifacts/final/mlp/scaler_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/final/mlp"
STATE_PATH = OUTPUT_DIR / "refit_state.pt"
MANIFEST_PATH = OUTPUT_DIR / "refit_manifest.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    selection_document = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selection = load_selection_manifest(SELECTION_PATH)
    if selection["training_source_sha256"] != source_sha256:
        raise ValueError("P11 selection decision and final training source differ.")
    decision = selection["decision"]
    rule = decision["mlp_refit_rule"]
    if rule["hidden_dims"] != [128, 64] or rule["frozen_epoch_count"] != 8:
        raise ValueError("P11 frozen MLP refit rule is not the expected 128/64, eight-epoch rule.")
    chosen_features = decision["selected_configuration"]["feature_config"]
    config = FeatureConfig(
        channels=tuple(chosen_features["channels"]),
        window_cycles=chosen_features["window_cycles"],
    )

    histories = load_engine_history(SOURCE_PATH)
    all_features = build_final_training_features(histories, config=config)
    labels = calculate_training_rul(histories)
    label_by_key = {
        (TRAIN_PARTITION, int(row["unit_id"]), int(row["cycle"])): label
        for row, label in zip(histories, labels, strict=True)
    }
    if set(label_by_key) != set(all_features.row_keys):
        raise ValueError("Complete-history targets do not align one-to-one with final features.")
    target_values = np.asarray([label_by_key[key] for key in all_features.row_keys], dtype=np.float64)

    scaler, scaler_payload = load_final_scaler_manifest(
        SCALER_PATH, expected_source_sha256=source_sha256, expected_config=config
    )
    if scaler_payload["selection_manifest_sha256"] != selection_document["sha256"]:
        raise ValueError("Final scaler was not built from the current P11 selection manifest.")
    if scaler_payload["training_source_row_count"] != len(all_features.row_keys):
        raise ValueError("Final scaler and final MLP row counts differ.")

    trained = train_final_mlp(
        all_features.matrix,
        target_values,
        config.output_schema,
        scaler,
        hidden_dims=tuple(rule["hidden_dims"]),
        epochs=rule["frozen_epoch_count"],
        batch_size=128,
        learning_rate=1e-3,
        random_seed=rule["seed"],
    )
    if len(trained.training_losses) != rule["frozen_epoch_count"]:
        raise RuntimeError("Final MLP did not complete the frozen epoch count.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in trained.model.state_dict().items()},
        STATE_PATH,
    )
    reloaded_state = torch.load(STATE_PATH, map_location="cpu", weights_only=True)
    expected_state = trained.model.state_dict()
    if set(reloaded_state) != set(expected_state) or any(
        not torch.equal(reloaded_state[name], expected_state[name]) for name in expected_state
    ):
        raise ValueError("Saved final MLP checkpoint did not reload exactly.")
    state_sha256 = _sha256_bytes(STATE_PATH.read_bytes())
    payload = {
        "schema_version": "p12-final-mlp-refit-v1",
        "artifact_role": "final-mlp-refit-checkpoint-for-p12-s3-packaging",
        "dataset_partition": "train",
        "training_source_sha256": source_sha256,
        "selection_manifest_sha256": selection_document["sha256"],
        "scaler_payload_sha256": _sha256_bytes(_canonical(scaler_payload)),
        "feature_config": {
            "version": config.version,
            "channels": list(config.channels),
            "window_cycles": config.window_cycles,
        },
        "feature_names": list(trained.feature_names),
        "training_engine_count": len(all_features.per_engine_row_counts),
        "training_row_count": len(all_features.row_keys),
        "per_engine_row_counts": [
            {"unit_id": unit_id, "row_count": row_count}
            for unit_id, row_count in all_features.per_engine_row_counts
        ],
        "architecture_and_training": trained.parameters,
        "random_seed": trained.random_seed,
        "epoch_training_mse_cycles_squared": list(trained.training_losses),
        "runtime_seconds": trained.runtime_seconds,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "scikit_learn_version": sklearn.__version__,
        "checkpoint_file": STATE_PATH.name,
        "checkpoint_sha256": state_sha256,
        "checkpoint_reload_exact": True,
        "validation_used": False,
        "official_test_data_opened": False,
        "official_test_labels_used": False,
    }
    digest = _sha256_bytes(_canonical(payload))
    MANIFEST_PATH.write_text(
        json.dumps({"payload": payload, "sha256": digest}, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"source_sha256: {source_sha256}")
    print(f"training_engines: {payload['training_engine_count']}")
    print(f"training_rows: {payload['training_row_count']}")
    print(f"features: {len(trained.feature_names)}")
    print(f"epochs: {len(trained.training_losses)}")
    print(f"training_mse_cycles_squared: {json.dumps(list(trained.training_losses))}")
    print(f"runtime_seconds: {trained.runtime_seconds:.3f}")
    print(f"checkpoint_sha256: {state_sha256}")
    print(f"manifest_sha256: {digest}")
    print(f"outputs: {STATE_PATH.relative_to(PROJECT_ROOT)}, {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
