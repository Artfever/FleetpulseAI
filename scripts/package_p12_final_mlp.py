"""Package the P12 final MLP weights and scaler as a standalone bundle."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_mlp_bundle import (
    FINAL_MLP_MANIFEST_FILENAME,
    load_final_mlp_bundle,
    package_final_mlp_bundle,
    predict_final_mlp,
)
from fleetpulse.models.final_scaler import build_final_training_features

FINAL_DIR = PROJECT_ROOT / "artifacts/final/mlp"
SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
SELECTION_PATH = PROJECT_ROOT / "artifacts/selection_manifest.json"
SCALER_PATH = FINAL_DIR / "scaler_manifest.json"
REFIT_STATE_PATH = FINAL_DIR / "refit_state.pt"
REFIT_MANIFEST_PATH = FINAL_DIR / "refit_manifest.json"
BUNDLE_DIR = FINAL_DIR / "bundle"


def main() -> None:
    selection = load_selection_manifest(SELECTION_PATH)
    feature_config_doc = selection["decision"]["selected_configuration"]["feature_config"]
    config = FeatureConfig(
        channels=tuple(feature_config_doc["channels"]),
        window_cycles=feature_config_doc["window_cycles"],
    )
    histories = load_engine_history(SOURCE_PATH)
    features = build_final_training_features(histories, config=config)

    packaged = package_final_mlp_bundle(
        REFIT_STATE_PATH,
        REFIT_MANIFEST_PATH,
        SCALER_PATH,
        SELECTION_PATH,
        BUNDLE_DIR,
    )
    loaded = load_final_mlp_bundle(BUNDLE_DIR / FINAL_MLP_MANIFEST_FILENAME)
    if packaged.payload != loaded.payload:
        raise ValueError("Final MLP bundle changed between package and reload.")
    sample_key = (TRAIN_PARTITION, 4, 164)
    try:
        row_index = features.row_keys.index(sample_key)
    except ValueError as error:
        raise ValueError(f"Reference training sample {sample_key} is missing.") from error
    raw_prediction = predict_final_mlp(
        loaded, np.asarray([features.matrix[row_index]], dtype=np.float64)
    )[0]
    if not np.isfinite(raw_prediction):
        raise ValueError("Reloaded final MLP produced a non-finite sample prediction.")

    manifest_path = BUNDLE_DIR / FINAL_MLP_MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_file = BUNDLE_DIR / document["payload"]["model"]["weights_file"]
    print(f"artifact_version: {document['payload']['model']['version']}")
    print(f"artifact_role: {document['payload']['artifact_role']}")
    print(f"architecture: {document['payload']['model']['architecture']}")
    print(f"random_seed: {document['payload']['training']['random_seed']}")
    print(f"epochs: {document['payload']['training']['epochs_trained']}")
    print(f"training_rows: {document['payload']['training']['training_row_count']}")
    print(f"feature_count: {len(document['payload']['training']['feature_names'])}")
    print(f"weights_sha256: {hashlib.sha256(weights_file.read_bytes()).hexdigest()}")
    print(f"manifest_sha256: {document['sha256']}")
    print("reload: passed (CPU model and scaler loaded from bundle)")
    print(
        "training_sample_prediction: "
        f"partition=FD001/train engine={sample_key[1]} cycle={sample_key[2]} "
        f"prediction={raw_prediction:.8f} cycles (training-row sanity check, not evaluation)"
    )
    print(f"output: {BUNDLE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
