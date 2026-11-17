"""Fit a fresh MLP scaler using every past-only feature row from FD001/train."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_scaler import (
    build_final_training_features,
    fit_final_scaler,
    load_final_scaler_manifest,
    save_final_scaler_manifest,
)

SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
SELECTION_PATH = PROJECT_ROOT / "artifacts/selection_manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/final/mlp/scaler_manifest.json"


def main() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    selection_document = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selection = load_selection_manifest(SELECTION_PATH)
    if selection["training_source_sha256"] != source_sha256:
        raise ValueError("P11 selection decision and final training source differ.")
    mlp_rule = selection["decision"]["mlp_refit_rule"]
    if mlp_rule["frozen_epoch_count"] != 8 or mlp_rule["hidden_dims"] != [128, 64]:
        raise ValueError("P11 MLP alternative/refit rule is not frozen as expected.")
    selected_features = selection["decision"]["selected_configuration"]["feature_config"]
    config = FeatureConfig(
        channels=tuple(selected_features["channels"]),
        window_cycles=selected_features["window_cycles"],
    )
    histories = load_engine_history(SOURCE_PATH)
    features = build_final_training_features(histories, config=config)
    scaler = fit_final_scaler(features)
    digest = save_final_scaler_manifest(
        scaler,
        features,
        config=config,
        training_source_sha256=source_sha256,
        selection_manifest_sha256=selection_document["sha256"],
        path=OUTPUT_PATH,
    )
    reloaded, payload = load_final_scaler_manifest(
        OUTPUT_PATH,
        expected_source_sha256=source_sha256,
        expected_config=config,
    )
    if not (
        (reloaded.mean_ == scaler.mean_).all()
        and (reloaded.scale_ == scaler.scale_).all()
        and (reloaded.var_ == scaler.var_).all()
    ):
        raise ValueError("Scaler manifest reload changed fitted statistics.")
    transformed = reloaded.transform(features.matrix)
    if not np.isfinite(transformed).all():
        raise ValueError("Reloaded final scaler produced non-finite transformed features.")

    print(f"source_sha256: {source_sha256}")
    print(f"selection_manifest_sha256: {selection_document['sha256']}")
    print(f"training_engines: {payload['training_engine_count']}")
    print(f"feature_rows: {payload['training_source_row_count']}")
    print(f"feature_count: {reloaded.n_features_in_}")
    print(f"past_only_window_cycles: {config.window_cycles}")
    print(f"scaler_reload: exact mean/scale/variance; transformed_rows={len(transformed)}")
    print(f"manifest_sha256: {digest}")
    print(f"output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
