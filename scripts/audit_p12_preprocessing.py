"""Audit that P12 final tree models need no learned preprocessing refit."""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_scaler import load_final_scaler_manifest

TREE_DIR = PROJECT_ROOT / "artifacts/final/tree/refit"
TREE_MANIFEST_PATH = TREE_DIR / "refit_manifest.json"
MLP_SCALER_PATH = PROJECT_ROOT / "artifacts/final/mlp/scaler_manifest.json"
SELECTION_PATH = PROJECT_ROOT / "artifacts/selection_manifest.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tree_document = json.loads(TREE_MANIFEST_PATH.read_text(encoding="utf-8"))
    tree_payload = tree_document["payload"]
    if tree_document.get("sha256") != hashlib.sha256(_canonical(tree_payload)).hexdigest():
        raise ValueError("P12-A1 tree refit manifest hash mismatch.")
    selection_document = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selection = load_selection_manifest(SELECTION_PATH)
    source_sha256 = selection["training_source_sha256"]
    if (
        tree_payload["training_source_sha256"] != source_sha256
        or tree_payload["selection_manifest_sha256"] != selection_document["sha256"]
    ):
        raise ValueError("Tree refit source/selection provenance differs from P11.")
    selected_config = selection["decision"]["selected_configuration"]["feature_config"]
    config = FeatureConfig(
        channels=tuple(selected_config["channels"]),
        window_cycles=selected_config["window_cycles"],
    )
    expected_feature_config = {
        "version": config.version,
        "channels": list(config.channels),
        "window_cycles": config.window_cycles,
    }
    if tree_payload["feature_config"] != expected_feature_config:
        raise ValueError("Tree refit features differ from the frozen P11 input configuration.")
    if (tree_payload["training_engine_count"], tree_payload["training_row_count"]) != (100, 20631):
        raise ValueError("Tree refit does not record all 100 original training engines/cycle rows.")
    if "scaler" in tree_payload or "preprocessor" in tree_payload:
        raise ValueError("Tree refit unexpectedly records learned preprocessing state.")

    for name, candidate_name in (
        ("decision-tree-rul", "Decision Tree, 30-cycle window"),
        ("random-forest-rul", "Random Forest, depth 16"),
        ("xgboost-rul", "XGBoost, depth 4"),
    ):
        artifact = tree_payload["models"][name]
        artifact_path = TREE_DIR / artifact["filename"]
        if _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Final {name} refit artifact hash mismatch.")
        frozen = next(
            item for item in selection_document["payload"]["candidates"]
            if item["name"] == candidate_name
        )
        if artifact["parameters"] != frozen["configuration"]:
            raise ValueError(f"Final {name} parameters differ from the frozen P11 candidate.")

    baseline = tree_payload["baseline"]
    baseline_path = TREE_DIR / baseline["filename"]
    if _sha256(baseline_path) != baseline["sha256"] or baseline["artifact"]["statistic"] != "mean":
        raise ValueError("Final constant baseline is missing, modified, or not the selected mean.")
    if baseline["artifact"]["fitting_label_count"] != tree_payload["training_row_count"]:
        raise ValueError("Final constant baseline did not use all original training labels.")

    scaler, scaler_payload = load_final_scaler_manifest(
        MLP_SCALER_PATH, expected_source_sha256=source_sha256, expected_config=config
    )
    if (
        scaler_payload["selection_manifest_sha256"] != selection_document["sha256"]
        or scaler_payload["training_engine_count"] != 100
        or scaler_payload["training_source_row_count"] != 20631
        or scaler.n_samples_seen_ != tree_payload["training_row_count"]
    ):
        raise ValueError("The separately required MLP scaler is not fit on all original training rows.")

    print("tree_preprocessing: none required; frozen features are deterministic past-only statistics")
    print("tree_models: 3 frozen candidates verified; all model file hashes and configs match P11")
    print("constant_baseline: selected mean; fitted on 20,631 original training labels")
    print("mlp_preprocessing: StandardScaler verified on 20,631 original training rows")
    print(f"training_source_sha256: {source_sha256}")
    print(f"p11_selection_manifest_sha256: {selection_document['sha256']}")
    print(f"tree_refit_manifest_sha256: {tree_document['sha256']}")
    print("official_test_data_opened: false")


if __name__ == "__main__":
    main()
