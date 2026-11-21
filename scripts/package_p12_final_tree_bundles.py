"""Package immutable P12 final tree and constant-mean model bundles."""

from __future__ import annotations

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split import TRAIN_PARTITION
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_scaler import build_final_training_features
from fleetpulse.models.final_tree_bundle import (
    FINAL_TREE_INDEX_FILENAME,
    FINAL_TREE_INDEX_SCHEMA,
    FINAL_TREE_MODELS,
    load_final_tree_bundle,
    package_final_tree_bundles,
    predict_final_tree_bundle,
)

REFIT_MANIFEST = PROJECT_ROOT / "artifacts/final/tree/refit/refit_manifest.json"
SELECTION_MANIFEST = PROJECT_ROOT / "artifacts/selection_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/final/tree/bundles"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_existing_bundles() -> dict:
    index_path = OUTPUT_DIR / FINAL_TREE_INDEX_FILENAME
    index_document = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload = index_document.get("payload")
    if (
        not isinstance(index_payload, dict)
        or index_payload.get("schema_version") != FINAL_TREE_INDEX_SCHEMA
        or index_document.get("sha256") != _sha256(_canonical(index_payload))
        or set(index_payload.get("bundles", {})) != set(FINAL_TREE_MODELS)
    ):
        raise ValueError("Existing final tree bundle index is invalid or incomplete.")
    bundles = {}
    for name, record in index_payload["bundles"].items():
        manifest_path = OUTPUT_DIR / record["manifest_file"]
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_document.get("sha256") != record.get("manifest_sha256"):
            raise ValueError(f"Existing {name} manifest differs from the bundle index.")
        bundles[name] = load_final_tree_bundle(manifest_path)
    return bundles


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        bundles = _load_existing_bundles()
        print("bundle_action: existing package verified; no files overwritten")
    else:
        bundles = package_final_tree_bundles(
            REFIT_MANIFEST, SELECTION_MANIFEST, OUTPUT_DIR
        )
        print("bundle_action: new immutable package created")
    refit_document = json.loads(REFIT_MANIFEST.read_text(encoding="utf-8"))
    feature_config = refit_document["payload"]["feature_config"]
    config = FeatureConfig(
        channels=tuple(feature_config["channels"]),
        window_cycles=feature_config["window_cycles"],
    )
    features = build_final_training_features(
        load_engine_history(PROJECT_ROOT / "data/raw/train_FD001.txt"), config=config
    )
    reference_key = (TRAIN_PARTITION, 4, 164)
    reference_row = features.row_keys.index(reference_key)
    sample = features.matrix[[reference_row]]
    print(f"output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    for name, bundle in bundles.items():
        prediction = predict_final_tree_bundle(bundle, sample)[0]
        if name == "constant-rul":
            expected = refit_document["payload"]["baseline"]["prediction_cycles"]
        else:
            expected = refit_document["payload"]["models"][name][
                "training_row_sanity_prediction_cycles"
            ]
        if prediction != expected:
            raise ValueError(f"{name} packaged prediction differs from its P12-A1 refit result.")
        if bundle.baseline is not None:
            print(
                f"{name}: version={bundle.version} statistic={bundle.baseline.statistic} "
                f"value={prediction:.8f} cycles reload=verified refit_match=yes"
            )
        else:
            print(
                f"{name}: version={bundle.version} training_row_prediction={prediction:.8f} "
                "cycles reload=verified refit_match=yes"
            )
    index = json.loads((OUTPUT_DIR / "final_tree_bundle_index.json").read_text(encoding="utf-8"))
    print(f"bundle_count: {len(index['payload']['bundles'])}")
    print(f"index_sha256: {index['sha256']}")


if __name__ == "__main__":
    main()
