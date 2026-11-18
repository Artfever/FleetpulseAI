"""Verify that bundles attributed to P13 results match the frozen P12 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TREE_DIR = Path("artifacts/final/tree/bundles")
MLP_DIR = Path("artifacts/final/mlp/bundle")
TREE_INDEX_PATH = TREE_DIR / "final_tree_bundle_index.json"
RESULTS_PATH = Path("artifacts/final/p13/final_test_results.json")
OUTPUT_PATH = Path("artifacts/final/p13/a3_frozen_bundle_audit.json")
TREE_INDEX_PATH = TREE_DIR / "final_tree_bundle_index.json"
EXPECTED_TREE_INDEX_SHA256 = "33bc4051835a5125f46ae31fc9f8a812325eb900559671de65dbe2e0273f8c89"
EXPECTED_FEATURE_CONFIG_SHA256 = "19dd0331ee3e0c3a90217446a4481507b7da94fe1424b1c251c4605a6b168dfe"
EXPECTED_RESULTS_SHA256 = "2a0686eaa3d369c435938cf6bf6256b047b3265f8ac8893e08e040adc798a6a6"
AUDIT_SCHEMA = "p13-a3-frozen-bundle-audit-v1"

EXPECTED_INVENTORY: dict[str, dict[str, Any]] = {
    "decision-tree-rul": {
        "version": "decision-tree-rul-v1",
        "manifest_path": "artifacts/final/tree/bundles/decision-tree-rul_manifest.json",
        "manifest_file": "decision-tree-rul_manifest.json",
        "manifest_payload_sha256": "ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89",
        "weights_file": "decision_tree_final_v1.joblib",
        "weights_sha256": "c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3",
    },
    "random-forest-rul": {
        "version": "random-forest-rul-v1",
        "manifest_path": "artifacts/final/tree/bundles/random-forest-rul_manifest.json",
        "manifest_file": "random-forest-rul_manifest.json",
        "manifest_payload_sha256": "f52ff4dbcb9c3876ff045bebfaef791652d996a151e987815cf37a6afed69b1a",
        "weights_file": "random_forest_final_v1.joblib",
        "weights_sha256": "712a8313271565091fb3a7c9b4fce5e5d8bf2caf64b77e8da9c942a477605d81",
    },
    "xgboost-rul": {
        "version": "xgboost-rul-v1",
        "manifest_path": "artifacts/final/tree/bundles/xgboost-rul_manifest.json",
        "manifest_file": "xgboost-rul_manifest.json",
        "manifest_payload_sha256": "e40f4a9142caf5dc6d8c4880a2b0e526ef0e9e097e472ab70d44aa79169c2e11",
        "weights_file": "xgboost_final_v1.joblib",
        "weights_sha256": "719a5553ba938fb40dd4196d081d7d07f6c7490be0dbbd4d0cf653e776a4816e",
    },
    "constant-rul": {
        "version": "constant-rul-v1",
        "manifest_path": "artifacts/final/tree/bundles/constant-rul_manifest.json",
        "manifest_file": "constant-rul_manifest.json",
        "manifest_payload_sha256": "dff249bdfef9426b731a4d8fb8f74bd298571d38e86eb277f0a2a8b27d724c96",
        "weights_file": None,
        "weights_sha256": None,
    },
    "dense-rul-mlp-final": {
        "version": "dense-rul-mlp-final-v1",
        "manifest_path": "artifacts/final/mlp/bundle/final_mlp_manifest.json",
        "manifest_file": "final_mlp_manifest.json",
        "manifest_payload_sha256": "7a3e56ff80df0ba194067956842dab1e9d4148f85f5dc1ad9378bca0b234549c",
        "weights_file": "final_mlp_state.pt",
        "weights_sha256": "fcbd5a67ca8bf307f049c7131fee08ee484f8f35e6cb4e8966aa3c45839b4650",
    },
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_document(path: Path, label: str) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    payload = document.get("payload")
    digest = document.get("sha256")
    if not isinstance(payload, dict) or digest != _canonical_sha256(payload):
        raise ValueError(f"{label} payload checksum is invalid: {path}.")
    return payload, digest, hashlib.sha256(raw).hexdigest()


def audit_bundles(
    tree_index_path: Path,
    tree_directory: Path,
    mlp_directory: Path,
    results_path: Path,
    *,
    expected_inventory: Mapping[str, Mapping[str, Any]] = EXPECTED_INVENTORY,
    expected_tree_index_sha256: str = EXPECTED_TREE_INDEX_SHA256,
    expected_results_sha256: str = EXPECTED_RESULTS_SHA256,
    expected_feature_config_sha256: str = EXPECTED_FEATURE_CONFIG_SHA256,
) -> dict[str, Any]:
    """Compare actual local bundle bytes against P12 inventory and P13 score report."""
    index, index_sha, _ = _read_document(tree_index_path, "Tree bundle index")
    if index_sha != expected_tree_index_sha256:
        raise ValueError("Frozen P12 tree bundle index differs from the tracked inventory.")
    report, report_sha, _ = _read_document(results_path, "P13 result report")
    if report_sha != expected_results_sha256 or report.get("dataset_partition") != "FD001/test":
        raise ValueError("P13 result report checksum or partition differs from the frozen report.")
    result_models = report.get("models")
    if not isinstance(result_models, list):
        raise TypeError("P13 result report is missing model identities.")
    evaluated: dict[str, Mapping[str, Any]] = {}
    for model in result_models:
        if not isinstance(model, Mapping) or not isinstance(model.get("name"), str):
            raise TypeError("P13 result report has a malformed model entry.")
        if model["name"] in evaluated:
            raise ValueError(f"Duplicate evaluated model in P13 report: {model['name']}.")
        evaluated[model["name"]] = model
    if set(evaluated) != set(expected_inventory):
        raise ValueError("P12 inventory and P13 evaluated model sets differ.")

    tree_entries = index.get("bundles")
    if not isinstance(tree_entries, Mapping):
        raise TypeError("P12 tree index does not contain a bundle inventory.")
    expected_tree_models = set(expected_inventory) - {"dense-rul-mlp-final"}
    if set(tree_entries) != expected_tree_models:
        raise ValueError("P12 tree index model set differs from the frozen tree inventory.")
    audited: list[dict[str, Any]] = []
    checked_files = 1  # the tree index itself
    for name, expected in expected_inventory.items():
        manifest_path = Path(expected["manifest_path"])
        bundle_directory = mlp_directory if name == "dense-rul-mlp-final" else tree_directory
        payload, payload_sha, manifest_file_sha = _read_document(
            bundle_directory / expected["manifest_file"], f"{name} manifest"
        )
        checked_files += 1
        if payload_sha != expected["manifest_payload_sha256"]:
            raise ValueError(f"P12 manifest payload differs from frozen inventory: {name}.")
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping) or model_payload.get("version") != expected["version"]:
            raise ValueError(f"P12 bundle version differs from frozen inventory: {name}.")
        if name != "constant-rul" and name != "dense-rul-mlp-final":
            index_entry = tree_entries.get(name)
            if (
                not isinstance(index_entry, Mapping)
                or index_entry.get("manifest_file") != expected["manifest_file"]
                or index_entry.get("manifest_sha256") != payload_sha
            ):
                raise ValueError(f"P12 tree index entry differs from its bundle manifest: {name}.")
        elif name == "constant-rul":
            index_entry = tree_entries.get(name)
            if (
                not isinstance(index_entry, Mapping)
                or index_entry.get("manifest_file") != expected["manifest_file"]
                or index_entry.get("manifest_sha256") != payload_sha
            ):
                raise ValueError("P12 tree index entry differs from the mean baseline manifest.")

        weights_file = expected["weights_file"]
        weights_sha = None
        if weights_file is not None:
            weights_path = bundle_directory / weights_file
            weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            checked_files += 1
        if (
            model_payload.get("weights_file") != weights_file
            or model_payload.get("weights_sha256") != expected["weights_sha256"]
            or weights_sha != expected["weights_sha256"]
        ):
            raise ValueError(f"P12 serialized model bytes differ from frozen inventory: {name}.")

        evaluated_model = evaluated[name]
        evaluated_artifact = evaluated_model.get("artifact")
        if not isinstance(evaluated_artifact, Mapping):
            raise TypeError(f"P13 report does not record loaded bundle metadata: {name}.")
        comparisons = {
            "version": expected["version"],
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "manifest_file_sha256": manifest_file_sha,
            "manifest_payload_sha256": payload_sha,
            "weights_file": weights_file,
            "weights_sha256": weights_sha,
            "feature_config_sha256": expected_feature_config_sha256,
        }
        for field, value in comparisons.items():
            if evaluated_model.get("version") != expected["version"] and field == "version":
                raise ValueError(f"Evaluated model version differs from frozen bundle: {name}.")
            if evaluated_artifact.get(field) != value:
                raise ValueError(f"P13 evaluated bundle metadata differs from frozen P12 file at {name}.{field}.")
        if evaluated_artifact.get("artifact_role") != "final-training-model-bundle":
            raise ValueError(f"P13 report does not identify {name} as a final-training bundle.")
        audited.append(
            {
                "name": name,
                "version": expected["version"],
                "manifest_path": str(manifest_path).replace("\\", "/"),
                "manifest_file_sha256": manifest_file_sha,
                "manifest_payload_sha256": payload_sha,
                "weights_file": weights_file,
                "weights_sha256": weights_sha,
                "reported_evaluation_matches": True,
                "byte_identity_matches_frozen_inventory": True,
            }
        )

    return {
        "schema_version": AUDIT_SCHEMA,
        "phase": "P13",
        "dataset_partition": "FD001/test",
        "tree_bundle_index_payload_sha256": index_sha,
        "reported_results_payload_sha256": report_sha,
        "model_count": len(audited),
        "bundle_files_checked": checked_files,
        "evaluated_models_match_frozen_inventory": True,
        "models": audited,
        "test_artifact_used_for_training_or_tuning": False,
    }


def run_audit(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    payload = audit_bundles(TREE_INDEX_PATH, TREE_DIR, MLP_DIR, RESULTS_PATH)
    document = {"payload": payload, "sha256": _canonical_sha256(payload)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **payload}, indent=2, sort_keys=True))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    run_audit(args.output)


if __name__ == "__main__":
    main()
