from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_p13_bundles import audit_bundles


def _seal(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"payload": payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, dict[str, object]], dict[str, object]]:
    tree_dir = tmp_path / "tree"
    mlp_dir = tmp_path / "mlp"
    expected: dict[str, dict[str, object]] = {}
    tree_entries: dict[str, dict[str, str]] = {}
    results: list[dict[str, object]] = []
    models = {
        "decision-tree-rul": ("decision-tree-rul-v1", "decision.json", "decision.bin", b"tree-a"),
        "random-forest-rul": ("random-forest-rul-v1", "forest.json", "forest.bin", b"tree-b"),
        "xgboost-rul": ("xgboost-rul-v1", "xgb.json", "xgb.bin", b"tree-c"),
        "constant-rul": ("constant-rul-v1", "mean.json", None, None),
        "dense-rul-mlp-final": ("dense-rul-mlp-final-v1", "mlp.json", "mlp.bin", b"mlp-weights"),
    }
    for name, (version, manifest_file, weights_file, weights_bytes) in models.items():
        is_mlp = name == "dense-rul-mlp-final"
        directory = mlp_dir if is_mlp else tree_dir
        weight_sha = None
        if weights_file is not None:
            (directory / weights_file).parent.mkdir(parents=True, exist_ok=True)
            (directory / weights_file).write_bytes(weights_bytes)
            weight_sha = hashlib.sha256(weights_bytes).hexdigest()
        manifest_payload = {
            "name": name,
            "model": {"version": version, "weights_file": weights_file, "weights_sha256": weight_sha},
            "training": {},
        }
        manifest = _seal(manifest_payload)
        _write(directory / manifest_file, manifest)
        manifest_bytes_sha = hashlib.sha256((directory / manifest_file).read_bytes()).hexdigest()
        manifest_path = (
            f"artifacts/final/mlp/bundle/{manifest_file}"
            if is_mlp
            else f"artifacts/final/tree/bundles/{manifest_file}"
        )
        expected[name] = {
            "version": version,
            "manifest_path": manifest_path,
            "manifest_file": manifest_file,
            "manifest_payload_sha256": manifest["sha256"],
            "weights_file": weights_file,
            "weights_sha256": weight_sha,
        }
        if not is_mlp:
            tree_entries[name] = {"manifest_file": manifest_file, "manifest_sha256": manifest["sha256"]}
        results.append(
            {
                "name": name,
                "version": version,
                "artifact": {
                    "artifact_role": "final-training-model-bundle",
                    "version": version,
                    "manifest_path": manifest_path,
                    "manifest_file_sha256": manifest_bytes_sha,
                    "manifest_payload_sha256": manifest["sha256"],
                    "weights_file": weights_file,
                    "weights_sha256": weight_sha,
                    "feature_config_sha256": "feature-sha",
                },
            }
        )

    index_path = tree_dir / "final_tree_bundle_index.json"
    index_document = _seal({"bundles": tree_entries})
    _write(index_path, index_document)
    results_path = tmp_path / "results.json"
    results_document = _seal({"dataset_partition": "FD001/test", "models": results})
    _write(results_path, results_document)
    return index_path, tree_dir, mlp_dir, results_path, expected, index_document


def test_audits_all_frozen_bundles_and_reported_bundle_hashes(tmp_path: Path) -> None:
    index_path, tree_dir, mlp_dir, results_path, expected, index = _fixture(tmp_path)

    report = audit_bundles(
        index_path,
        tree_dir,
        mlp_dir,
        results_path,
        expected_inventory=expected,
        expected_tree_index_sha256=index["sha256"],
        expected_results_sha256=json.loads(results_path.read_text())["sha256"],
        expected_feature_config_sha256="feature-sha",
    )

    assert report["model_count"] == 5
    assert report["bundle_files_checked"] == 10
    assert report["evaluated_models_match_frozen_inventory"] is True


def test_rejects_changed_serialized_weight_bytes(tmp_path: Path) -> None:
    index_path, tree_dir, mlp_dir, results_path, expected, index = _fixture(tmp_path)
    (tree_dir / "decision.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="serialized model bytes"):
        audit_bundles(
            index_path,
            tree_dir,
            mlp_dir,
            results_path,
            expected_inventory=expected,
            expected_tree_index_sha256=index["sha256"],
            expected_results_sha256=json.loads(results_path.read_text())["sha256"],
            expected_feature_config_sha256="feature-sha",
        )


def test_rejects_reported_bundle_metadata_that_does_not_match(tmp_path: Path) -> None:
    index_path, tree_dir, mlp_dir, results_path, expected, index = _fixture(tmp_path)
    reported = json.loads(results_path.read_text())
    reported["payload"]["models"][0]["artifact"]["weights_sha256"] = "0" * 64
    _write(results_path, _seal(reported["payload"]))

    with pytest.raises(ValueError, match="metadata differs"):
        audit_bundles(
            index_path,
            tree_dir,
            mlp_dir,
            results_path,
            expected_inventory=expected,
            expected_tree_index_sha256=index["sha256"],
            expected_results_sha256=json.loads(results_path.read_text())["sha256"],
            expected_feature_config_sha256="feature-sha",
        )


def test_manifest_identity_is_stable_across_windows_crlf_checkout(tmp_path: Path) -> None:
    index_path, tree_dir, mlp_dir, results_path, expected, index = _fixture(tmp_path)
    manifest_path = tree_dir / "decision.json"
    lf_bytes = manifest_path.read_bytes().replace(b"\r\n", b"\n")
    manifest_path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))

    results = json.loads(results_path.read_text())
    results["payload"]["models"][0]["artifact"]["manifest_file_sha256"] = hashlib.sha256(
        lf_bytes
    ).hexdigest()
    _write(results_path, _seal(results["payload"]))

    report = audit_bundles(
        index_path,
        tree_dir,
        mlp_dir,
        results_path,
        expected_inventory=expected,
        expected_tree_index_sha256=index["sha256"],
        expected_results_sha256=json.loads(results_path.read_text())["sha256"],
        expected_feature_config_sha256="feature-sha",
    )

    assert report["evaluated_models_match_frozen_inventory"] is True
