"""The P11 selection view accepts only intact, cohort-qualified summaries."""

import hashlib
import json

import pytest

from fleetpulse.evaluation.selection_manifest import load_selection_manifest


def _write_manifest(path, *, sample_count=20):
    candidates = [
        {
            "name": f"Candidate {index}",
            "sample_count": sample_count,
            "training_source_sha256": "a" * 64,
            "split_manifest_sha256": "b" * 64,
            "mae_cycles": float(index),
            "rmse_cycles": float(index + 1),
            "result_artifact_sha256": str(index) * 64,
        }
        for index in range(6)
    ]
    payload = {
        "schema_version": "p11-selection-manifest-v1",
        "dataset_partition": "FD001/train",
        "validation_sample_unit": "engine-prefix",
        "validation_engine_count": 20,
        "training_source_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "official_test_data_opened": False,
        "candidates": candidates,
        "decision": {
            "status": "owner_selected_candidate",
            "selected_candidate": "Candidate 0",
            "selection_authority": "Owner delegation",
            "lowest_mae_candidate": "Candidate 0",
            "lowest_rmse_candidate": "Candidate 0",
            "reason": "A documented metric choice.",
            "tradeoff": "There is no tie-break guarantee.",
            "selected_artifact": {
                "name": "candidate",
                "version": "candidate-v1",
                "sha256": "c" * 64,
                "matched_p11_run_checkpoint_sha256": "c" * 64,
            },
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.write_text(
        json.dumps(
            {"payload": payload, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
        ),
        encoding="utf-8",
    )


def test_load_selection_manifest_accepts_intact_six_candidate_comparison(tmp_path):
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path)

    manifest = load_selection_manifest(path)

    assert manifest["validation_engine_count"] == 20
    assert len(manifest["candidates"]) == 6


def test_load_selection_manifest_rejects_changed_content_hash(tmp_path):
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["candidates"][0]["mae_cycles"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_selection_manifest(path)


def test_load_selection_manifest_rejects_wrong_validation_sample_count(tmp_path):
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path, sample_count=19)

    with pytest.raises(ValueError, match="sample count mismatch"):
        load_selection_manifest(path)


def test_load_selection_manifest_rejects_unmatched_selected_checkpoint(tmp_path):
    path = tmp_path / "selection_manifest.json"
    _write_manifest(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["decision"]["selected_artifact"]["sha256"] = "e" * 64
    payload = document["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its P11 comparison run"):
        load_selection_manifest(path)
