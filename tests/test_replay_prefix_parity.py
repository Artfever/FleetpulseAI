"""Contract tests for independent saved-prefix replay checks."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import check_replay_prefix_parity as replay


def _setup(tmp_path, monkeypatch, *, alter_prediction=False):
    source = tmp_path / "data/raw/train_FD001.txt"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic training history\n", encoding="utf-8")
    split_path = tmp_path / "artifacts/split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split = {
        "validation_prefixes": [
            {"dataset_partition": "FD001/train", "unit_id": 1, "cutoff_cycle": 2, "prefix_row_count": 2},
            {"dataset_partition": "FD001/train", "unit_id": 2, "cutoff_cycle": 1, "prefix_row_count": 1},
        ]
    }
    split_path.write_text(json.dumps(split), encoding="utf-8")
    manifest_path = tmp_path / "artifacts/development/decision_tree_validation.json"
    manifest_path.parent.mkdir(parents=True)
    model_path = manifest_path.parent / "decision_tree_model.joblib"
    model_path.write_bytes(b"synthetic saved estimator")
    saved_predictions = [
        {**row, "prediction_cycles": float(row["unit_id"] * 10)}
        for row in split["validation_prefixes"]
    ]
    bundle = {
        "payload": {
            "model": {"name": "test-tree", "version": "test-v1"},
            "evaluation": {"predictions": saved_predictions},
        }
    }
    manifest_path.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(replay, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(replay, "TRAIN_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(replay, "EXPECTED_VALIDATION_COUNT", 2)
    monkeypatch.setattr(
        replay,
        "load_engine_history",
        lambda _: [
            {"unit_id": 1, "cycle": 1},
            {"unit_id": 1, "cycle": 2},
            {"unit_id": 2, "cycle": 1},
        ],
    )

    def predict(prefix, *, dataset_partition, unit_id, manifest_path):
        assert dataset_partition == "FD001/train"
        assert prefix[-1]["cycle"] == (2 if unit_id == 1 else 1)
        value = float(unit_id * 10) + (1 if alter_prediction and unit_id == 2 else 0)
        return SimpleNamespace(raw_rul_cycles=value)

    monkeypatch.setattr(replay, "predict_rul", predict)


def test_every_saved_replay_point_matches_its_standalone_prefix(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)

    result = replay.run_replay_check()

    assert result["compared_replay_points"] == 2
    assert result["exact_matches"] == 2
    assert result["max_abs_difference_cycles"] == 0.0
    assert [(row["unit_id"], row["cutoff_cycle"]) for row in result["examples"]] == [
        (1, 2),
        (2, 1),
    ]


def test_saved_replay_mismatch_fails_instead_of_reporting_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, alter_prediction=True)

    with pytest.raises(AssertionError, match="engine 2 cycle 1"):
        replay.run_replay_check()
