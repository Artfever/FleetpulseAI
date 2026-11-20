"""Synthetic P17-S1 tests; no trained FleetPulse artifact is required."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.features import FeatureConfig
from fleetpulse.models.final_tree_bundle import FinalTreeBundle
from fleetpulse.services import explanations


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _history(
    unit_id: int, offset: float = 0.0, cycles: int = 30
) -> list[dict[str, float | int]]:
    rows = []
    for cycle in range(1, cycles + 1):
        row: dict[str, float | int] = {
            "unit_id": unit_id,
            "cycle": cycle,
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        }
        row.update(
            {
                f"sensor_{number}": offset + cycle * (number / 20)
                for number in range(1, 22)
            }
        )
        rows.append(row)
    return rows


def _training_source(path) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for unit_id in range(1, 101):
            for row in _history(unit_id, offset=unit_id / 10, cycles=90):
                values = [row["unit_id"], row["cycle"]]
                values.extend(row[channel] for channel in READING_COLUMNS)
                output.write(" ".join(str(value) for value in values) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_bundle(background, config):
    matrix = np.asarray(background.values, dtype=np.float64)
    target = matrix[:, 0] * 0.25 + matrix[:, 4] * 0.75
    model = DecisionTreeRegressor(max_depth=4, random_state=42).fit(matrix, target)
    payload = {
        "artifact_role": "final-training-model-bundle",
        "name": "decision-tree-rul",
        "model": {"weights_sha256": "a" * 64},
        "training": {"training_source_sha256": background.training_source_sha256},
    }
    return FinalTreeBundle(
        name="decision-tree-rul",
        version="decision-tree-rul-v1",
        model=model,
        baseline=None,
        feature_config=config,
        feature_names=config.output_schema,
        payload=payload,
    )


def _build_inputs(tmp_path, monkeypatch):
    training_path = tmp_path / "train_FD001.txt"
    source_hash = _training_source(training_path)
    config = FeatureConfig()
    background = explanations.build_fd001_training_background(
        training_path,
        expected_training_source_sha256=source_hash,
        feature_config=config,
    )
    bundle = _synthetic_bundle(background, config)
    monkeypatch.setattr(explanations, "load_final_tree_bundle", lambda _: bundle)
    return source_hash, background, bundle


def test_fixed_training_background_and_shap_explanation_are_reproducible(
    tmp_path, monkeypatch
):
    source_hash, background, bundle = _build_inputs(tmp_path, monkeypatch)

    result = explanations.explain_final_tree_prediction(
        _history(501, offset=3.0),
        dataset_partition="FD001/train",
        unit_id=501,
        background=background,
    )

    assert len(background.values) == 100
    assert background.training_source_sha256 == source_hash
    assert background.selection_policy == explanations.BACKGROUND_SELECTION_POLICY
    assert result.model_name == bundle.name
    assert result.model_version == bundle.version
    assert result.raw_rul_cycles == pytest.approx(bundle.model.predict([_history_vector(501)] )[0])
    assert result.base_value_cycles + sum(
        item.contribution_cycles for item in result.contributions
    ) == pytest.approx(result.raw_rul_cycles, abs=1e-6)
    assert len(result.contributions) == 96
    assert {item.feature_name for item in result.contributions} == set(bundle.feature_names)
    assert len(result.top_contributions(5)) == 5
    assert result.background_row_count == 100
    assert len(result.global_summary) == 10
    assert all(item.mean_absolute_contribution_cycles >= 0 for item in result.global_summary)


def _history_vector(unit_id: int) -> list[float]:
    from fleetpulse.features import engineer_features

    return list(
        engineer_features(
            _history(unit_id, offset=3.0),
            dataset_partition="FD001/train",
            unit_id=unit_id,
        ).values
    )


def test_background_source_hash_and_feature_order_are_required(tmp_path, monkeypatch):
    _, background, _ = _build_inputs(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="source, selection policy or schema"):
        explanations._validate_background(
            background.__class__(
                **{
                    **background.__dict__,
                    "training_source_sha256": "b" * 64,
                }
            ),
            explanations.load_final_tree_bundle("unused"),
            "a" * 64,
        )


def test_feature_background_round_trip_and_integrity(tmp_path, monkeypatch):
    source_hash, background, _ = _build_inputs(tmp_path, monkeypatch)
    path = tmp_path / "decision-tree-background.json"

    artifact_sha = explanations.save_training_feature_background(background, path)
    loaded = explanations.load_training_feature_background(
        path,
        expected_training_source_sha256=source_hash,
        feature_config=FeatureConfig(),
    )

    assert artifact_sha == _sha(json.loads(path.read_text(encoding="utf-8"))["payload"])
    assert loaded.values == background.values
    assert loaded.feature_names == background.feature_names
    assert loaded.background_sha256 == background.background_sha256
    assert loaded.source_path == path.as_posix()

    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["values"][0][0] += 1.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        explanations.load_training_feature_background(
            path,
            expected_training_source_sha256=source_hash,
            feature_config=FeatureConfig(),
        )


def test_explanation_uses_packaged_background_without_raw_history_file(
    tmp_path, monkeypatch
):
    source_hash, background, bundle = _build_inputs(tmp_path, monkeypatch)
    artifact_root = tmp_path / "workspace"
    background_path = artifact_root / explanations.BACKGROUND_ARTIFACT_PATH
    explanations.save_training_feature_background(background, background_path)

    result = explanations.explain_final_tree_history(
        _history(501, offset=3.0),
        dataset_partition="FD001/train",
        unit_id=501,
        artifact_root=artifact_root,
        training_path=tmp_path / "missing" / "train_FD001.txt",
    )

    assert result.model_version == bundle.version
    assert result.training_source_sha256 == source_hash
    assert result.background_sha256 == background.background_sha256
    assert result.background_row_count == 100


def test_invalid_observed_history_is_rejected_before_explaining(tmp_path, monkeypatch):
    _, background, _ = _build_inputs(tmp_path, monkeypatch)
    invalid = _history(501)
    invalid[0]["sensor_1"] = float("nan")

    with pytest.raises(explanations.PredictionInputError, match="Invalid observed engine history"):
        explanations.explain_final_tree_prediction(
            invalid,
            dataset_partition="FD001/train",
            unit_id=501,
            background=background,
        )


def test_background_builder_rejects_wrong_source_digest(tmp_path):
    training_path = tmp_path / "train_FD001.txt"
    _training_source(training_path)

    with pytest.raises(ValueError, match="hash differs"):
        explanations.build_fd001_training_background(
            training_path,
            expected_training_source_sha256="0" * 64,
            feature_config=FeatureConfig(),
        )
