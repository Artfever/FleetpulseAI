"""Synthetic P17-A3 checks bind explanation output to raw predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from fleetpulse.features import engineer_features
from fleetpulse.services import (
    ExplanationConsistency,
    check_final_explanation_consistency,
)
from fleetpulse.services.explanations import FeatureContribution, FinalTreeExplanation
from fleetpulse.services.final_comparison import (
    FinalHistoryComparison,
    FinalModelPrediction,
)
from fleetpulse.services.predictor import RULPrediction


def _history(unit_id: int = 41):
    rows = []
    for cycle in range(1, 31):
        row = {
            "unit_id": unit_id,
            "cycle": cycle,
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        }
        row.update({f"sensor_{n}": cycle * n / 10 for n in range(1, 22)})
        rows.append(row)
    return rows


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _pair():
    history = _history()
    vector = engineer_features(history, dataset_partition="FD001/train", unit_id=41)
    manifest_sha, weights_sha = "a" * 64, "b" * 64
    config_sha = _sha(asdict(vector.config))
    source_sha, selection_sha = "c" * 64, "d" * 64
    history_sha = _sha(
        tuple({**row, "unit_id": int(row["unit_id"]), "cycle": int(row["cycle"])} for row in history)
    )
    prediction = FinalModelPrediction(
        model_name="decision-tree-rul",
        model_version="decision-tree-rul-v1",
        stage="final",
        raw_rul_cycles=12.5,
        manifest_payload_sha256=manifest_sha,
        weights_sha256=weights_sha,
        feature_config_sha256=config_sha,
        training_source_sha256=source_sha,
        selection_manifest_sha256=selection_sha,
        vector=vector,
    )
    comparison = FinalHistoryComparison(
        dataset_partition="FD001/train",
        unit_id=41,
        observed_cycle=30,
        input_history_sha256=history_sha,
        predictions=(prediction,),
        failures=(),
    )
    contributions = tuple(
        FeatureContribution(name, value, 0.0)
        for name, value in zip(vector.config.output_schema, vector.values, strict=True)
    )
    explanation = FinalTreeExplanation(
        model_name="decision-tree-rul",
        model_version="decision-tree-rul-v1",
        dataset_partition="FD001/train",
        unit_id=41,
        observed_cycle=30,
        raw_rul_cycles=12.5,
        base_value_cycles=12.5,
        contributions=contributions,
        global_summary=(),
        feature_config_sha256=config_sha,
        input_history_sha256=history_sha,
        manifest_payload_sha256=manifest_sha,
        weights_sha256=weights_sha,
        training_source_sha256=source_sha,
        background_sha256="e" * 64,
        background_selection_policy="fd001-train-all-engines-observed-cycle-90-v1",
        background_row_count=100,
        additivity_error_cycles=0.0,
    )
    return comparison, explanation


def test_exact_final_prediction_and_explanation_are_consistent():
    comparison, explanation = _pair()

    result = check_final_explanation_consistency(comparison, explanation)

    assert result == ExplanationConsistency(
        is_consistent=True,
        mismatch_codes=(),
        model_name="decision-tree-rul",
        model_version="decision-tree-rul-v1",
        dataset_partition="FD001/train",
        unit_id=41,
        observed_cycle=30,
        raw_rul_cycles=12.5,
        input_history_sha256=comparison.input_history_sha256,
    )


def test_version_artifact_input_and_schema_mismatches_are_reported():
    comparison, explanation = _pair()

    stale_version = check_final_explanation_consistency(
        comparison, replace(explanation, model_version="decision-tree-rul-v0")
    )
    assert not stale_version.is_consistent
    assert "model_version_mismatch" in stale_version.mismatch_codes

    unsupported = check_final_explanation_consistency(
        comparison, replace(explanation, model_name="random-forest-rul")
    )
    assert "unsupported_explanation_model" in unsupported.mismatch_codes

    changed_bundle = check_final_explanation_consistency(
        comparison, replace(explanation, manifest_payload_sha256="f" * 64)
    )
    assert "manifest_mismatch" in changed_bundle.mismatch_codes

    wrong_prefix = check_final_explanation_consistency(
        comparison, replace(explanation, input_history_sha256="9" * 64)
    )
    assert "input_history_mismatch" in wrong_prefix.mismatch_codes

    shuffled = replace(
        explanation,
        contributions=tuple(reversed(explanation.contributions)),
    )
    schema_result = check_final_explanation_consistency(comparison, shuffled)
    assert "feature_schema_mismatch" in schema_result.mismatch_codes


def test_raw_value_is_compared_and_display_only_objects_are_rejected():
    comparison, explanation = _pair()

    clipped = check_final_explanation_consistency(
        comparison, replace(explanation, raw_rul_cycles=0.0)
    )
    assert "raw_prediction_mismatch" in clipped.mismatch_codes
    with pytest.raises(TypeError, match="FinalTreeExplanation"):
        check_final_explanation_consistency(
            comparison,
            RULPrediction(12.5, 12.5, "decision-tree-rul", "v1", 30, 30, ()),
        )


def test_missing_tree_prediction_fails_closed():
    comparison, explanation = _pair()
    no_tree = replace(comparison, predictions=())

    result = check_final_explanation_consistency(no_tree, explanation)

    assert not result.is_consistent
    assert result.mismatch_codes == ("prediction_missing_or_ambiguous",)
