"""Development candidates and P12 final refits remain separately registered."""

import pytest

from fleetpulse.models.registry import (
    DEVELOPMENT_MODEL_REGISTRY,
    FINAL_MODEL_REGISTRY,
    MODEL_REGISTRIES,
)


def test_registry_separates_development_candidates_from_final_refits():
    assert set(DEVELOPMENT_MODEL_REGISTRY) == {
        "decision-tree-rul",
        "random-forest-rul",
        "xgboost-rul",
        "dense-rul-mlp",
    }
    assert set(FINAL_MODEL_REGISTRY) == {
        "decision-tree-rul",
        "random-forest-rul",
        "xgboost-rul",
        "constant-rul",
        "dense-rul-mlp-final",
    }
    assert MODEL_REGISTRIES["development"] is DEVELOPMENT_MODEL_REGISTRY
    assert MODEL_REGISTRIES["final"] is FINAL_MODEL_REGISTRY
    assert all(spec.stage == "development" for spec in DEVELOPMENT_MODEL_REGISTRY.values())
    assert all(spec.stage == "final" for spec in FINAL_MODEL_REGISTRY.values())
    assert all(
        (spec.feature_config_version, spec.feature_window_cycles, spec.feature_count)
        == ("past-only-v1", 30, 96)
        for spec in (*DEVELOPMENT_MODEL_REGISTRY.values(), *FINAL_MODEL_REGISTRY.values())
    )
    assert FINAL_MODEL_REGISTRY["dense-rul-mlp-final"].version == "dense-rul-mlp-final-v1"
    assert FINAL_MODEL_REGISTRY["xgboost-rul"].manifest_path.endswith(
        "artifacts/final/tree/bundles/xgboost-rul_manifest.json"
    )


def test_registry_mappings_are_immutable():
    with pytest.raises(TypeError):
        FINAL_MODEL_REGISTRY["new-model"] = FINAL_MODEL_REGISTRY["xgboost-rul"]
    with pytest.raises(TypeError):
        MODEL_REGISTRIES["candidate"] = DEVELOPMENT_MODEL_REGISTRY
