"""Supported saved estimators; shared inference uses their exact saved features.

Registration does not select, train, or download a model. Callers supply an
explicit manifest. Only trusted local bundles may be deserialized.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from fleetpulse.models import (
    forest_artifact,
    train_forest,
    train_tree,
    train_xgboost,
    tree_artifact,
    xgboost_artifact,
)
from fleetpulse.models.mlp_contract import MLP_ARTIFACT_SCHEMA, MODEL_NAME, MODEL_VERSION


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    artifact_schema: str
    filename: str
    estimator_type: type | None
    display_name: str
    stage: Literal["development"] = "development"
    feature_config_version: str = "past-only-v1"
    feature_window_cycles: int = 30
    feature_count: int = 96


@dataclass(frozen=True)
class FinalModelSpec:
    """Immutable inventory metadata for one P12 final refit bundle."""

    name: str
    version: str
    display_name: str
    manifest_path: str
    feature_config_version: str
    feature_window_cycles: int
    feature_count: int
    artifact_role: Literal["final-training-model-bundle"] = "final-training-model-bundle"
    stage: Literal["final"] = "final"


MODEL_REGISTRY = MappingProxyType(
    {
        "decision-tree-rul": ModelSpec(
            "decision-tree-rul",
            train_tree.MODEL_VERSION,
            tree_artifact.TREE_ARTIFACT_SCHEMA,
            tree_artifact.MODEL_FILENAME,
            DecisionTreeRegressor,
            "Decision Tree",
        ),
        "random-forest-rul": ModelSpec(
            "random-forest-rul",
            train_forest.MODEL_VERSION,
            forest_artifact.FOREST_ARTIFACT_SCHEMA,
            forest_artifact.MODEL_FILENAME,
            RandomForestRegressor,
            "Random Forest",
        ),
        "xgboost-rul": ModelSpec(
            "xgboost-rul",
            train_xgboost.MODEL_VERSION,
            xgboost_artifact.XGBOOST_ARTIFACT_SCHEMA,
            xgboost_artifact.MODEL_FILENAME,
            None,
            "XGBoost",
        ),
        MODEL_NAME: ModelSpec(
            MODEL_NAME,
            MODEL_VERSION,
            MLP_ARTIFACT_SCHEMA,
            "mlp_best_state.pt",
            None,
            "Dense RUL MLP",
        ),
    }
)

# Keep the predictor's registered development candidates separate from the
# final, full-training refits. The latter are loaded through their dedicated
# checksum-verifying bundle loaders, not through the development predictor.
DEVELOPMENT_MODEL_REGISTRY = MODEL_REGISTRY
FINAL_MODEL_REGISTRY = MappingProxyType(
    {
        "decision-tree-rul": FinalModelSpec(
            "decision-tree-rul",
            "decision-tree-rul-v1",
            "Decision Tree",
            "artifacts/final/tree/bundles/decision-tree-rul_manifest.json",
            "past-only-v1",
            30,
            96,
        ),
        "random-forest-rul": FinalModelSpec(
            "random-forest-rul",
            "random-forest-rul-v1",
            "Random Forest",
            "artifacts/final/tree/bundles/random-forest-rul_manifest.json",
            "past-only-v1",
            30,
            96,
        ),
        "xgboost-rul": FinalModelSpec(
            "xgboost-rul",
            "xgboost-rul-v1",
            "XGBoost",
            "artifacts/final/tree/bundles/xgboost-rul_manifest.json",
            "past-only-v1",
            30,
            96,
        ),
        "constant-rul": FinalModelSpec(
            "constant-rul",
            "constant-rul-v1",
            "Constant mean baseline",
            "artifacts/final/tree/bundles/constant-rul_manifest.json",
            "past-only-v1",
            30,
            96,
        ),
        "dense-rul-mlp-final": FinalModelSpec(
            "dense-rul-mlp-final",
            "dense-rul-mlp-final-v1",
            "Dense RUL MLP",
            "artifacts/final/mlp/bundle/final_mlp_manifest.json",
            "past-only-v1",
            30,
            96,
        ),
    }
)

MODEL_REGISTRIES = MappingProxyType(
    {"development": DEVELOPMENT_MODEL_REGISTRY, "final": FINAL_MODEL_REGISTRY}
)
