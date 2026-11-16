"""Supported saved estimators; shared inference uses their exact saved features.

Registration does not select, train, or download a model. Callers supply an
explicit manifest. Only trusted local bundles may be deserialized.
"""

from dataclasses import dataclass
from types import MappingProxyType

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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    artifact_schema: str
    filename: str
    estimator_type: type | None
    display_name: str


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
    }
)
