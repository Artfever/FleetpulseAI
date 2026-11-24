"""Load and serve the saved C-MAPSS subset models."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fleetpulse.data.quality import SENSOR_COLUMNS, SETTING_COLUMNS, validate_engine_history
from fleetpulse.features.config import FeatureConfig
from fleetpulse.features.engineering import engineer_features

ENVIRONMENTS = {
    "FD001": "1 operating condition · 1 fault mode",
    "FD002": "6 operating conditions · 1 fault mode",
    "FD003": "1 operating condition · 2 fault modes",
    "FD004": "6 operating conditions · 2 fault modes",
}
NEW_ENVIRONMENTS = ("FD002", "FD003", "FD004")
FEATURES = FeatureConfig()
BUNDLE_SCHEMA = "cmapss-xgboost-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def group_histories(records):
    quality = validate_engine_history(records)
    if not records or quality.errors:
        message = quality.errors[0].message if quality.errors else "History is empty."
        raise ValueError(message)
    by_engine = defaultdict(list)
    for row in records:
        by_engine[row["unit_id"]].append(row)
    return dict(by_engine)


def _matrix(rows, columns):
    return np.asarray([[row[key] for key in columns] for row in rows], dtype=float)


@dataclass(frozen=True)
class ConditionTransform:
    """Nearest operating condition in standardized setting space."""

    center: np.ndarray
    scale: np.ndarray
    prototypes: np.ndarray
    sensor_mean: np.ndarray
    sensor_scale: np.ndarray

    def __post_init__(self):
        expected = ((3,), (3,), (6, 3), (6, 21), (6, 21))
        arrays = (self.center, self.scale, self.prototypes, self.sensor_mean, self.sensor_scale)
        if any(
            a.shape != shape or not np.isfinite(a).all()
            for a, shape in zip(arrays, expected, strict=True)
        ):
            raise ValueError("Invalid operating condition transform.")
        if (self.scale <= 0).any() or (self.sensor_scale <= 0).any():
            raise ValueError("Operating condition scales must be positive.")

    @staticmethod
    def assignments(settings, prototypes):
        return ((settings[:, None, :] - prototypes) ** 2).sum(axis=2).argmin(axis=1)

    @classmethod
    def fit(cls, rows):
        from sklearn.cluster import KMeans

        settings = _matrix(rows, SETTING_COLUMNS)
        center = settings.mean(axis=0)
        scale = settings.std(axis=0)
        scale[scale < 1e-9] = 1
        settings = (settings - center) / scale
        prototypes = KMeans(n_clusters=6, n_init=10, random_state=42).fit(settings).cluster_centers_
        groups = cls.assignments(settings, prototypes)
        sensors = _matrix(rows, SENSOR_COLUMNS)
        if len(set(groups)) != 6:
            raise ValueError("The fitting histories must cover all six operating conditions.")
        means = np.stack([sensors[groups == i].mean(axis=0) for i in range(6)])
        scales = np.stack([sensors[groups == i].std(axis=0) for i in range(6)])
        scales[scales < 1e-9] = 1
        return cls(center, scale, prototypes, means, scales)

    def apply(self, rows):
        settings = (_matrix(rows, SETTING_COLUMNS) - self.center) / self.scale
        groups = self.assignments(settings, self.prototypes)
        sensors = (_matrix(rows, SENSOR_COLUMNS) - self.sensor_mean[groups]) / self.sensor_scale[
            groups
        ]
        return [
            {**row, **dict(zip(SENSOR_COLUMNS, values.tolist(), strict=True))}
            for row, values in zip(rows, sensors, strict=True)
        ]


@dataclass(frozen=True)
class ObservedHistory:
    """Validated engine history with bounded windows for training and prediction."""

    subset: str
    rows: list
    cycles: tuple[int, ...]

    @classmethod
    def build(cls, subset, rows, transform, *, validated=False):
        if not validated and len(group_histories(rows)) != 1:
            raise ValueError("Select a single engine history.")
        return cls(
            subset,
            transform.apply(rows) if transform else rows,
            tuple(row["cycle"] for row in rows),
        )

    def features(self, cutoff):
        end = bisect_right(self.cycles, cutoff)
        start = bisect_left(self.cycles, cutoff - FEATURES.window_cycles + 1)
        vector = engineer_features(
            self.rows[start:end],
            dataset_partition=self.subset,
            unit_id=self.rows[0]["unit_id"],
            selected_cycle=cutoff,
            config=FEATURES,
        )
        return replace(vector, observed_row_count=end)


@dataclass(frozen=True)
class EnvironmentModel:
    subset: str
    version: str
    estimator: object
    transform: ConditionTransform | None

    def predict(self, rows, cutoffs):
        cutoffs = tuple(cutoffs)
        if not cutoffs:
            raise ValueError("Select at least one observed cycle.")
        if len(group_histories(rows)) != 1:
            raise ValueError("Select a single engine history.")
        last_cutoff = max(cutoffs)
        visible = [row for row in rows if row["cycle"] <= last_cutoff]
        if not visible:
            raise ValueError("Select a cycle in the observed history.")
        history = ObservedHistory.build(self.subset, visible, self.transform, validated=True)
        vectors = [history.features(cycle) for cycle in cutoffs]
        values = self.estimator.predict(np.asarray([v.values for v in vectors], dtype=np.float32))
        if len(values) != len(vectors) or not np.isfinite(values).all():
            raise ValueError("The environment model returned invalid predictions.")
        return tuple(zip(vectors, map(float, values), strict=True))


def load_environment_model(directory: Path, subset: str) -> EnvironmentModel:
    from xgboost import XGBRegressor

    if subset not in NEW_ENVIRONMENTS:
        raise ValueError("Unknown environment model.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    version = f"{subset.lower()}-xgboost-v1"
    if (
        manifest["schema"] != BUNDLE_SCHEMA
        or manifest["subset"] != subset
        or manifest["version"] != version
        or manifest["feature_version"] != FEATURES.version
        or manifest["window_cycles"] != FEATURES.window_cycles
        or tuple(manifest["feature_names"]) != FEATURES.output_schema
    ):
        raise ValueError("Environment bundle metadata is incompatible.")
    payload = manifest["condition_transform"]
    if (payload is None) != (subset == "FD003"):
        raise ValueError("Environment bundle has the wrong preprocessing.")
    transform = (
        None
        if payload is None
        else ConditionTransform(
            **{key: np.asarray(value, dtype=float) for key, value in payload.items()}
        )
    )
    model_path = directory / "model.json"
    if sha256(model_path) != manifest["model_sha256"]:
        raise ValueError("Environment model checksum does not match.")
    estimator = XGBRegressor(n_jobs=2)
    estimator.load_model(model_path)
    estimator.set_params(n_jobs=2)
    if estimator.n_features_in_ != len(FEATURES.output_schema):
        raise ValueError("Environment model feature width does not match.")
    return EnvironmentModel(subset, version, estimator, transform)
