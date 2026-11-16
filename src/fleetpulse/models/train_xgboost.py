"""One reproducible XGBoost candidate on the shared FD001 feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import TYPE_CHECKING

from fleetpulse.evaluation import (
    ValidationMetrics,
    ValidationPrediction,
    calculate_validation_metrics,
)
from fleetpulse.features.build_dataset import FeatureDataset
from fleetpulse.models.training_validation import (
    validate_integer_parameter,
    validate_training_datasets,
)

if TYPE_CHECKING:
    from xgboost import XGBRegressor

MODEL_VERSION = "xgboost-rul-v1"
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 4
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MIN_CHILD_WEIGHT = 1.0
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLSAMPLE_BYTREE = 0.8
DEFAULT_REG_LAMBDA = 1.0


@dataclass(frozen=True)
class XGBoostValidationResult:
    """Fitted estimator and one prediction for each saved validation engine."""

    model: XGBRegressor
    metrics: ValidationMetrics
    predictions: tuple[ValidationPrediction, ...]
    feature_names: tuple[str, ...]
    random_seed: int
    parameters: dict[str, int | float | str]
    runtime_seconds: float


def train_xgboost(
    fitting: FeatureDataset,
    validation: FeatureDataset,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_child_weight: float = DEFAULT_MIN_CHILD_WEIGHT,
    subsample: float = DEFAULT_SUBSAMPLE,
    colsample_bytree: float = DEFAULT_COLSAMPLE_BYTREE,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
) -> XGBoostValidationResult:
    """Fit only fitting X/y and score the unchanged validation rows once.

    The validated shared tables own feature construction, labels, split/cutoff
    choices, and source provenance. Validation labels are attached only after
    prediction. This function performs no search, early stopping, or refitting.
    XGBoost is imported lazily so the optional training dependency is not
    needed by users who only run the existing app and other models.
    """
    validate_training_datasets(fitting, validation)
    validate_integer_parameter(random_seed, "random_seed", allow_zero=True)
    validate_integer_parameter(n_estimators, "n_estimators")
    validate_integer_parameter(max_depth, "max_depth")
    _positive_finite(learning_rate, "learning_rate")
    _nonnegative_finite(min_child_weight, "min_child_weight")
    _bounded_fraction(subsample, "subsample")
    _bounded_fraction(colsample_bytree, "colsample_bytree")
    _nonnegative_finite(reg_lambda, "reg_lambda")

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "XGBoost training requires the optional project extra; install with "
            "`python -m pip install -e '.[xgboost]'`."
        ) from error

    parameters: dict[str, int | float | str] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": random_seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    model = XGBRegressor(**parameters)
    started = perf_counter()
    model.fit(fitting.X, fitting.y)
    predicted = model.predict(validation.X)
    runtime_seconds = perf_counter() - started

    predictions = tuple(
        ValidationPrediction(
            dataset_partition=row.vector.dataset_partition,
            unit_id=row.vector.unit_id,
            cutoff_cycle=row.vector.selected_cycle,
            prediction=float(value),
            true_rul=row.true_rul,
        )
        for row, value in zip(validation.rows, predicted, strict=True)
    )
    return XGBoostValidationResult(
        model=model,
        metrics=calculate_validation_metrics(predictions),
        predictions=predictions,
        feature_names=fitting.feature_names,
        random_seed=random_seed,
        parameters=parameters,
        runtime_seconds=runtime_seconds,
    )


def _positive_finite(value: Real, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def _nonnegative_finite(value: Real, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number.")


def _bounded_fraction(value: Real, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in (0, 1].")
