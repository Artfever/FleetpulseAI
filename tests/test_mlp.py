"""Synthetic checks for the bounded CPU MLP trainer."""

from dataclasses import replace

import pytest
import torch

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models.mlp import (
    MAX_EPOCHS,
    DenseRULRegressor,
    evaluate_mlp,
    train_mlp,
)


@pytest.fixture
def mlp_tables():
    rows = [
        {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle + unit)}
        for unit, lifetime in ((1, 5), (2, 7), (3, 9), (4, 11))
        for cycle in range(1, lifetime + 1)
    ]
    split = build_split_manifest(
        rows,
        dataset_partition="FD001/train",
        source_path="synthetic/train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    options = {
        "manifest": split,
        "source_sha256": "a" * 64,
        "config": FeatureConfig(channels=("sensor_1",)),
    }
    return build_fitting_dataset(rows, **options), build_validation_dataset(rows, **options)


def test_dense_regressor_returns_one_value_per_row_and_checks_shape():
    model = DenseRULRegressor(input_dim=4, hidden_dims=(8, 4))
    assert model(torch.ones((3, 4))).shape == (3,)
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones((3, 5)))


def test_training_uses_batches_and_returns_finite_losses_and_predictions(mlp_tables):
    fitting, validation = mlp_tables
    result = train_mlp(fitting, validation, epochs=2, batch_size=5, random_seed=7)

    assert len(result.training_losses) == len(result.validation_losses) == 2
    assert len(result.validation_predictions) == len(validation.rows)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in (*result.training_losses,
                                                               *result.validation_losses))
    assert all(torch.isfinite(torch.tensor(value)) for value in result.validation_predictions)
    assert result.parameters["device"] == "cpu"
    assert result.parameters["max_epochs"] == MAX_EPOCHS
    assert result.model.training is False
    assert result.feature_names == fitting.feature_names
    assert result.runtime_seconds >= 0


def test_validation_is_batched_and_does_not_update_parameters(mlp_tables):
    _, validation = mlp_tables
    model = DenseRULRegressor(input_dim=len(validation.feature_names))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = evaluate_mlp(model, validation, batch_size=1)

    assert len(result.predictions) == len(validation.rows)
    assert torch.isfinite(torch.tensor(result.mean_squared_error))
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_training_is_repeatable_for_same_seed(mlp_tables):
    fitting, validation = mlp_tables
    first = train_mlp(fitting, validation, epochs=2, batch_size=4, random_seed=11)
    second = train_mlp(fitting, validation, epochs=2, batch_size=4, random_seed=11)

    assert first.training_losses == second.training_losses
    assert first.validation_losses == second.validation_losses
    assert first.validation_predictions == second.validation_predictions


@pytest.mark.parametrize(
    "fitting_change,validation_change",
    [
        (None, {"role": "fitting"}),
        ({"source_sha256": "b" * 64}, None),
        (None, {"config": FeatureConfig(channels=("sensor_2",))}),
    ],
)
def test_training_rejects_invalid_dataset_pairs(mlp_tables, fitting_change, validation_change):
    fitting, validation = mlp_tables
    if fitting_change:
        fitting = replace(fitting, **fitting_change)
    if validation_change:
        validation = replace(validation, **validation_change)

    with pytest.raises(ValueError):
        train_mlp(fitting, validation, epochs=1)


def test_training_rejects_epochs_over_run_budget(mlp_tables):
    with pytest.raises(ValueError, match="run budget"):
        train_mlp(*mlp_tables, epochs=MAX_EPOCHS + 1)
