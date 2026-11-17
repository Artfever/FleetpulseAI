"""Synthetic checks for the bounded CPU MLP trainer."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models import mlp
from fleetpulse.models.mlp import (
    MAX_EPOCHS,
    DenseRULRegressor,
    MLPEpochResult,
    evaluate_mlp,
    train_mlp,
)
from fleetpulse.models.mlp_artifact import save_mlp_artifact, verify_mlp_artifact
from fleetpulse.models.registry import MODEL_REGISTRY
from fleetpulse.services import predict_rul


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
    result = train_mlp(
        fitting, validation, max_epochs=2, patience=2, batch_size=5, random_seed=7
    )

    assert 1 <= len(result.training_losses) <= 2
    assert len(result.validation_losses) == len(result.training_losses)
    assert len(result.validation_predictions) == len(validation.rows)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in (*result.training_losses,
                                                               *result.validation_losses))
    assert all(torch.isfinite(torch.tensor(value)) for value in result.validation_predictions)
    assert result.parameters["device"] == "cpu"
    assert result.parameters["max_epochs"] == 2
    assert result.parameters["epoch_budget"] == MAX_EPOCHS
    assert result.model.training is False
    assert result.feature_names == fitting.feature_names
    assert result.scaler.n_features_in_ == len(fitting.feature_names)
    assert result.best_epoch == int(np.argmin(result.validation_losses)) + 1
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
    options = {"max_epochs": 2, "patience": 2, "batch_size": 4, "random_seed": 11}
    first = train_mlp(fitting, validation, **options)
    second = train_mlp(fitting, validation, **options)

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
        train_mlp(fitting, validation, max_epochs=1, patience=1)


def test_training_rejects_epochs_over_run_budget(mlp_tables):
    with pytest.raises(ValueError, match="run budget"):
        train_mlp(*mlp_tables, max_epochs=MAX_EPOCHS + 1)


def test_scaler_statistics_use_fitting_rows_only_and_validation_transform_does_not_refit(
    mlp_tables,
):
    fitting, validation = mlp_tables
    result = train_mlp(fitting, validation, max_epochs=1, patience=1, batch_size=4)
    fitting_mean = np.asarray(fitting.X, dtype=np.float64).mean(axis=0)
    combined_mean = np.concatenate((fitting.X, validation.X), axis=0).mean(axis=0)
    saved_mean = result.scaler.mean_.copy()

    evaluation = evaluate_mlp(result.model, validation, batch_size=1, scaler=result.scaler)

    np.testing.assert_allclose(saved_mean, fitting_mean)
    assert not np.allclose(saved_mean, combined_mean)
    np.testing.assert_array_equal(result.scaler.mean_, saved_mean)
    assert len(evaluation.predictions) == len(validation.rows)


def test_early_stopping_restores_best_validation_checkpoint(mlp_tables, monkeypatch):
    fitting, validation = mlp_tables
    scores = iter((10.0, 9.0, 9.5, 9.6))
    model_states = []

    def controlled_evaluation(model, dataset, *, batch_size, scaler):
        model_states.append(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        )
        return MLPEpochResult(next(scores), (0.0,) * len(dataset.rows))

    monkeypatch.setattr(mlp, "evaluate_mlp", controlled_evaluation)
    result = train_mlp(
        fitting, validation, max_epochs=6, patience=2, batch_size=4, random_seed=5
    )

    assert result.best_epoch == 2
    assert result.stopped_early is True
    assert len(result.validation_losses) == 4
    for name, parameter in result.model.state_dict().items():
        assert torch.equal(parameter, model_states[1][name])


def test_saved_mlp_bundle_reloads_on_cpu_and_reproduces_predictions(mlp_tables, tmp_path):
    fitting, validation = mlp_tables
    result = train_mlp(
        fitting, validation, hidden_dims=(8, 4), max_epochs=3, patience=3, batch_size=8
    )
    saved = save_mlp_artifact(
        result, fitting, validation, tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )

    outcome = verify_mlp_artifact(
        saved["manifest_path"], fitting, validation,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )

    assert outcome == {
        "prediction_count": len(validation.rows),
        "best_epoch": result.best_epoch,
        "eval_mode": True,
        "device": "cpu",
        "max_prediction_difference_cycles": 0.0,
    }
    assert saved["payload"]["training"]["training_mse_cycles_squared"] == list(
        result.training_losses
    )
    assert saved["payload"]["training"]["validation_mse_cycles_squared"] == list(
        result.validation_losses
    )
    assert saved["payload"]["training"]["scaler"]["fitting_row_count"] == len(fitting.rows)


def test_mlp_bundle_rejects_changed_weights_or_dataset_hash(mlp_tables, tmp_path):
    fitting, validation = mlp_tables
    result = train_mlp(fitting, validation, max_epochs=1, patience=1, batch_size=16)
    saved = save_mlp_artifact(
        result, fitting, validation, tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="provenance"):
        verify_mlp_artifact(
            saved["manifest_path"], fitting, validation,
            fitting_dataset_sha256="d" * 64,
            validation_dataset_sha256="c" * 64,
        )

    saved["weights_path"].write_bytes(saved["weights_path"].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="weights hash"):
        verify_mlp_artifact(
            saved["manifest_path"], fitting, validation,
            fitting_dataset_sha256="b" * 64,
            validation_dataset_sha256="c" * 64,
        )


def test_registered_mlp_service_applies_saved_scaler_and_reloads_cpu_weights(mlp_tables, tmp_path):
    fitting, validation = mlp_tables
    result = train_mlp(
        fitting, validation, hidden_dims=(8, 4), max_epochs=2, patience=2, batch_size=8
    )
    saved = save_mlp_artifact(
        result,
        fitting,
        validation,
        tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    row = validation.rows[0]
    engine = row.vector.unit_id
    cutoff = row.vector.selected_cycle
    history = [
        {"unit_id": engine, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle + engine)}
        for cycle in range(1, cutoff + 1)
    ]

    prediction = predict_rul(
        history,
        dataset_partition="FD001/train",
        unit_id=engine,
        manifest_path=saved["manifest_path"],
    )

    assert prediction.raw_rul_cycles == pytest.approx(result.validation_predictions[0], abs=1e-6)
    assert MODEL_REGISTRY["dense-rul-mlp"].version == "dense-rul-mlp-v1"
    from fleetpulse.services import predictor

    loaded = predictor._load_model(saved["manifest_path"])[1]
    assert loaded.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    np.testing.assert_array_equal(loaded.scaler.mean_, result.scaler.mean_)


def test_registered_mlp_rejects_scaler_architecture_mismatch(mlp_tables, tmp_path):
    import hashlib
    import json

    from fleetpulse.services import PredictionArtifactError, predictor

    fitting, validation = mlp_tables
    result = train_mlp(fitting, validation, max_epochs=1, patience=1, batch_size=8)
    saved = save_mlp_artifact(
        result,
        fitting,
        validation,
        tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    document = json.loads(saved["manifest_path"].read_text(encoding="utf-8"))
    document["payload"]["training"]["scaler"]["feature_count"] += 1
    payload = json.dumps(
        document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    document["sha256"] = hashlib.sha256(payload).hexdigest()
    saved["manifest_path"].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PredictionArtifactError, match="scaler width"):
        predictor._load_model(saved["manifest_path"])


def test_model_comparison_loads_only_verified_mlp_scores(mlp_tables, tmp_path):
    from fleetpulse.evaluation.model_comparison import load_verified_mlp_scores

    fitting, validation = mlp_tables
    result = train_mlp(fitting, validation, max_epochs=2, patience=2, batch_size=8)
    artifact_dir = tmp_path / "mlp"
    artifact_dir.mkdir()
    save_mlp_artifact(
        result,
        fitting,
        validation,
        artifact_dir,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )

    scores = load_verified_mlp_scores(
        tmp_path,
        fitting,
        validation,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )

    assert scores is not None
    assert scores.model_version == "dense-rul-mlp-v1"
    assert scores.sample_count == len(validation.rows)
    assert scores.best_epoch == result.best_epoch
    assert scores.selected_validation_mse_cycles_squared == min(result.validation_losses)
    assert len(scores.predictions) == len(validation.rows)
