"""Small CPU dense/ReLU regressor and bounded batch training loops."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fleetpulse.features.build_dataset import FeatureDataset
from fleetpulse.models.training_validation import (
    validate_integer_parameter,
    validate_training_datasets,
)

DEFAULT_HIDDEN_DIMS = (64, 32)
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 3
MAX_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_RANDOM_SEED = 42


class DenseRULRegressor(nn.Module):
    """A compact fully connected regressor with ReLU hidden layers."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS):
        super().__init__()
        validate_integer_parameter(input_dim, "input_dim")
        if not isinstance(hidden_dims, tuple) or not hidden_dims:
            raise ValueError("hidden_dims must be a nonempty tuple of positive integers.")
        for width in hidden_dims:
            validate_integer_parameter(width, "hidden layer width")

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar RUL estimate per feature row."""
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape (n, {self.input_dim}).")
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class MLPEpochResult:
    """Mean squared error and predictions from one validation pass."""

    mean_squared_error: float
    predictions: tuple[float, ...]


@dataclass(frozen=True)
class MLPTrainingResult:
    """Model and per-epoch losses from a single bounded fitting run."""

    model: DenseRULRegressor
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    validation_predictions: tuple[float, ...]
    feature_names: tuple[str, ...]
    random_seed: int
    parameters: dict[str, object]
    runtime_seconds: float


def _feature_tensor(dataset: FeatureDataset) -> torch.Tensor:
    values = torch.tensor(dataset.X, dtype=torch.float32, device="cpu")
    if not torch.isfinite(values).all().item():
        raise ValueError("Feature values must all be finite.")
    return values


def _target_tensor(dataset: FeatureDataset) -> torch.Tensor:
    values = torch.tensor(dataset.y, dtype=torch.float32, device="cpu")
    if not torch.isfinite(values).all().item():
        raise ValueError("RUL targets must all be finite.")
    return values


def evaluate_mlp(
    model: DenseRULRegressor,
    dataset: FeatureDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> MLPEpochResult:
    """Evaluate in batches without gradients or parameter updates."""
    validate_integer_parameter(batch_size, "batch_size")
    if not isinstance(dataset, FeatureDataset) or not dataset.rows:
        raise ValueError("dataset must be a nonempty FeatureDataset.")
    if not isinstance(model, DenseRULRegressor):
        raise TypeError("model must be a DenseRULRegressor.")
    features = _feature_tensor(dataset)
    targets = _target_tensor(dataset)
    if features.shape[1] != model.input_dim:
        raise ValueError("Dataset feature count does not match the model input dimension.")

    model.to("cpu")
    model.eval()
    loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=False)
    squared_error_sum = 0.0
    predictions: list[float] = []
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_predictions = model(batch_features)
            batch_loss = nn.functional.mse_loss(batch_predictions, batch_targets, reduction="sum")
            squared_error_sum += float(batch_loss.item())
            predictions.extend(float(value) for value in batch_predictions.tolist())
    mean_squared_error = squared_error_sum / len(dataset.rows)
    if not isfinite(mean_squared_error) or not all(isfinite(value) for value in predictions):
        raise FloatingPointError("Validation produced a non-finite loss or prediction.")
    return MLPEpochResult(mean_squared_error, tuple(predictions))


def train_mlp(
    fitting: FeatureDataset,
    validation: FeatureDataset,
    *,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> MLPTrainingResult:
    """Train on fitting rows only, recording a batched validation pass per epoch.

    The run is CPU-only and capped at ``MAX_EPOCHS``. Scaling and validation-based
    early stopping are deliberately left to P10-S2.
    """
    validate_training_datasets(fitting, validation)
    for name, value in (("batch_size", batch_size), ("epochs", epochs)):
        validate_integer_parameter(value, name)
    if epochs > MAX_EPOCHS:
        raise ValueError(f"epochs must not exceed the run budget of {MAX_EPOCHS}.")
    validate_integer_parameter(random_seed, "random_seed", allow_zero=True)
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
        raise TypeError("learning_rate must be a finite positive number.")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")

    fitting_features = _feature_tensor(fitting)
    fitting_targets = _target_tensor(fitting)
    _feature_tensor(validation)
    if fitting_features.shape[1] != len(fitting.feature_names):
        raise ValueError("Fitting feature count does not match its feature names.")

    # Isolate model initialization from the caller's global Torch RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        model = DenseRULRegressor(fitting_features.shape[1], hidden_dims).to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_function = nn.MSELoss()
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_predictions: tuple[float, ...] = ()
    started = perf_counter()

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(random_seed + epoch)
        loader = DataLoader(
            TensorDataset(fitting_features, fitting_targets),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        squared_error_sum = 0.0
        observations = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training produced a non-finite loss.")
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach().item()) * len(batch_targets)
            observations += len(batch_targets)

        training_loss = squared_error_sum / observations
        validation_result = evaluate_mlp(
            model, validation, batch_size=batch_size
        )
        if not isfinite(training_loss):
            raise FloatingPointError("Training produced a non-finite epoch loss.")
        training_losses.append(training_loss)
        validation_losses.append(validation_result.mean_squared_error)
        validation_predictions = validation_result.predictions

    runtime = perf_counter() - started
    model.eval()
    return MLPTrainingResult(
        model=model,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        validation_predictions=validation_predictions,
        feature_names=fitting.feature_names,
        random_seed=random_seed,
        parameters={
            "hidden_dims": hidden_dims,
            "activation": "ReLU",
            "batch_size": batch_size,
            "epochs": epochs,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": float(learning_rate),
            "loss": "mean_squared_error",
            "device": "cpu",
        },
        runtime_seconds=runtime,
    )
