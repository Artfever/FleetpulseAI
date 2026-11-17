"""Fixed-epoch MLP refit for the frozen final FD001 model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fleetpulse.models.mlp import DEFAULT_BATCH_SIZE, DenseRULRegressor


@dataclass(frozen=True)
class FinalMLPTrainingResult:
    """Final model and training-only evidence from a fixed-epoch refit."""

    model: DenseRULRegressor
    training_losses: tuple[float, ...]
    feature_names: tuple[str, ...]
    random_seed: int
    parameters: dict[str, object]
    runtime_seconds: float


def train_final_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...],
    scaler: StandardScaler,
    *,
    hidden_dims: tuple[int, ...] = (128, 64),
    epochs: int = 8,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> FinalMLPTrainingResult:
    """Train on all supplied original-training rows without validation or stopping.

    The caller must supply labels from complete run-to-failure training histories
    and the fresh, independently persisted final scaler. This function has no
    validation input and always performs exactly ``epochs`` passes over the rows.
    """
    matrix = np.asarray(features, dtype=np.float64)
    label_values = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("features must be a nonempty finite two-dimensional matrix.")
    if label_values.shape != (matrix.shape[0],) or not np.isfinite(label_values).all():
        raise ValueError("targets must be one finite label per feature row.")
    if len(feature_names) != matrix.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must uniquely describe every feature column.")
    if not isinstance(scaler, StandardScaler) or not hasattr(scaler, "n_features_in_"):
        raise ValueError("scaler must be a fitted StandardScaler.")
    if scaler.n_features_in_ != matrix.shape[1]:
        raise ValueError("scaler feature count does not match the training matrix.")
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("random_seed", random_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "random_seed" else 1):
            raise ValueError(f"{name} must be a valid positive integer.")
    if not isinstance(hidden_dims, tuple) or not hidden_dims or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in hidden_dims
    ):
        raise ValueError("hidden_dims must be a nonempty tuple of positive integers.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or (
        not isfinite(learning_rate) or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive.")

    standardized = scaler.transform(matrix)
    x = torch.tensor(standardized, dtype=torch.float32, device="cpu")
    y = torch.tensor(label_values, dtype=torch.float32, device="cpu")
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise ValueError("standardized features and targets must remain finite.")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        model = DenseRULRegressor(matrix.shape[1], hidden_dims).to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_function = nn.MSELoss()
    losses: list[float] = []
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(random_seed + epoch)
        loader = DataLoader(
            TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=generator
        )
        squared_error_sum = 0.0
        observations = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Final MLP training produced a non-finite loss.")
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach().item()) * len(batch_y)
            observations += len(batch_y)
        epoch_loss = squared_error_sum / observations
        if not isfinite(epoch_loss):
            raise FloatingPointError("Final MLP training produced a non-finite epoch loss.")
        losses.append(epoch_loss)

    model.eval()
    return FinalMLPTrainingResult(
        model=model,
        training_losses=tuple(losses),
        feature_names=tuple(feature_names),
        random_seed=random_seed,
        parameters={
            "hidden_dims": hidden_dims,
            "activation": "ReLU",
            "batch_size": batch_size,
            "epochs_trained": epochs,
            "epoch_budget": epochs,
            "learning_rate": float(learning_rate),
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "scaler": "P12 final StandardScaler",
            "device": "cpu",
            "validation_used": False,
            "early_stopping": False,
        },
        runtime_seconds=perf_counter() - started,
    )
