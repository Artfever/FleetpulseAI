"""Contracts for the fixed-epoch final MLP refit."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from fleetpulse.models.final_mlp import train_final_mlp


def test_final_mlp_runs_exact_fixed_epoch_count_deterministically() -> None:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
        dtype=np.float64,
    )
    targets = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    scaler = StandardScaler().fit(features)
    arguments = (features, targets, ("x", "y"), scaler)

    first = train_final_mlp(
        *arguments, hidden_dims=(4, 2), epochs=2, batch_size=3, random_seed=19
    )
    second = train_final_mlp(
        *arguments, hidden_dims=(4, 2), epochs=2, batch_size=3, random_seed=19
    )

    assert len(first.training_losses) == 2
    assert first.parameters["validation_used"] is False
    assert first.parameters["early_stopping"] is False
    assert first.training_losses == second.training_losses
    assert all(
        np.array_equal(first.model.state_dict()[name].numpy(), second.model.state_dict()[name].numpy())
        for name in first.model.state_dict()
    )


def test_final_mlp_rejects_feature_or_target_mismatch() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    scaler = StandardScaler().fit(features)

    try:
        train_final_mlp(features, np.asarray([1.0]), ("x", "y"), scaler, epochs=1)
    except ValueError as error:
        assert "one finite label per feature row" in str(error)
    else:
        raise AssertionError("mismatched labels must be rejected")
