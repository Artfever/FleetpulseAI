"""P22-S2 bootstrap resamples complete, paired engine errors."""

import pytest

from scripts.run_p22_s2_engine_bootstrap import (
    bootstrap_mae_distributions,
    percentile,
)


def test_seeded_bootstrap_reuses_common_whole_engine_indices():
    draws = bootstrap_mae_distributions(
        {"a": [1.0, 3.0], "b": [2.0, 4.0]},
        replicates=4,
        seed=42,
    )
    assert draws == {
        "a": [1.0, 2.0, 1.0, 1.0],
        "b": [2.0, 3.0, 2.0, 2.0],
    }


def test_bootstrap_is_repeatable_and_uses_full_engine_sample_size():
    first = bootstrap_mae_distributions({"model": [1.0, 4.0, 7.0]}, replicates=25, seed=7)
    second = bootstrap_mae_distributions({"model": [1.0, 4.0, 7.0]}, replicates=25, seed=7)
    assert first == second
    assert len(first["model"]) == 25
    assert all(value in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0} for value in first["model"])


def test_percentile_uses_linear_interpolation():
    assert percentile([1.0, 3.0, 5.0], 0.25) == 2.0
    assert percentile([1.0, 3.0, 5.0], 0.975) == 4.9


@pytest.mark.parametrize(
    "errors,kwargs,message",
    [
        ({}, {}, "at least one model"),
        ({"model": [1.0]}, {}, "at least two validation engines"),
        ({"a": [1.0, 2.0], "b": [1.0]}, {}, "shared engine count"),
        ({"model": [1.0, -1.0]}, {}, "invalid absolute error"),
        ({"model": [1.0, float("nan")]}, {}, "invalid absolute error"),
        ({"model": [1.0, 2.0]}, {"replicates": 0}, "positive integer"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(errors, kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_mae_distributions(errors, **kwargs)
