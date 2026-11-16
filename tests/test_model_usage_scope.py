"""Training-model replay labels must reflect saved engine membership."""

import pytest

from fleetpulse.models.usage_scope import classify_training_engine


@pytest.fixture
def split():
    return {
        "fitting_engines": [
            {"dataset_partition": "FD001/train", "unit_id": 1},
        ],
        "validation_prefixes": [
            {"dataset_partition": "FD001/train", "unit_id": 2, "target_rul": 99},
        ],
    }


def test_classifies_fitting_validation_and_unknown_engines(split):
    assert classify_training_engine(split, dataset_partition="FD001/train", unit_id=1) == "fitting"
    assert (
        classify_training_engine(split, dataset_partition="FD001/train", unit_id=2) == "validation"
    )
    assert classify_training_engine(split, dataset_partition="FD001/test", unit_id=2) == "unknown"


def test_rejects_split_with_overlapping_engine_identity(split):
    split["validation_prefixes"].append(
        {"dataset_partition": "FD001/train", "unit_id": 1, "target_rul": 12}
    )

    with pytest.raises(ValueError, match="overlap"):
        classify_training_engine(split, dataset_partition="FD001/train", unit_id=1)
