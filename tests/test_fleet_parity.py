"""Batch and independent saved-model predictions agree by engine identity."""

import pytest

from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models.train_tree import train_decision_tree
from fleetpulse.models.tree_artifact import save_decision_tree_artifact
from fleetpulse.services import predict_fleet, predict_rul_with_context


def _row(unit_id, cycle):
    return {
        field: unit_id
        if field == "unit_id"
        else cycle
        if field == "cycle"
        else 2.0 * cycle
        if field == "sensor_1"
        else 10.0 * cycle
        if field == "sensor_2"
        else 0.5
        for field in HISTORY_COLUMNS
    }


@pytest.fixture
def saved_synthetic_bundle(tmp_path):
    rows = [
        _row(unit_id, cycle)
        for unit_id, final_cycle in ((1, 5), (2, 7), (3, 9), (4, 11))
        for cycle in range(1, final_cycle + 1)
    ]
    split = build_split_manifest(
        rows,
        dataset_partition="FD001/train",
        source_path="synthetic/p15_train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    config = FeatureConfig(channels=("sensor_2", "sensor_1"), window_cycles=2)
    datasets = {
        "manifest": split,
        "source_sha256": "a" * 64,
        "config": config,
    }
    fitting = build_fitting_dataset(rows, **datasets)
    validation = build_validation_dataset(rows, **datasets)
    trained = train_decision_tree(fitting, validation)
    paths = save_decision_tree_artifact(
        trained,
        fitting,
        validation,
        tmp_path / "bundle",
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    return paths["manifest_path"]


def test_batch_predictions_match_single_engine_calls_for_same_saved_bundle(saved_synthetic_bundle):
    partition = "synthetic/P15-S1"
    engine_histories = {
        5: [_row(5, cycle) for cycle in range(1, 5)],
        2: [_row(2, cycle) for cycle in range(1, 4)],
        9: [_row(9, cycle) for cycle in range(1, 2)],
    }
    interleaved = [
        engine_histories[5][0],
        engine_histories[2][0],
        engine_histories[9][0],
        engine_histories[5][1],
        engine_histories[2][1],
        engine_histories[5][2],
        engine_histories[2][2],
        engine_histories[5][3],
    ]

    batch = predict_fleet(
        interleaved,
        dataset_partition=partition,
        manifest_path=saved_synthetic_bundle,
    )
    single = {
        unit_id: predict_rul_with_context(
            history,
            dataset_partition=partition,
            unit_id=unit_id,
            manifest_path=saved_synthetic_bundle,
        )
        for unit_id, history in engine_histories.items()
    }

    assert [result.identity.unit_id for result in batch] == [2, 5, 9]
    assert not any(result.error for result in batch)
    for result in batch:
        expected = single[result.identity.unit_id]
        assert result.identity.dataset_partition == partition
        assert result.observed_cycle == expected.observed_cycle
        assert result.raw_rul_cycles == expected.raw_rul_cycles
        assert result.displayed_rul_cycles == expected.displayed_rul_cycles
        assert result.model_name == expected.model_name
        assert result.model_version == expected.model_version
        assert result.input_window_rows == expected.input_window_rows
        assert result.warnings == expected.warnings
