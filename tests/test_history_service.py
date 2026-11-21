"""P16 history replay sends only each observed prefix to saved inference."""

import json
from itertools import count

import pytest

from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.services import PredictionInputError, RULPrediction
from fleetpulse.services import history as history_service


def _row(unit_id, cycle, *, sensor_1=0.25):
    return {
        field: unit_id if field == "unit_id" else cycle if field == "cycle" else sensor_1
        for field in HISTORY_COLUMNS
    }


def _write_cache_bundle(directory, *, version="model-v1", feature_config=None, model_bytes=b"model"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(model_bytes)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "payload": {
                    "model": {"filename": "model.bin", "version": version},
                    "training": {
                        "feature_config": feature_config or {"window_cycles": 30, "channels": ["sensor_1"]}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _controlled_prediction(calls):
    def predict(prefix, **kwargs):
        calls.append((prefix, kwargs))
        cutoff = int(prefix[-1]["cycle"])
        return RULPrediction(
            float(cutoff), float(cutoff), "test-model", "test-v1", cutoff, len(prefix), ()
        )

    return predict


def test_replay_predicts_each_sorted_cutoff_from_only_its_engine_prefix(monkeypatch):
    records = [
        row
        for cycle in range(1, 6)
        for row in (_row(4, cycle), _row(8, cycle, sensor_1=9.0))
    ]
    calls = []

    def controlled_prediction(prefix, **kwargs):
        cycles = [int(row["cycle"]) for row in prefix]
        unit_ids = {int(row["unit_id"]) for row in prefix}
        assert len(unit_ids) == 1
        assert unit_ids == {4}
        calls.append((cycles, kwargs))
        cutoff = cycles[-1]
        return RULPrediction(
            float(cutoff),
            float(cutoff),
            "test-model",
            "test-v1",
            cutoff,
            len(cycles),
            (),
        )

    monkeypatch.setattr(history_service, "predict_rul_with_context", controlled_prediction)

    replay = history_service.replay_engine_history(
        records,
        dataset_partition="FD001/train",
        unit_id=4,
        selected_cycle=4,
        replay_cutoffs=[4, 2],
        manifest_path="test-manifest.json",
    )

    assert replay.identity.dataset_partition == "FD001/train"
    assert replay.identity.unit_id == 4
    assert replay.selected_cycle == 4
    assert [point.cutoff_cycle for point in replay.points] == [2, 4]
    assert [(point.prediction.observed_cycle, point.prediction.raw_rul_cycles) for point in replay.points] == [
        (2, 2.0),
        (4, 4.0),
    ]
    assert [call[0] for call in calls] == [[1, 2], [1, 2, 3, 4]]
    assert all(call[1]["dataset_partition"] == "FD001/train" for call in calls)
    assert all(call[1]["unit_id"] == 4 for call in calls)
    assert all(call[1]["manifest_path"] == "test-manifest.json" for call in calls)


def test_each_cutoff_gets_an_independent_prefix_copy(monkeypatch):
    records = [_row(4, cycle, sensor_1=float(cycle)) for cycle in range(1, 5)]
    seen_sensor_values = []

    def mutating_predictor(prefix, **kwargs):
        seen_sensor_values.append([row["sensor_1"] for row in prefix])
        if len(seen_sensor_values) == 1:
            prefix[0]["sensor_1"] = -999.0
        cutoff = int(prefix[-1]["cycle"])
        return RULPrediction(float(cutoff), float(cutoff), "test-model", "test-v1", cutoff, len(prefix), ())

    monkeypatch.setattr(history_service, "predict_rul_with_context", mutating_predictor)

    history_service.replay_engine_history(
        records,
        dataset_partition="FD001/train",
        unit_id=4,
        selected_cycle=4,
        replay_cutoffs=[2, 4],
        manifest_path="test-manifest.json",
    )

    assert seen_sensor_values == [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]]
    assert [row["sensor_1"] for row in records] == [1.0, 2.0, 3.0, 4.0]


def test_replay_cache_reuses_identical_history_model_and_feature_version(tmp_path, monkeypatch):
    history_service.clear_history_replay_cache()
    manifest = _write_cache_bundle(tmp_path / "bundle")
    records = [_row(4, cycle, sensor_1=float(cycle)) for cycle in range(1, 5)]
    calls = []
    monkeypatch.setattr(history_service, "predict_rul_with_context", _controlled_prediction(calls))

    for _ in range(2):
        replay = history_service.replay_engine_history(
            records,
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=4,
            replay_cutoffs=[2, 4],
            manifest_path=manifest,
        )
        assert [point.prediction.raw_rul_cycles for point in replay.points] == [2.0, 4.0]

    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    ["history", "dataset_partition", "engine_id", "model_version", "feature_config", "model_bytes"],
)
def test_replay_cache_invalidates_on_history_or_saved_bundle_change(tmp_path, monkeypatch, change):
    history_service.clear_history_replay_cache()
    bundle = tmp_path / "bundle"
    manifest = _write_cache_bundle(bundle)
    records = [_row(4, cycle, sensor_1=float(cycle)) for cycle in range(1, 5)]
    calls = []
    monkeypatch.setattr(history_service, "predict_rul_with_context", _controlled_prediction(calls))

    def replay(current_records=records, *, partition="FD001/train", unit_id=4):
        return history_service.replay_engine_history(
            current_records,
            dataset_partition=partition,
            unit_id=unit_id,
            selected_cycle=4,
            replay_cutoffs=[4],
            manifest_path=manifest,
        )

    replay()
    if change == "history":
        changed_records = [dict(row) for row in records]
        changed_records[-1]["sensor_1"] = 99.0
        replay(changed_records)
    elif change == "dataset_partition":
        replay(partition="FD001/test")
    elif change == "engine_id":
        changed_records = [dict(row, unit_id=8) for row in records]
        replay(changed_records, unit_id=8)
    elif change == "model_bytes":
        (bundle / "model.bin").write_bytes(b"changed model")
        replay()
    else:
        updated_manifest = _write_cache_bundle(
            bundle,
            version="model-v2" if change == "model_version" else "model-v1",
            feature_config={"window_cycles": 20, "channels": ["sensor_1"]}
            if change == "feature_config"
            else None,
        )
        manifest = updated_manifest
        replay()

    assert len(calls) == 2


def test_replay_cache_is_bounded_and_evicts_least_recently_used(tmp_path, monkeypatch):
    history_service.clear_history_replay_cache()
    monkeypatch.setattr(history_service, "MAX_REPLAY_CACHE_POINTS", 2)
    manifest = _write_cache_bundle(tmp_path / "bundle")
    calls = []
    monkeypatch.setattr(history_service, "predict_rul_with_context", _controlled_prediction(calls))

    for unit_id in (1, 2, 3):
        history_service.replay_engine_history(
            [_row(unit_id, cycle) for cycle in range(1, 3)],
            dataset_partition="FD001/train",
            unit_id=unit_id,
            selected_cycle=2,
            replay_cutoffs=[2],
            manifest_path=manifest,
        )

    assert len(history_service._REPLAY_CACHE) == 2
    history_service.replay_engine_history(
        [_row(1, cycle) for cycle in range(1, 3)],
        dataset_partition="FD001/train",
        unit_id=1,
        selected_cycle=2,
        replay_cutoffs=[2],
        manifest_path=manifest,
    )
    assert len(calls) == 4


@pytest.mark.parametrize(
    ("selected_cycle", "cutoffs", "message"),
    [
        (4, [], "at least one"),
        (4, [2, 2], "duplicate"),
        (4, [2, 5], "later than selected_cycle"),
        (5, range(1, 12), "cannot contain more than"),
    ],
)
def test_replay_rejects_empty_duplicate_future_and_oversized_grids(
    selected_cycle, cutoffs, message, monkeypatch
):
    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda *args, **kwargs: pytest.fail("invalid replay grid must not predict"),
    )

    with pytest.raises(PredictionInputError, match=message):
        history_service.replay_engine_history(
            [_row(4, cycle) for cycle in range(1, 6)],
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=selected_cycle,
            replay_cutoffs=cutoffs,
            manifest_path="test-manifest.json",
        )


def test_replay_rejects_an_unobserved_selected_cycle():
    with pytest.raises(PredictionInputError, match="selected_cycle must be an observed"):
        history_service.replay_engine_history(
            [_row(4, cycle) for cycle in range(1, 5)],
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=5,
            replay_cutoffs=[2],
            manifest_path="test-manifest.json",
        )


def test_replay_rejects_a_cutoff_missing_from_engine_history(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda *args, **kwargs: pytest.fail("unobserved cutoff must not predict"),
    )
    with pytest.raises(PredictionInputError, match="Every replay cutoff must be an observed cycle"):
        history_service.replay_engine_history(
            [_row(4, cycle) for cycle in (1, 2, 4, 5)],
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=4,
            replay_cutoffs=[3],
            manifest_path="test-manifest.json",
        )


@pytest.mark.parametrize("unit_id", [4.0])
def test_replay_keeps_accepting_integral_numeric_engine_ids(unit_id):
    selected = history_service._select_replay_engine_records(
        (_row(unit_id, 1),), 4
    )
    assert len(selected) == 1


def test_replay_still_rejects_invalid_identity_in_an_unselected_row(monkeypatch):
    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda *args, **kwargs: pytest.fail("invalid row identity must fail before prediction"),
    )
    rows = [_row(4, 1), _row(0, 1)]
    with pytest.raises(PredictionInputError, match="no valid engine identity"):
        history_service.replay_engine_history(
            rows,
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=1,
            replay_cutoffs=[1],
            manifest_path="test-manifest.json",
        )


def test_replay_caps_iteration_for_unbounded_cutoff_iterables():
    with pytest.raises(PredictionInputError, match="cannot contain more than"):
        history_service.replay_engine_history(
            [_row(4, cycle) for cycle in range(1, 5)],
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=4,
            replay_cutoffs=count(1),
            manifest_path="test-manifest.json",
        )
