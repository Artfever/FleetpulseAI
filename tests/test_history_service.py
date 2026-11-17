"""P16 history replay sends only each observed prefix to saved inference."""

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
