"""The feature-preview service is UI-free and keeps future rows out of X."""

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.features import FeatureConfig
from fleetpulse.services.feature_preview import build_engine_feature_preview


def _row(unit_id, cycle, sensor_1):
    return {
        "unit_id": unit_id,
        "cycle": cycle,
        **dict.fromkeys(READING_COLUMNS, 0.0),
        "sensor_1": sensor_1,
    }


def test_feature_preview_returns_selected_prefix_and_ordered_features():
    history = [
        _row(4, 1, 2.0),
        _row(4, 2, 4.0),
        _row(4, 3, 1e100),
    ]
    preview = build_engine_feature_preview(
        history,
        dataset_partition="FD001/train",
        unit_id=4,
        selected_cycle=2,
        config=FeatureConfig(channels=("sensor_1",), window_cycles=30),
    )

    assert [row["cycle"] for row in preview.visible_history] == [1, 2]
    assert all(row["unit_id"] == 4 for row in preview.visible_history)
    assert preview.vector.engine_key == ("FD001/train", 4)
    assert preview.vector.selected_cycle == 2
    assert preview.vector.config.output_schema == (
        "sensor_1_latest",
        "sensor_1_mean",
        "sensor_1_std",
        "sensor_1_slope",
    )
    assert preview.vector.values == (4.0, 3.0, 1.0, 2.0)
