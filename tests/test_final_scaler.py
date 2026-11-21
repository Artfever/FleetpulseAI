"""Final MLP scaling uses every training-cycle feature and no holdout rows."""

import json

import numpy as np
import pytest

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_scaler import (
    build_final_training_features,
    fit_final_scaler,
    load_final_scaler_manifest,
    save_final_scaler_manifest,
)


def _histories():
    return [
        {
            "unit_id": unit_id,
            "cycle": cycle,
            **{
                channel: float(unit_id * 10 + cycle * channel_index)
                for channel_index, channel in enumerate(READING_COLUMNS, start=1)
            },
        }
        for unit_id in (1, 2)
        for cycle in (1, 2, 3)
    ]


def test_final_scaler_fits_every_cycle_vector_from_every_training_engine(tmp_path):
    config = FeatureConfig(channels=("sensor_1",), window_cycles=2)
    features = build_final_training_features(_histories(), config=config)

    assert features.matrix.shape == (6, 4)
    assert features.row_keys == tuple(
        ("FD001/train", unit_id, cycle)
        for unit_id in (1, 2)
        for cycle in (1, 2, 3)
    )
    assert features.per_engine_row_counts == ((1, 3), (2, 3))

    scaler = fit_final_scaler(features)
    assert scaler.n_samples_seen_ == 6
    np.testing.assert_allclose(scaler.mean_, features.matrix.mean(axis=0))
    np.testing.assert_allclose(scaler.transform(features.matrix).mean(axis=0), 0.0, atol=1e-12)

    path = tmp_path / "scaler_manifest.json"
    save_final_scaler_manifest(
        scaler,
        features,
        config=config,
        training_source_sha256="a" * 64,
        selection_manifest_sha256="b" * 64,
        path=path,
    )
    loaded, payload = load_final_scaler_manifest(
        path, expected_source_sha256="a" * 64, expected_config=config
    )
    np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
    np.testing.assert_array_equal(loaded.scale_, scaler.scale_)
    np.testing.assert_array_equal(loaded.var_, scaler.var_)
    assert payload["training_engine_count"] == 2
    assert payload["training_source_row_count"] == 6


def test_final_scaler_rejects_changed_manifest_content(tmp_path):
    config = FeatureConfig(channels=("sensor_1",), window_cycles=2)
    features = build_final_training_features(_histories(), config=config)
    scaler = fit_final_scaler(features)
    path = tmp_path / "scaler_manifest.json"
    save_final_scaler_manifest(
        scaler,
        features,
        config=config,
        training_source_sha256="a" * 64,
        selection_manifest_sha256="b" * 64,
        path=path,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["scaler"]["mean"][0] += 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_final_scaler_manifest(path)
