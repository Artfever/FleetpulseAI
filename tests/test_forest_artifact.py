"""Forest candidate persistence keeps identities, parameters and predictions bound."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.features.dataset_io import save_feature_dataset
from fleetpulse.models.forest_artifact import (
    save_random_forest_artifact,
    verify_random_forest_artifact,
)
from fleetpulse.models.train_forest import train_random_forest


@pytest.fixture
def saved_forest(tmp_path):
    records = [
        {"unit_id": unit, "cycle": c, **dict.fromkeys(READING_COLUMNS, c + unit)}
        for unit, last in ((1, 5), (2, 7), (3, 9), (4, 11))
        for c in range(1, last + 1)
    ]
    split = build_split_manifest(
        records,
        dataset_partition="FD001/train",
        source_path="synthetic/train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    options = {
        "manifest": split,
        "source_sha256": "a" * 64,
        "config": FeatureConfig(channels=("sensor_1",)),
    }
    fitting = build_fitting_dataset(records, **options)
    validation = build_validation_dataset(records, **options)
    result = train_random_forest(
        fitting, validation, n_estimators=8, max_depth=4, min_samples_leaf=1
    )
    validation_path = tmp_path / "validation.json"
    vh = save_feature_dataset(validation, validation_path)
    fh = save_feature_dataset(fitting, tmp_path / "fitting.json")
    paths = save_random_forest_artifact(
        result,
        fitting,
        validation,
        tmp_path,
        fitting_dataset_sha256=fh,
        validation_dataset_sha256=vh,
    )
    return result, fitting, validation, validation_path, paths


def _rewrite(path, document):
    canonical = json.dumps(
        document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    document["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


def test_saved_forest_reloads_in_fresh_process_and_preserves_full_predictions(saved_forest):
    result, _, validation, table_path, paths = saved_forest
    checked = verify_random_forest_artifact(paths["manifest_path"], validation)
    assert checked == {
        "prediction_count": len(validation.rows),
        "mae_cycles": result.metrics.mae,
        "rmse_cycles": result.metrics.rmse,
    }
    output = subprocess.check_output(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/verify_random_forest_artifact.py"),
            str(paths["manifest_path"]),
            str(table_path),
        ],
        text=True,
    )
    assert json.loads(output) == checked


def test_forest_model_tampering_is_rejected(saved_forest):
    _, _, validation, _, paths = saved_forest
    paths["model_path"].write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_random_forest_artifact(paths["manifest_path"], validation)


@pytest.mark.parametrize("kind", ["identity", "prediction", "parameter", "config"])
def test_rehashed_manifest_mismatch_is_rejected(saved_forest, kind):
    _, _, validation, _, paths = saved_forest
    path = paths["manifest_path"]
    document = json.loads(path.read_text())
    payload = document["payload"]
    if kind == "identity":
        payload["evaluation"]["predictions"][0]["unit_id"] += 100
    elif kind == "prediction":
        payload["evaluation"]["predictions"][0]["prediction_cycles"] += 1
    elif kind == "parameter":
        payload["training"]["parameters"]["n_estimators"] += 1
    else:
        payload["training"]["feature_names"].reverse()
    _rewrite(path, document)
    with pytest.raises(ValueError):
        verify_random_forest_artifact(path, validation)


def test_common_metrics_provenance_is_bound_to_the_saved_model(saved_forest):
    _, _, validation, _, paths = saved_forest
    document = json.loads(paths["metrics_path"].read_text())
    document["model"]["version"] = "wrong-model-version"
    paths["metrics_path"].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Common metrics record"):
        verify_random_forest_artifact(paths["manifest_path"], validation)


def test_saver_rejects_misaligned_candidate_before_writing(saved_forest, tmp_path):
    result, fitting, validation, _, _ = saved_forest
    wrong = replace(result, predictions=tuple(reversed(result.predictions)))
    with pytest.raises(ValueError, match="predictions/labels/metrics"):
        save_random_forest_artifact(
            wrong,
            fitting,
            validation,
            tmp_path / "wrong",
            fitting_dataset_sha256="b" * 64,
            validation_dataset_sha256="c" * 64,
        )
    assert not (tmp_path / "wrong").exists()
