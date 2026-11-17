"""Prediction-page behavior with synthetic history and controlled service results."""

import json
import math

import streamlit as st
from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.services import RULPrediction
from fleetpulse.services import history as history_service


def _page(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    rows = [
        {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, 0.5)}
        for unit in (1, 2)
        for cycle in (1, 2, 3)
    ]
    monkeypatch.setattr(ui, "render_training_data", lambda: rows)
    return AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/3_RUL_Prediction.py"), default_timeout=30
    )


def test_selected_engine_and_cutoff_reach_service_without_future_or_target(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(
            12.5, 12.5, "test-model", "test-v1", cutoff, len(history), ("short_window",)
        )

    chart_calls = []
    original_line_chart = st.line_chart

    def capture_chart(data, *args, **kwargs):
        chart_calls.append((data, kwargs))
        return original_line_chart(data, *args, **kwargs)

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    monkeypatch.setattr(st, "line_chart", capture_chart)
    page.run()
    assert len(calls) == 3
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    assert not page.exception
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert all(
        row["unit_id"] == 2 and set(row) == {"unit_id", "cycle", *READING_COLUMNS}
        for row in calls[-1][0]
    )
    assert calls[-1][1]["dataset_partition"] == "FD001/train"
    assert calls[-1][1]["unit_id"] == 2
    assert (
        calls[-1][1]["manifest_path"]
        == tmp_path / "artifacts/development/decision_tree_validation.json"
    )
    assert chart_calls[-2][0]["cycle"] == [1, 2]
    assert len(chart_calls[-2][0]["sensor_1"]) == 2
    assert chart_calls[-1][0]["Observed cycle"] == [1, 2]
    assert chart_calls[-1][0]["Estimated remaining cycles"] == [12.5, 12.5]
    assert len(page.dataframe[0].value) == 2
    assert any("chart ends at cycle 2" in item.value for item in page.caption)
    assert page.metric[0].value == "12.50"
    assert any("test-model" in item.value and "test-v1" in item.value for item in page.markdown)
    assert any("short" in item.value for item in page.warning)
    page.select_slider[0].set_value(1).run()
    assert page.metric[0].value == "12.50"
    assert chart_calls[-2][0]["cycle"] == [1]
    assert chart_calls[-1][0]["Observed cycle"] == [1]


def test_training_replay_is_labeled_by_saved_model_fit_membership(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    split_path = tmp_path / "artifacts/split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        json.dumps(
            {
                "fitting_engines": [{"dataset_partition": "FD001/train", "unit_id": 1}],
                "validation_prefixes": [
                    {"dataset_partition": "FD001/train", "unit_id": 2, "target_rul": 99}
                ],
            }
        ),
        encoding="utf-8",
    )

    page.run()
    assert any("In-sample illustration" in item.value for item in page.warning)

    page.selectbox[2].select(2).run()
    assert any("Held-out development engine" in item.value for item in page.info)
    assert any("does not load a final refitted model" in item.value for item in page.caption)


def test_missing_bundle_shows_error_without_fabricating_prediction(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path).run()
    assert "XGBoost" not in page.selectbox[1].options
    assert "Dense MLP (CPU)" not in page.selectbox[1].options
    assert any("XGBoost is unavailable" in item.value for item in page.info)
    assert any("Dense MLP is unavailable" in item.value for item in page.info)
    assert not page.exception
    assert not page.metric
    assert any("Prediction is unavailable" in item.value for item in page.warning)


def test_packaged_illustrative_history_is_distinct_and_has_no_future_rows(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    demo_dir = tmp_path / "data/demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "illustrative_history.txt").write_bytes(
        (PROJECT_ROOT / "data/demo/illustrative_history.txt").read_bytes()
    )
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(3.0, 3.0, "test-model", "test-v1", cutoff, len(history), ())

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    page.run()
    page.selectbox[0].select("Illustrative synthetic example").run()
    assert any("not NASA data" in item.value for item in page.warning)
    page.select_slider[0].set_value(2).run()
    assert not page.exception
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert calls[-1][1]["dataset_partition"] == "illustrative/synthetic"
    assert calls[-1][1]["unit_id"] == 900001
    assert page.metric[0].value == "3.00"


def test_packaged_fd001_subset_uses_only_selected_training_prefix(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    rows = [
        {"unit_id": engine, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, 0.5)}
        for engine, cutoff in ((4, 164), (5, 58), (12, 7))
        for cycle in range(1, cutoff + 1)
    ]
    subset_path = tmp_path / "data/demo/fd001_training_prefixes.txt"
    subset_path.parent.mkdir(parents=True)
    subset_path.write_text(
        "\n".join(" ".join(str(row[column]) for column in HISTORY_COLUMNS) for row in rows) + "\n",
        encoding="utf-8",
    )
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(
            34.19607843137255,
            34.20,
            "decision-tree-rul",
            "decision-tree-rul-v1",
            cutoff,
            len(history),
            (),
        )

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    page.run()
    page.selectbox[0].select("Packaged FD001/train parity subset").run()
    assert any("not a complete dataset" in item.value for item in page.info)
    calls_before_slider_change = len(calls)
    page.select_slider[0].set_value(164).run()

    assert not page.exception
    assert len(calls[-1][0]) == 164
    assert len(calls) - calls_before_slider_change <= 10
    assert {row["unit_id"] for row in calls[-1][0]} == {4}
    assert max(row["cycle"] for row in calls[-1][0]) == 164
    assert all(set(row) == {"unit_id", "cycle", *READING_COLUMNS} for row in calls[-1][0])
    assert calls[-1][1]["dataset_partition"] == "FD001/train"
    assert calls[-1][1]["unit_id"] == 4
    assert page.metric[0].value == "34.20"


def test_forest_is_selectable_only_when_its_complete_bundle_is_present(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    page.run()
    assert list(page.selectbox[1].options) == ["Decision Tree"]
    assert any("Random Forest is unavailable" in item.value for item in page.info)

    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "random_forest_validation.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "random_forest_model.joblib").write_bytes(b"test fixture")
    page.run()
    assert list(page.selectbox[1].options) == ["Decision Tree", "Random Forest"]
    assert not any("Random Forest is unavailable" in item.value for item in page.info)


def test_switching_models_keeps_engine_and_cycle_and_uses_matching_manifest(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "random_forest_validation.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "random_forest_model.joblib").write_bytes(b"test fixture")
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        if kwargs["manifest_path"].name == "random_forest_validation.json":
            return RULPrediction(
                18.8, 18.8, "random-forest-rul", "random-forest-rul-v1", cutoff, len(history), ()
            )
        return RULPrediction(
            12.5, 12.5, "decision-tree-rul", "decision-tree-rul-v1", cutoff, len(history), ()
        )

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    page.run()
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    page.selectbox[1].select("Random Forest").run()
    assert page.selectbox[2].value == 2
    assert page.select_slider[0].value == 2
    assert any("available saved development model" in item.value for item in page.caption)
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == artifact_dir / "random_forest_validation.json"
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "18.80"
    assert any("random-forest-rul-v1" in item.value for item in page.markdown)

    page.selectbox[1].select("Decision Tree").run()
    assert page.selectbox[2].value == 2
    assert page.select_slider[0].value == 2
    assert not any("P06" in item.value for item in page.caption)
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == artifact_dir / "decision_tree_validation.json"
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "12.50"
    assert any("decision-tree-rul-v1" in item.value for item in page.markdown)


def test_xgboost_is_available_only_with_bundle_and_uses_registered_manifest(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    manifest = artifact_dir / "xgboost_validation.json"
    model = artifact_dir / "xgboost_model.joblib"
    manifest.write_text("{}", encoding="utf-8")
    model.write_bytes(b"test fixture")
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(
            28.6938, 28.69, "xgboost-rul", "xgboost-rul-v1", cutoff, len(history), ()
        )

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    page.run()
    assert list(page.selectbox[1].options) == ["Decision Tree", "XGBoost"]
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    page.selectbox[1].select("XGBoost").run()
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == manifest
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "28.69"
    assert any("xgboost-rul-v1" in item.value for item in page.markdown)


def test_app_loads_saved_mlp_and_predicts_without_starting_training(monkeypatch, tmp_path):
    from fleetpulse.data.split_manifest import build_split_manifest
    from fleetpulse.features import FeatureConfig
    from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
    from fleetpulse.models import mlp
    from fleetpulse.models.mlp_artifact import save_mlp_artifact

    rows = [
        {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle + unit)}
        for unit, lifetime in ((1, 5), (2, 7), (3, 9), (4, 11))
        for cycle in range(1, lifetime + 1)
    ]
    split = build_split_manifest(
        rows,
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
    fitting = build_fitting_dataset(rows, **options)
    validation = build_validation_dataset(rows, **options)
    training_result = mlp.train_mlp(
        fitting, validation, max_epochs=1, patience=1, batch_size=8, random_seed=42
    )
    artifact_dir = tmp_path / "artifacts/development/mlp"
    artifact_dir.mkdir(parents=True)
    save_mlp_artifact(
        training_result,
        fitting,
        validation,
        artifact_dir,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )

    trainer_calls = []

    def unexpected_training(*_args, **_kwargs):
        trainer_calls.append(True)
        raise AssertionError("app runtime must not call the MLP training loop")

    monkeypatch.setattr(mlp, "train_mlp", unexpected_training)
    page = _page(monkeypatch, tmp_path)

    page.run()
    assert not page.exception
    assert "Dense MLP (CPU)" in page.selectbox[1].options
    assert not trainer_calls

    page.selectbox[1].select("Dense MLP (CPU)").run()
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(3).run()

    assert not page.exception
    assert not trainer_calls
    assert page.metric[0].label == "Estimated remaining cycles"
    assert math.isfinite(float(page.metric[0].value))
    assert any("dense-rul-mlp-v1" in item.value for item in page.markdown)


def test_mlp_is_selectable_only_with_bundle_and_uses_its_saved_manifest(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    artifact_dir = tmp_path / "artifacts/development/mlp"
    artifact_dir.mkdir(parents=True)
    manifest = artifact_dir / "mlp_training_manifest.json"
    weights = artifact_dir / "mlp_best_state.pt"
    manifest.write_text("test manifest", encoding="utf-8")
    weights.write_bytes(b"test weights")
    calls = []

    def predict(history, **kwargs):
        calls.append((history, kwargs))
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(
            9.25, 9.25, "dense-rul-mlp", "dense-rul-mlp-v1", cutoff, len(history), ()
        )

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    page.run()
    assert "Dense MLP (CPU)" in page.selectbox[1].options
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    page.selectbox[1].select("Dense MLP (CPU)").run()
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == manifest
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert page.metric[0].value == "9.25"
    assert any("dense-rul-mlp-v1" in item.value for item in page.markdown)
