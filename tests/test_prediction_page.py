"""Prediction-page behavior with synthetic history and controlled service results."""

import json
import math
from types import SimpleNamespace

import streamlit as st
from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, services, ui
from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.features import FeatureConfig
from fleetpulse.services import RULPrediction
from fleetpulse.services import history as history_service
from fleetpulse.services.explanations import FeatureContribution, FinalTreeExplanation

TREE_OPTION = "Decision Tree"
FOREST_OPTION = "Random Forest"
XGBOOST_OPTION = "XGBoost"
MLP_OPTION = "Dense MLP (CPU)"


def _page(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    rows = [
        {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, 0.5)}
        for unit in (1, 2)
        for cycle in (1, 2, 3)
    ]
    monkeypatch.setattr(ui, "render_training_data", lambda **kwargs: rows)
    return AppTest.from_file(
        str(PROJECT_ROOT / "app/modules/3_RUL_Prediction.py"), default_timeout=30
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
    assert any("sensor_1 through cycle 2" in item.value for item in page.caption)
    assert any("FD001 is simulated" in item.value for item in page.caption)
    assert page.metric[0].value == "12.50"
    assert any("test-model" in item.value and "test-v1" in item.value for item in page.caption)
    assert any("short" in item.value for item in page.warning)
    trace = page.get("iframe")[0].proto.srcdoc
    data = json.loads(trace.split('id="payload">', 1)[1].split("</script>", 1)[0])
    assert data["points"] == [
        {"engine": 2, "cycle": 1, "value": 12.5},
        {"engine": 2, "cycle": 2, "value": 12.5},
    ]
    page.select_slider[0].set_value(1).run()
    assert page.metric[0].value == "12.50"
    assert chart_calls[-2][0]["cycle"] == [1]
    assert chart_calls[-1][0]["Observed cycle"] == [1]


def test_final_tree_explanation_shows_signed_cycle_contributions(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    history_calls = []

    def predict(history, **kwargs):
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(12.5, 12.5, "dev-tree", "dev-v1", cutoff, len(history), ())

    config = FeatureConfig()
    contributions = tuple(
        FeatureContribution(
            name,
            0.5,
            2.0 if index == 0 else -1.0 if index == 1 else 0.0,
        )
        for index, name in enumerate(config.output_schema)
    )
    explanation = FinalTreeExplanation(
        model_name="decision-tree-rul",
        model_version="decision-tree-rul-v1",
        dataset_partition="FD001/train",
        unit_id=1,
        observed_cycle=3,
        raw_rul_cycles=21.0,
        base_value_cycles=20.0,
        contributions=contributions,
        global_summary=(),
        feature_config_sha256="a" * 64,
        input_history_sha256="b" * 64,
        manifest_payload_sha256="c" * 64,
        weights_sha256="d" * 64,
        training_source_sha256="e" * 64,
        background_sha256="f" * 64,
        background_selection_policy="fd001-train-all-engines-observed-cycle-90-v1",
        background_row_count=100,
        additivity_error_cycles=0.0,
    )

    def explain(history, **kwargs):
        history_calls.append((history, kwargs))
        return explanation

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    monkeypatch.setattr(services, "explain_final_tree_history", explain)
    page.run()
    assert not history_calls
    page.button[0].click().run()

    assert len(history_calls) == 1
    assert [row["cycle"] for row in history_calls[0][0]] == [1, 2, 3]
    assert history_calls[0][1]["dataset_partition"] == "FD001/train"
    assert history_calls[0][1]["unit_id"] == 1
    assert history_calls[0][1]["artifact_root"] == tmp_path
    assert page.metric[-2].label == "Final raw RUL estimate"
    assert page.metric[-2].value == "21.00 cycles"
    assert page.metric[-1].label == "SHAP reference baseline"
    assert page.metric[-1].value == "20.00 cycles"
    positive_rows = page.dataframe[-2].value.to_dict("records")
    negative_rows = page.dataframe[-1].value.to_dict("records")
    assert positive_rows[0]["Contribution (cycles)"] == 2.0
    assert negative_rows[0]["Contribution (cycles)"] == -1.0
    assert not page.exception


def test_missing_final_tree_explanation_fails_gracefully(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)

    def predict(history, **kwargs):
        cutoff = int(history[-1]["cycle"])
        return RULPrediction(12.5, 12.5, "dev-tree", "dev-v1", cutoff, len(history), ())

    def explain(history, **kwargs):
        raise FileNotFoundError("saved model bundle is missing")

    monkeypatch.setattr(history_service, "predict_rul_with_context", predict)
    monkeypatch.setattr(services, "explain_final_tree_history", explain)
    page.run()
    page.button[0].click().run()

    assert not page.exception
    assert page.metric[0].value == "12.50"
    assert any(
        "Final Decision Tree explanation is unavailable" in item.value
        and "saved model bundle is missing" in item.value
        for item in page.info
    )


def test_frozen_model_comparison_shows_raw_spread_and_lists_failures_separately(
    monkeypatch, tmp_path
):
    page = _page(monkeypatch, tmp_path)
    comparison_calls = []
    comparison = SimpleNamespace(
        dataset_partition="FD001/train",
        unit_id=1,
        observed_cycle=3,
        predictions=(
            SimpleNamespace(
                model_name="decision-tree-rul",
                model_version="decision-tree-rul-v1",
                raw_rul_cycles=-2.5,
            ),
            SimpleNamespace(
                model_name="random-forest-rul",
                model_version="random-forest-rul-v1",
                raw_rul_cycles=8.0,
            ),
        ),
        failures=(
            SimpleNamespace(
                model_name="xgboost-rul",
                code="artifact_unavailable",
                reason="synthetic missing bundle",
            ),
        ),
        raw_min_rul_cycles=-2.5,
        raw_max_rul_cycles=8.0,
        raw_spread_cycles=10.5,
    )

    def compare(history, **kwargs):
        comparison_calls.append((history, kwargs))
        return comparison

    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda history, **kwargs: RULPrediction(
            12.5,
            12.5,
            "dev-tree",
            "dev-v1",
            int(history[-1]["cycle"]),
            len(history),
            (),
        ),
    )
    monkeypatch.setattr(services, "compare_final_models", compare)
    page.run()
    page.select_slider[0].set_value(2).run()
    page.button[-1].click().run()

    assert len(comparison_calls) == 1
    assert [row["cycle"] for row in comparison_calls[0][0]] == [1, 2]
    assert comparison_calls[0][1] == {
        "dataset_partition": "FD001/train",
        "unit_id": 1,
        "artifact_root": tmp_path,
    }
    comparison_rows = page.dataframe[-2].value.to_dict("records")
    failure_rows = page.dataframe[-1].value.to_dict("records")
    assert comparison_rows == [
        {
            "Model": "decision-tree-rul",
            "Version": "decision-tree-rul-v1",
            "Raw estimate (cycles)": -2.5,
        },
        {
            "Model": "random-forest-rul",
            "Version": "random-forest-rul-v1",
            "Raw estimate (cycles)": 8.0,
        },
    ]
    assert failure_rows == [
        {
            "Model": "xgboost-rul",
            "Status": "artifact_unavailable",
            "Reason": "synthetic missing bundle",
        }
    ]
    assert page.metric[-1].label == "Raw model disagreement (max − min)"
    assert page.metric[-1].value == "10.50 cycles"
    assert page.metric[-1].label == "Raw model disagreement (max − min)"
    assert not page.exception


def test_frozen_model_comparison_reports_no_spread_when_all_models_fail(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    result = SimpleNamespace(
        dataset_partition="FD001/train",
        unit_id=1,
        observed_cycle=3,
        predictions=(),
        failures=(
            SimpleNamespace(
                model_name="decision-tree-rul",
                code="artifact_unavailable",
                reason="manifest missing",
            ),
        ),
        raw_min_rul_cycles=None,
        raw_max_rul_cycles=None,
        raw_spread_cycles=None,
    )
    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda history, **kwargs: RULPrediction(
            12.5,
            12.5,
            "dev-tree",
            "dev-v1",
            int(history[-1]["cycle"]),
            len(history),
            (),
        ),
    )
    monkeypatch.setattr(services, "compare_final_models", lambda *args, **kwargs: result)
    page.run()
    page.button[-1].click().run()

    assert any("no spread is available" in item.value for item in page.info)
    assert not any("Raw model disagreement" in item.label for item in page.metric)
    assert page.dataframe[-1].value.iloc[0]["Reason"] == "manifest missing"
    assert not page.exception


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
    monkeypatch.setattr(
        history_service,
        "predict_rul_with_context",
        lambda history, **kwargs: RULPrediction(
            20.0,
            20.0,
            "decision-tree-rul",
            "decision-tree-rul-v1",
            int(history[-1]["cycle"]),
            len(history),
            (),
        ),
    )

    page.run()
    assert any("In-sample estimate" in item.value for item in page.warning)
    assert any("decision-tree-rul-v1" in item.value for item in page.caption)
    assert any("Engine fit membership: Seen during fitting" in item.value for item in page.caption)

    page.selectbox[2].select(2).run()
    assert any("Development holdout" in item.value for item in page.caption)
    assert any(
        "Engine fit membership: Not seen during fitting (development validation engine)" in item.value
        for item in page.caption
    )


def test_missing_bundle_shows_error_without_fabricating_prediction(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path).run()
    assert not any(option.startswith("XGBoost ·") for option in page.selectbox[1].options)
    assert not any(option.startswith("Dense MLP (CPU) ·") for option in page.selectbox[1].options)
    assert not any(option.startswith("XGBoost") for option in page.selectbox[1].options)
    assert not any(option.startswith("Dense MLP") for option in page.selectbox[1].options)
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
    assert any("Synthetic demo only" in item.value for item in page.warning)
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
    assert any("Packaged FD001 parity sample" in item.value for item in page.caption)
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
    assert list(page.selectbox[1].options) == [TREE_OPTION]
    assert not any(option.startswith("Random Forest") for option in page.selectbox[1].options)
    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "random_forest_validation.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "random_forest_model.joblib").write_bytes(b"test fixture")
    page.run()
    assert list(page.selectbox[1].options) == [TREE_OPTION, FOREST_OPTION]
    assert any(option.startswith("Random Forest") for option in page.selectbox[1].options)


def test_model_selector_keeps_model_names_short(monkeypatch, tmp_path):
    page = _page(monkeypatch, tmp_path)
    page.run()

    assert not page.exception
    assert list(page.selectbox[1].options) == ["Decision Tree"]


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
    page.selectbox[1].select(FOREST_OPTION).run()
    assert page.selectbox[2].value == 2
    assert page.select_slider[0].value == 2
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == artifact_dir / "random_forest_validation.json"
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "18.80"
    assert any("random-forest-rul-v1" in item.value for item in page.caption)

    page.selectbox[1].select(TREE_OPTION).run()
    assert page.selectbox[2].value == 2
    assert page.select_slider[0].value == 2
    assert not any("P06" in item.value for item in page.caption)
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == artifact_dir / "decision_tree_validation.json"
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "12.50"
    assert any("decision-tree-rul-v1" in item.value for item in page.caption)


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
    assert list(page.selectbox[1].options) == [TREE_OPTION, XGBOOST_OPTION]
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    page.selectbox[1].select(XGBOOST_OPTION).run()
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == manifest
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert {row["unit_id"] for row in calls[-1][0]} == {2}
    assert page.metric[0].value == "28.69"
    assert any("xgboost-rul-v1" in item.value for item in page.caption)


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
    assert MLP_OPTION in page.selectbox[1].options
    assert not trainer_calls

    page.selectbox[1].select(MLP_OPTION).run()
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(3).run()

    assert not page.exception
    assert not trainer_calls
    assert page.metric[0].label == "Estimated remaining cycles"
    assert math.isfinite(float(page.metric[0].value))
    assert any("dense-rul-mlp-v1" in item.value for item in page.caption)


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
    assert MLP_OPTION in page.selectbox[1].options
    page.selectbox[2].select(2).run()
    page.select_slider[0].set_value(2).run()
    page.selectbox[1].select(MLP_OPTION).run()
    assert not page.exception
    assert calls[-1][1]["manifest_path"] == manifest
    assert [row["cycle"] for row in calls[-1][0]] == [1, 2]
    assert page.metric[0].value == "9.25"
    assert any("dense-rul-mlp-v1" in item.value for item in page.caption)
