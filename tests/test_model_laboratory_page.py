"""The Model Laboratory shows only verified, split-matched forest scores."""

import hashlib
import json
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.evaluation.model_comparison import ForestValidationScores, MLPValidationScores


def _prepare_model_laboratory(monkeypatch, tmp_path, *, forest_scores, mlp_scores=None):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    fitting_rows = [{"dataset_partition": "FD001/train", "unit_id": 1}]
    validation_rows = [
        {"dataset_partition": "FD001/train", "unit_id": 2, "target_rul": 7}
    ]
    manifest = {
        "source": {"sha256": "a" * 64},
        "split": {"seed": 42},
        "cutoff_policy": {"seed": 42},
        "fitting_engines": fitting_rows,
        "validation_prefixes": validation_rows,
    }
    monkeypatch.setattr(
        __import__("fleetpulse.data.split_audit_view", fromlist=["verified_split_manifest"]),
        "verified_split_manifest",
        lambda *_args: manifest,
    )
    monkeypatch.setattr(
        __import__("fleetpulse.features.dataset_io", fromlist=["load_feature_dataset"]),
        "load_feature_dataset",
        lambda path, **_kwargs: path.name,
    )
    comparison = SimpleNamespace(
        baseline_name="Constant mean baseline",
        baseline_prediction=30.0,
        baseline_metrics=SimpleNamespace(mae=10.0, rmse=12.0),
        tree_name="Decision Tree",
        tree_metrics=SimpleNamespace(mae=8.0, rmse=9.0),
        validation_engine_count=1,
        predictions=(
            {
                "Model": "Constant mean baseline",
                "Engine": 2,
                "Cutoff cycle": 3,
                "Actual RUL (cycles)": 7.0,
                "Predicted RUL (cycles)": 30.0,
            },
            {
                "Model": "Decision Tree",
                "Engine": 2,
                "Cutoff cycle": 3,
                "Actual RUL (cycles)": 7.0,
                "Predicted RUL (cycles)": 15.0,
            },
        ),
    )
    comparison_module = __import__(
        "fleetpulse.evaluation.model_comparison", fromlist=["load_model_comparison"]
    )
    monkeypatch.setattr(comparison_module, "load_model_comparison", lambda *_a, **_k: comparison)
    monkeypatch.setattr(
        comparison_module, "load_verified_forest_scores", lambda *_a, **_k: forest_scores
    )
    monkeypatch.setattr(
        comparison_module, "load_verified_mlp_scores", lambda *_a, **_k: mlp_scores
    )

    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    for name in ("fitting_features.json", "validation_features.json"):
        (artifact_dir / name).write_text(json.dumps({"sha256": "a" * 64}), encoding="utf-8")
    selection_candidates = [
        {
            "name": f"Candidate {index}",
            "configuration": {"depth": index},
            "mae_cycles": float(index),
            "rmse_cycles": float(10 - index),
            "sample_count": 20,
            "training_source_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
            "result_artifact_sha256": str(index) * 64,
        }
        for index in range(1, 7)
    ]
    selection_payload = {
        "schema_version": "p11-selection-manifest-v1",
        "dataset_partition": "FD001/train",
        "validation_sample_unit": "engine-prefix",
        "validation_engine_count": 20,
        "training_source_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "official_test_data_opened": False,
        "candidates": selection_candidates,
        "decision": {
            "status": "team_choice_pending_p11_t3",
            "lowest_mae_candidate": "Candidate 1",
            "lowest_rmse_candidate": "Candidate 6",
        },
    }
    encoded = json.dumps(selection_payload, sort_keys=True, separators=(",", ":"))
    selection_path = tmp_path / "artifacts/selection_manifest.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "payload": selection_payload,
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_model_laboratory_displays_verified_forest_validation_scores(monkeypatch, tmp_path):
    _prepare_model_laboratory(
        monkeypatch,
        tmp_path,
        forest_scores=ForestValidationScores(
            "random-forest-rul", "random-forest-rul-v1", 1, 4.25, 6.5
        ),
    )

    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/5_Model_Laboratory.py"), default_timeout=30
    ).run()

    assert not page.exception
    assert any(item.value == "P11 candidate selection comparison" for item in page.subheader)
    selection_tables = [
        item.value
        for item in page.dataframe
        if hasattr(item.value, "columns") and "Selection MAE (cycles)" in item.value.columns
    ]
    assert len(selection_tables) == 1
    assert len(selection_tables[0]) == 6
    assert any("team choice remains for P11-T3" in item.value for item in page.markdown)
    assert any(
        "Constant baseline, Decision Tree, and Random Forest" == item.value
        for item in page.subheader
    )
    score_tables = [
        item.value
        for item in page.dataframe
        if hasattr(item.value, "columns") and "MAE (cycles)" in item.value.columns
    ]
    assert len(score_tables) == 1
    forest_row = score_tables[0].loc[score_tables[0]["Model"].str.startswith("Random Forest")]
    assert forest_row.iloc[0]["MAE (cycles)"] == 4.25
    assert forest_row.iloc[0]["RMSE (cycles)"] == 6.5
    assert any("1 held-out development engines" in item.value for item in page.markdown)


def test_model_laboratory_says_forest_scores_are_unavailable_when_bundle_missing(
    monkeypatch, tmp_path
):
    _prepare_model_laboratory(monkeypatch, tmp_path, forest_scores=None)

    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/5_Model_Laboratory.py"), default_timeout=30
    ).run()

    assert not page.exception
    assert any("Verified Random Forest scores are unavailable" in item.value for item in page.info)
    assert any(
        "Constant baseline and Decision Tree" == item.value for item in page.subheader
    )


def test_model_laboratory_displays_verified_mlp_results_and_checkpoint_selection(
    monkeypatch, tmp_path
):
    scores = MLPValidationScores(
        model_name="dense-rul-mlp",
        model_version="dense-rul-mlp-v1",
        sample_count=1,
        mae_cycles=22.5,
        rmse_cycles=28.0,
        best_epoch=2,
        selected_validation_mse_cycles_squared=526.15,
        fitting_row_count=16,
        training_mse_cycles_squared=(900.0, 700.0, 500.0),
        validation_mse_cycles_squared=(800.0, 526.15, 600.0),
        predictions=(
            {
                "Model": "Dense MLP (dense-rul-mlp-v1)",
                "Engine": 2,
                "Cutoff cycle": 3,
                "Actual RUL (cycles)": 7.0,
                "Predicted RUL (cycles)": 10.0,
            },
        ),
    )
    _prepare_model_laboratory(monkeypatch, tmp_path, forest_scores=None, mlp_scores=scores)

    page = AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/5_Model_Laboratory.py"), default_timeout=30
    ).run()

    assert not page.exception
    assert any("selected at epoch 2" in item.value for item in page.markdown)
    assert any("not an untouched performance estimate" in item.value for item in page.markdown)
    chart_specs = [json.loads(item.proto.spec) for item in page.get("plotly_chart")]
    learning_curve = next(
        chart for chart in chart_specs if chart["layout"]["title"]["text"] == "Dense MLP saved learning curves"
    )
    assert [trace["name"] for trace in learning_curve["data"]] == [
        "Training MSE — fitting rows (16)",
        "Validation MSE — engine prefixes (1)",
    ]
    assert learning_curve["layout"]["xaxis"]["title"]["text"] == "Training epoch"
    assert learning_curve["layout"]["yaxis"]["title"]["text"] == "Mean squared error (cycles²)"
    assert any("different sample units" in item.value for item in page.caption)
    tables = [
        item.value
        for item in page.dataframe
        if hasattr(item.value, "columns") and "MAE (cycles)" in item.value.columns
    ]
    assert len(tables) == 1
    mlp_row = tables[0].loc[tables[0]["Model"].str.startswith("Dense MLP")]
    assert mlp_row.iloc[0]["MAE (cycles)"] == 22.5
    assert mlp_row.iloc[0]["RMSE (cycles)"] == 28.0
