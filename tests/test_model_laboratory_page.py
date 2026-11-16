"""The Model Laboratory shows only verified, split-matched forest scores."""

import json
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT, ui
from fleetpulse.evaluation.model_comparison import ForestValidationScores


def _prepare_model_laboratory(monkeypatch, tmp_path, *, forest_scores):
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

    artifact_dir = tmp_path / "artifacts/development"
    artifact_dir.mkdir(parents=True)
    for name in ("fitting_features.json", "validation_features.json"):
        (artifact_dir / name).write_text(json.dumps({"sha256": "a" * 64}), encoding="utf-8")


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
