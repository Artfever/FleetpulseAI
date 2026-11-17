"""Verify and align saved constant and Decision Tree validation results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from fleetpulse.evaluation import (
    ValidationMetrics,
    calculate_validation_metrics,
    score_constant_baseline,
    validate_metrics_record,
)
from fleetpulse.evaluation.validation import ValidationPrediction
from fleetpulse.features.build_dataset import FeatureDataset
from fleetpulse.models.constant_baseline import ConstantRULBaseline
from fleetpulse.models.forest_artifact import (
    MANIFEST_FILENAME as FOREST_MANIFEST_FILENAME,
)
from fleetpulse.models.forest_artifact import (
    verify_random_forest_artifact,
)
from fleetpulse.models.mlp_contract import MANIFEST_FILENAME as MLP_MANIFEST_FILENAME
from fleetpulse.models.tree_artifact import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    verify_decision_tree_artifact,
)

BASELINE_ARTIFACT = "constant_rul_validation.json"
BASELINE_METRICS = "metrics.json"


@dataclass(frozen=True)
class ModelComparison:
    """Verified per-engine predictions and aggregate metrics for both models."""

    baseline_name: str
    baseline_prediction: float
    baseline_metrics: ValidationMetrics
    tree_name: str
    tree_metrics: ValidationMetrics
    predictions: tuple[dict[str, Any], ...]
    validation_engine_count: int


@dataclass(frozen=True)
class ForestValidationScores:
    """Verified aggregate scores from the saved forest on this validation split."""

    model_name: str
    model_version: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float


@dataclass(frozen=True)
class MLPValidationScores:
    """Verified saved MLP results aligned to the checked development split."""

    model_name: str
    model_version: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float
    best_epoch: int
    selected_validation_mse_cycles_squared: float
    predictions: tuple[dict[str, Any], ...]


def load_verified_mlp_scores(
    artifact_dir: str | Path,
    fitting: FeatureDataset,
    validation: FeatureDataset,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> MLPValidationScores | None:
    """Verify the optional MLP bundle and return its saved held-out results."""
    manifest_path = Path(artifact_dir) / "mlp" / MLP_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    payload = _read_hashed_payload(manifest_path)
    training = payload["training"]
    _require(
        training["fitting_dataset_sha256"] == fitting_dataset_sha256,
        "MLP fitting table hash mismatch",
    )
    _require(
        training["validation_dataset_sha256"] == validation_dataset_sha256,
        "MLP validation table hash mismatch",
    )
    _require(training["source_sha256"] == fitting.source_sha256 == validation.source_sha256,
             "MLP source hash mismatch")
    _require(
        training["split_manifest_sha256"]
        == fitting.split_manifest_sha256
        == validation.split_manifest_sha256,
        "MLP split hash mismatch",
    )

    # Import only when an MLP bundle is present so PyTorch remains optional.
    import sklearn
    import torch

    from fleetpulse.models.mlp_artifact import verify_mlp_artifact

    _require(
        payload["model"]["torch_version"] == torch.__version__,
        "MLP PyTorch version does not match the runtime",
    )
    _require(
        training["scikit_learn_version"] == sklearn.__version__,
        "MLP scikit-learn version does not match the runtime",
    )

    result = verify_mlp_artifact(
        manifest_path,
        fitting,
        validation,
        fitting_dataset_sha256=fitting_dataset_sha256,
        validation_dataset_sha256=validation_dataset_sha256,
    )
    evaluation = payload["evaluation"]
    saved_rows = evaluation["predictions"]
    _require(evaluation["partition"] == "FD001/train", "MLP partition mismatch")
    _require(evaluation["sample_unit"] == "engine-prefix", "MLP sample unit mismatch")
    _require(
        evaluation["sample_count"] == len(validation.rows) == result["prediction_count"],
        "MLP validation sample count mismatch",
    )
    predictions = []
    values = []
    for saved, row in zip(saved_rows, validation.rows, strict=True):
        identity = (
            row.vector.dataset_partition,
            row.vector.unit_id,
            row.vector.selected_cycle,
            row.true_rul,
        )
        saved_identity = (
            saved["dataset_partition"],
            saved["unit_id"],
            saved["cutoff_cycle"],
            saved["true_rul_cycles"],
        )
        _require(saved_identity == identity, "MLP saved outputs do not align to the validation split")
        prediction = saved["prediction_cycles"]
        values.append(
            ValidationPrediction(
                dataset_partition=identity[0],
                unit_id=identity[1],
                cutoff_cycle=identity[2],
                prediction=prediction,
                true_rul=identity[3],
            )
        )
        predictions.append(
            {
                "Model": f"Dense MLP ({payload['model']['version']})",
                "Engine": identity[1],
                "Cutoff cycle": identity[2],
                "Actual RUL (cycles)": identity[3],
                "Predicted RUL (cycles)": prediction,
            }
        )
    metrics = calculate_validation_metrics(values)
    best_epoch = training["best_epoch"]
    validation_curve = training["validation_mse_cycles_squared"]
    _require(
        type(best_epoch) is int
        and 1 <= best_epoch <= len(validation_curve)
        and validation_curve[best_epoch - 1] == min(validation_curve),
        "MLP selected epoch does not match its saved validation curve",
    )
    return MLPValidationScores(
        model_name=payload["model"]["name"],
        model_version=payload["model"]["version"],
        sample_count=metrics.engine_count,
        mae_cycles=metrics.mae,
        rmse_cycles=metrics.rmse,
        best_epoch=best_epoch,
        selected_validation_mse_cycles_squared=validation_curve[best_epoch - 1],
        predictions=tuple(predictions),
    )


def load_verified_forest_scores(
    artifact_dir: str | Path,
    validation: FeatureDataset,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> ForestValidationScores | None:
    """Verify the optional saved forest and return scores aligned to the P06 split."""
    manifest_path = Path(artifact_dir) / FOREST_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    payload = _read_hashed_payload(manifest_path)
    training = payload["training"]
    _require(
        training["fitting_dataset_sha256"] == fitting_dataset_sha256,
        "forest fitting table hash mismatch",
    )
    _require(
        training["validation_dataset_sha256"] == validation_dataset_sha256,
        "forest validation table hash mismatch",
    )
    result = verify_random_forest_artifact(manifest_path, validation)
    evaluation = payload["evaluation"]
    _require(evaluation["partition"] == "FD001/train", "forest partition mismatch")
    _require(evaluation["sample_unit"] == "engine", "forest sample unit mismatch")
    _require(
        evaluation["sample_count"] == len(validation.rows) == result["prediction_count"],
        "forest validation sample count mismatch",
    )
    return ForestValidationScores(
        model_name=payload["model"]["name"],
        model_version=payload["model"]["version"],
        sample_count=result["prediction_count"],
        mae_cycles=result["mae_cycles"],
        rmse_cycles=result["rmse_cycles"],
    )


def load_model_comparison(
    artifact_dir: str | Path,
    fitting: FeatureDataset,
    validation: FeatureDataset,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> ModelComparison:
    """Verify both saved outputs against the same checked development tables."""
    directory = Path(artifact_dir)
    baseline_payload = _read_hashed_payload(directory / BASELINE_ARTIFACT)
    base_evaluation = baseline_payload["evaluation"]
    baseline = ConstantRULBaseline.from_artifact(baseline_payload["baseline"])
    expected = score_constant_baseline(fitting, validation, statistic=baseline.statistic)
    _require(baseline == expected.baseline, "saved constant baseline differs from fitting labels")
    _require(base_evaluation["dataset_partition"] == "FD001/train", "baseline partition mismatch")
    _require(base_evaluation["sample_count"] == len(validation.rows), "baseline sample count mismatch")
    _require(base_evaluation["fitting_dataset_sha256"] == fitting_dataset_sha256,
             "baseline fitting table hash mismatch")
    _require(base_evaluation["validation_dataset_sha256"] == validation_dataset_sha256,
             "baseline validation table hash mismatch")
    _require(base_evaluation["source_sha256"] == fitting.source_sha256,
             "baseline source hash mismatch")
    _require(base_evaluation["split_manifest_sha256"] == fitting.split_manifest_sha256,
             "baseline split hash mismatch")
    baseline_rows = base_evaluation["predictions"]
    _require(len(baseline_rows) == len(expected.metrics.per_engine), "baseline prediction count mismatch")
    for saved, measured in zip(baseline_rows, expected.metrics.per_engine, strict=True):
        _require(
            saved
            == {
                "dataset_partition": measured.dataset_partition,
                "unit_id": measured.unit_id,
                "cutoff_cycle": measured.cutoff_cycle,
                "prediction_cycles": measured.prediction,
                "true_rul_cycles": measured.true_rul,
                "absolute_error_cycles": measured.absolute_error,
                "squared_error_cycles": measured.squared_error,
            },
            "baseline per-engine output differs from the checked tables",
        )
    _metric_close(base_evaluation["mae_cycles"], expected.metrics.mae, "baseline MAE")
    _metric_close(base_evaluation["rmse_cycles"], expected.metrics.rmse, "baseline RMSE")

    baseline_record = json.loads((directory / BASELINE_METRICS).read_text(encoding="utf-8"))
    validate_metrics_record(baseline_record)
    _check_metrics_record(
        baseline_record,
        model_name="constant-rul-baseline",
        version=baseline.artifact_version,
        count=expected.metrics.engine_count,
        mae=expected.metrics.mae,
        rmse=expected.metrics.rmse,
    )
    _require(
        baseline_record["evaluation"]["input_hashes"].get("fitting_dataset_sha256")
        == fitting_dataset_sha256,
        "baseline metrics fitting table hash mismatch",
    )
    _require(
        baseline_record["evaluation"]["input_hashes"].get("validation_dataset_sha256")
        == validation_dataset_sha256,
        "baseline metrics validation table hash mismatch",
    )

    tree_manifest_path = directory / MANIFEST_FILENAME
    tree_recheck = verify_decision_tree_artifact(tree_manifest_path, validation)
    tree_payload = _read_hashed_payload(tree_manifest_path)
    _require(
        tree_payload["training"]["fitting_dataset_sha256"] == fitting_dataset_sha256,
        "tree fitting table hash mismatch",
    )
    _require(
        tree_payload["training"]["validation_dataset_sha256"] == validation_dataset_sha256,
        "tree validation table hash mismatch",
    )
    tree_eval = tree_payload["evaluation"]
    tree_metrics_record = json.loads((directory / METRICS_FILENAME).read_text(encoding="utf-8"))
    validate_metrics_record(tree_metrics_record)
    _check_metrics_record(
        tree_metrics_record,
        model_name="decision-tree-rul",
        version=tree_payload["model"]["version"],
        count=tree_recheck["prediction_count"],
        mae=tree_recheck["mae_cycles"],
        rmse=tree_recheck["rmse_cycles"],
    )
    _require(
        tree_metrics_record["evaluation"]["input_hashes"].get("fitting_dataset_sha256")
        == fitting_dataset_sha256,
        "tree metrics fitting table hash mismatch",
    )
    _require(
        tree_metrics_record["evaluation"]["input_hashes"].get("validation_dataset_sha256")
        == validation_dataset_sha256,
        "tree metrics validation table hash mismatch",
    )
    _require(tree_eval["sample_count"] == len(validation.rows), "tree sample count mismatch")
    _require(tree_eval["partition"] == "FD001/train", "tree partition mismatch")
    _require(tree_eval["mae_cycles"] == tree_recheck["mae_cycles"], "tree MAE mismatch")
    _require(tree_eval["rmse_cycles"] == tree_recheck["rmse_cycles"], "tree RMSE mismatch")
    tree_rows = tree_eval["predictions"]

    combined = []
    for base_row, tree_row, validation_row in zip(
        baseline_rows, tree_rows, validation.rows, strict=True
    ):
        identity = (
            validation_row.vector.dataset_partition,
            validation_row.vector.unit_id,
            validation_row.vector.selected_cycle,
            validation_row.true_rul,
        )
        base_identity = (
            base_row["dataset_partition"],
            base_row["unit_id"],
            base_row["cutoff_cycle"],
            base_row["true_rul_cycles"],
        )
        tree_identity = (
            tree_row["dataset_partition"],
            tree_row["unit_id"],
            tree_row["cutoff_cycle"],
            tree_row["true_rul_cycles"],
        )
        _require(base_identity == tree_identity == identity, "model outputs do not align by engine/cutoff")
        for model_name, prediction in (
            ("Constant mean baseline", base_row["prediction_cycles"]),
            ("Decision Tree", tree_row["prediction_cycles"]),
        ):
            combined.append(
                {
                    "Model": model_name,
                    "Engine": validation_row.vector.unit_id,
                    "Cutoff cycle": validation_row.vector.selected_cycle,
                    "Actual RUL (cycles)": validation_row.true_rul,
                    "Predicted RUL (cycles)": prediction,
                }
            )
    return ModelComparison(
        baseline_name="Constant mean baseline",
        baseline_prediction=baseline.value_cycles,
        baseline_metrics=expected.metrics,
        tree_name="Decision Tree",
        tree_metrics=tree_recheck_to_metrics(tree_rows, validation),
        predictions=tuple(combined),
        validation_engine_count=len(validation.rows),
    )


def tree_recheck_to_metrics(tree_rows: list[dict[str, Any]], validation: FeatureDataset) -> ValidationMetrics:
    from fleetpulse.evaluation import ValidationPrediction, calculate_validation_metrics

    values = tuple(
        ValidationPrediction(
            dataset_partition=row.vector.dataset_partition,
            unit_id=row.vector.unit_id,
            cutoff_cycle=row.vector.selected_cycle,
            prediction=saved["prediction_cycles"],
            true_rul=row.true_rul,
        )
        for row, saved in zip(validation.rows, tree_rows, strict=True)
    )
    return calculate_validation_metrics(values)


def _read_hashed_payload(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    _require(document["sha256"] == digest, f"content hash mismatch: {path.name}")
    return payload


def _check_metrics_record(record, *, model_name, version, count, mae, rmse) -> None:
    _require(record["model"] == {"name": model_name, "version": version}, "metrics model identity mismatch")
    evaluation = record["evaluation"]
    _require(evaluation["partition"] == "FD001/train", "metrics partition mismatch")
    _require(evaluation["sample_unit"] == "engine", "metrics sample unit mismatch")
    _require(evaluation["sample_count"] == count, "metrics sample count mismatch")
    _metric_close(evaluation["metrics"]["mae_cycles"], mae, "metrics MAE")
    _metric_close(evaluation["metrics"]["rmse_cycles"], rmse, "metrics RMSE")


def _metric_close(actual, expected, name: str) -> None:
    _require(isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"{name} differs from per-engine output")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid model comparison: {message}.")
