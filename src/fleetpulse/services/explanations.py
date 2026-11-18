"""Local SHAP explanations for the frozen P12 Decision Tree (P17-S1).

The explanation contract is proposed for the P17 consumers: explain one
validated FD001 history using the exact feature vector and configuration from
the frozen Decision Tree bundle. Values and the baseline are in raw RUL cycles.
The result is local model attribution, not a causal explanation or confidence
estimate. No labels, future rows, fitting, or display clipping are used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.quality import validate_engine_history
from fleetpulse.features import FeatureConfig, engineer_features
from fleetpulse.models.final_tree_bundle import FinalTreeBundle, load_final_tree_bundle
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services.final_comparison import FinalHistoryComparison
from fleetpulse.services.predictor import PredictionInputError

BACKGROUND_SELECTION_POLICY = "fd001-train-all-engines-observed-cycle-90-v1"
BACKGROUND_CYCLE = 90
DECISION_TREE_KEY = "decision-tree-rul"
ADDITIVITY_ABSOLUTE_TOLERANCE = 2e-5
ADDITIVITY_RELATIVE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class TrainingFeatureBackground:
    """Feature-only reference sample with explicit source and selection lineage."""

    values: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    training_source_sha256: str
    selection_policy: str
    source_path: str
    background_sha256: str


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed contribution to the raw Decision Tree output."""

    feature_name: str
    feature_value: float
    contribution_cycles: float


@dataclass(frozen=True)
class GlobalFeatureImportance:
    """One bounded mean absolute local attribution over reference rows."""

    feature_name: str
    mean_absolute_contribution_cycles: float


@dataclass(frozen=True)
class FinalTreeExplanation:
    """Reproducible local explanation bound to a model, row and background."""

    model_name: str
    model_version: str
    dataset_partition: str
    unit_id: int
    observed_cycle: int
    raw_rul_cycles: float
    base_value_cycles: float
    contributions: tuple[FeatureContribution, ...]
    global_summary: tuple[GlobalFeatureImportance, ...]
    feature_config_sha256: str
    input_history_sha256: str
    manifest_payload_sha256: str
    weights_sha256: str
    training_source_sha256: str
    background_sha256: str
    background_selection_policy: str
    background_row_count: int
    additivity_error_cycles: float

    def top_contributions(self, limit: int = 10) -> tuple[FeatureContribution, ...]:
        """Return a bounded, stable ranking by absolute contribution size."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        order = {item.feature_name: index for index, item in enumerate(self.contributions)}
        return tuple(
            sorted(
                self.contributions,
                key=lambda item: (-abs(item.contribution_cycles), order[item.feature_name]),
            )[:limit]
        )


@dataclass(frozen=True)
class ExplanationConsistency:
    """Machine-readable agreement result for one comparison/explanation pair."""

    is_consistent: bool
    mismatch_codes: tuple[str, ...]
    model_name: str
    model_version: str
    dataset_partition: str
    unit_id: int
    observed_cycle: int
    raw_rul_cycles: float
    input_history_sha256: str


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fd001_training_background(
    training_path: str | Path = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt",
    *,
    expected_training_source_sha256: str,
    feature_config: FeatureConfig,
) -> TrainingFeatureBackground:
    """Build deterministic feature rows from train histories at fixed cutoffs.

    Include one row per training engine at observed cycle 90. Every row is
    constructed from the prefix through that fixed cutoff. The selector is
    fixed before looking at model output and reads only the raw 26-column train
    file; it never reads RUL labels or official-test data.
    """
    path = Path(training_path)
    if not isinstance(feature_config, FeatureConfig):
        raise TypeError("feature_config must be a FeatureConfig.")
    if not _is_sha256(expected_training_source_sha256):
        raise ValueError("expected_training_source_sha256 must be a lowercase SHA-256 digest.")
    if not path.is_file():
        raise FileNotFoundError(f"FD001 training histories are unavailable: {path}")
    source_sha = _file_sha256(path)
    if source_sha != expected_training_source_sha256:
        raise ValueError("FD001 training history hash differs from the frozen model source.")

    records = load_engine_history(path)
    by_unit: dict[int, list[dict[str, int | float]]] = {}
    for record in records:
        by_unit.setdefault(int(record["unit_id"]), []).append(record)
    if set(by_unit) != set(range(1, 101)):
        raise ValueError("FD001 background source must contain exactly training engines 1–100.")

    values: list[tuple[float, ...]] = []
    for unit_id in range(1, 101):
        engine_rows = by_unit[unit_id]
        cycles = {int(row["cycle"]) for row in engine_rows}
        if BACKGROUND_CYCLE not in cycles:
            raise ValueError(
                f"FD001 training engine {unit_id} has no observed cycle {BACKGROUND_CYCLE}."
            )
        prefix = [row for row in engine_rows if int(row["cycle"]) <= BACKGROUND_CYCLE]
        vector = engineer_features(
            prefix,
            dataset_partition="FD001/train",
            unit_id=unit_id,
            selected_cycle=BACKGROUND_CYCLE,
            config=feature_config,
        )
        values.append(vector.values)
    if len(values) != 100:
        raise ValueError(f"Expected one reference row per training engine, got {len(values)}.")
    background_digest = _canonical_sha256(
        {
            "values": values,
            "feature_names": feature_config.output_schema,
            "training_source_sha256": source_sha,
            "selection_policy": BACKGROUND_SELECTION_POLICY,
        }
    )
    return TrainingFeatureBackground(
        values=tuple(values),
        feature_names=feature_config.output_schema,
        training_source_sha256=source_sha,
        selection_policy=BACKGROUND_SELECTION_POLICY,
        source_path=path.as_posix(),
        background_sha256=background_digest,
    )


def explain_final_tree_prediction(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    background: TrainingFeatureBackground,
    artifact_root: str | Path = PROJECT_ROOT,
) -> FinalTreeExplanation:
    """Explain one validated observed prefix with its frozen exact feature row.

    The final observed row is the cutoff. The background must use the frozen
    model's same original training source, feature order and documented policy.
    ``TreeExplainer`` uses interventional SHAP values in raw RUL cycles.
    """
    try:
        rows = tuple(history)
    except TypeError as error:
        raise PredictionInputError("history must be an iterable of raw engine records.") from error
    report = validate_engine_history(rows)
    if not report.is_valid:
        codes = ", ".join(sorted({finding.code for finding in report.errors}))
        raise PredictionInputError(f"Invalid observed engine history: {codes}.")
    if {int(row["unit_id"]) for row in rows} != {unit_id}:
        raise PredictionInputError("History must contain only the requested engine's unit_id.")
    normalized_rows = tuple(
        {**row, "unit_id": int(row["unit_id"]), "cycle": int(row["cycle"])} for row in rows
    )

    if not isinstance(dataset_partition, str) or not dataset_partition.strip():
        raise PredictionInputError("dataset_partition must be a nonempty string.")

    spec = FINAL_MODEL_REGISTRY[DECISION_TREE_KEY]
    manifest_path = Path(artifact_root) / spec.manifest_path
    bundle = load_final_tree_bundle(manifest_path)
    _validate_tree_bundle(bundle, spec)
    if not isinstance(background, TrainingFeatureBackground):
        raise TypeError("background must be a TrainingFeatureBackground.")
    training_source_sha = bundle.payload["training"]["training_source_sha256"]
    _validate_background(background, bundle, training_source_sha)

    try:
        vector = engineer_features(
            normalized_rows,
            dataset_partition=dataset_partition,
            unit_id=unit_id,
            config=bundle.feature_config,
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise PredictionInputError(f"Cannot form finite saved-schema features: {error}") from error
    row = np.asarray([vector.values], dtype=np.float64)
    background_matrix = np.asarray(background.values, dtype=np.float64)
    prediction = float(bundle.model.predict(row)[0])
    if not math.isfinite(prediction):
        raise ValueError("Frozen Decision Tree returned a non-finite raw prediction.")

    try:
        import shap
    except ImportError as error:  # pragma: no cover - covered by dependency contract in CI
        raise RuntimeError(
            "SHAP is required for explanations; reinstall the project dependencies."
        ) from error
    explainer = shap.TreeExplainer(
        bundle.model,
        data=background_matrix,
        feature_perturbation="interventional",
        model_output="raw",
        feature_names=list(bundle.feature_names),
    )
    explanation = explainer(row, check_additivity=True)
    global_explanation = explainer(background_matrix, check_additivity=True)
    contributions = np.asarray(explanation.values, dtype=np.float64)
    base_values = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
    if contributions.shape != (1, len(bundle.feature_names)) or base_values.size != 1:
        raise ValueError("SHAP returned values with an unexpected feature or row shape.")
    baseline = float(base_values[0])
    feature_contributions = tuple(
        FeatureContribution(name, float(value), float(shap_value))
        for name, value, shap_value in zip(
            bundle.feature_names, vector.values, contributions[0], strict=True
        )
    )
    mean_absolute = np.mean(np.abs(np.asarray(global_explanation.values, dtype=np.float64)), axis=0)
    ranked_global = sorted(
        zip(bundle.feature_names, mean_absolute, strict=True),
        key=lambda item: (-float(item[1]), bundle.feature_names.index(item[0])),
    )[:10]
    global_summary = tuple(
        GlobalFeatureImportance(name, float(value)) for name, value in ranked_global
    )
    reconstructed = baseline + math.fsum(item.contribution_cycles for item in feature_contributions)
    error = abs(reconstructed - prediction)
    tolerance = ADDITIVITY_ABSOLUTE_TOLERANCE + ADDITIVITY_RELATIVE_TOLERANCE * abs(prediction)
    if error > tolerance:
        raise ValueError(
            f"SHAP contributions do not reproduce the raw prediction within {tolerance:g} cycles."
        )

    return FinalTreeExplanation(
        model_name=bundle.name,
        model_version=bundle.version,
        dataset_partition=vector.dataset_partition,
        unit_id=vector.unit_id,
        observed_cycle=vector.selected_cycle,
        raw_rul_cycles=prediction,
        base_value_cycles=baseline,
        contributions=feature_contributions,
        global_summary=global_summary,
        feature_config_sha256=_canonical_sha256(asdict(bundle.feature_config)),
        input_history_sha256=_canonical_sha256(normalized_rows),
        manifest_payload_sha256=_canonical_sha256(bundle.payload),
        weights_sha256=bundle.payload["model"]["weights_sha256"],
        training_source_sha256=training_source_sha,
        background_sha256=background.background_sha256,
        background_selection_policy=background.selection_policy,
        background_row_count=len(background.values),
        additivity_error_cycles=error,
    )


def explain_final_tree_history(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    artifact_root: str | Path = PROJECT_ROOT,
    training_path: str | Path = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt",
) -> FinalTreeExplanation:
    """Explain a prefix using the registered final tree and its train reference.

    This convenience entry point loads the frozen Decision Tree, verifies the
    original train-source hash, builds the documented fixed background, then
    delegates to the strict single-prefix explanation service.
    """
    rows = tuple(history)
    spec = FINAL_MODEL_REGISTRY[DECISION_TREE_KEY]
    bundle = load_final_tree_bundle(Path(artifact_root) / spec.manifest_path)
    background = build_fd001_training_background(
        training_path,
        expected_training_source_sha256=bundle.payload["training"]["training_source_sha256"],
        feature_config=bundle.feature_config,
    )
    return explain_final_tree_prediction(
        rows,
        dataset_partition=dataset_partition,
        unit_id=unit_id,
        background=background,
        artifact_root=artifact_root,
    )


def check_final_explanation_consistency(
    comparison: FinalHistoryComparison,
    explanation: FinalTreeExplanation,
) -> ExplanationConsistency:
    """Check that an explanation belongs to the unclipped prediction beside it.

    A raw ``FinalHistoryComparison`` is required deliberately, so a display-
    clamped value cannot be substituted for the model's evaluation output.
    """
    if not isinstance(comparison, FinalHistoryComparison):
        raise TypeError("comparison must be a FinalHistoryComparison with raw model outputs.")
    if not isinstance(explanation, FinalTreeExplanation):
        raise TypeError("explanation must be a FinalTreeExplanation.")

    mismatches: list[str] = []
    if explanation.model_name != DECISION_TREE_KEY:
        mismatches.append("unsupported_explanation_model")
    candidates = [
        item for item in comparison.predictions if item.model_name == explanation.model_name
    ]
    prediction = candidates[0] if len(candidates) == 1 else None
    if prediction is None:
        mismatches.append("prediction_missing_or_ambiguous")
    if explanation.dataset_partition != comparison.dataset_partition:
        mismatches.append("dataset_partition_mismatch")
    if explanation.unit_id != comparison.unit_id:
        mismatches.append("unit_id_mismatch")
    if explanation.observed_cycle != comparison.observed_cycle:
        mismatches.append("observed_cycle_mismatch")
    if explanation.input_history_sha256 != comparison.input_history_sha256:
        mismatches.append("input_history_mismatch")

    if prediction is not None:
        vector = prediction.vector
        if prediction.stage != "final":
            mismatches.append("prediction_not_final")
        if prediction.model_version != explanation.model_version:
            mismatches.append("model_version_mismatch")
        if prediction.manifest_payload_sha256 != explanation.manifest_payload_sha256:
            mismatches.append("manifest_mismatch")
        if prediction.weights_sha256 != explanation.weights_sha256:
            mismatches.append("weights_mismatch")
        if prediction.feature_config_sha256 != explanation.feature_config_sha256:
            mismatches.append("feature_config_mismatch")
        if prediction.training_source_sha256 != explanation.training_source_sha256:
            mismatches.append("training_source_mismatch")
        if (
            vector.dataset_partition != explanation.dataset_partition
            or vector.unit_id != explanation.unit_id
            or vector.selected_cycle != explanation.observed_cycle
        ):
            mismatches.append("feature_row_identity_mismatch")
        if tuple(item.feature_name for item in explanation.contributions) != vector.config.output_schema:
            mismatches.append("feature_schema_mismatch")
        if tuple(item.feature_value for item in explanation.contributions) != vector.values:
            mismatches.append("feature_values_mismatch")
        if not math.isclose(
            prediction.raw_rul_cycles,
            explanation.raw_rul_cycles,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches.append("raw_prediction_mismatch")

    return ExplanationConsistency(
        is_consistent=not mismatches,
        mismatch_codes=tuple(mismatches),
        model_name=explanation.model_name,
        model_version=explanation.model_version,
        dataset_partition=explanation.dataset_partition,
        unit_id=explanation.unit_id,
        observed_cycle=explanation.observed_cycle,
        raw_rul_cycles=explanation.raw_rul_cycles,
        input_history_sha256=explanation.input_history_sha256,
    )


def _validate_tree_bundle(bundle: FinalTreeBundle, spec) -> None:
    if (
        bundle.name != spec.name
        or bundle.version != spec.version
        or bundle.payload.get("artifact_role") != spec.artifact_role
        or bundle.feature_config.version != spec.feature_config_version
        or bundle.feature_config.window_cycles != spec.feature_window_cycles
        or bundle.feature_names != bundle.feature_config.output_schema
        or len(bundle.feature_names) != spec.feature_count
        or not isinstance(bundle.model, DecisionTreeRegressor)
    ):
        raise ValueError("Loaded bundle is not the registered frozen Decision Tree/schema.")


def _validate_background(
    background: TrainingFeatureBackground,
    bundle: FinalTreeBundle,
    training_source_sha256: str,
) -> None:
    if (
        background.training_source_sha256 != training_source_sha256
        or background.feature_names != bundle.feature_names
        or background.selection_policy != BACKGROUND_SELECTION_POLICY
        or not _is_sha256(background.background_sha256)
    ):
        raise ValueError("Background source, selection policy or schema differs from the bundle contract.")
    matrix = np.asarray(background.values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(bundle.feature_names)
        or not 100 <= matrix.shape[0] <= 1000
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Background must contain 100–1000 finite rows in the saved feature order.")
    expected_digest = _canonical_sha256(
        {
            "values": background.values,
            "feature_names": background.feature_names,
            "training_source_sha256": background.training_source_sha256,
            "selection_policy": background.selection_policy,
        }
    )
    if expected_digest != background.background_sha256:
        raise ValueError("Background content hash does not match its declared rows and provenance.")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
