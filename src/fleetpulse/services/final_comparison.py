"""Run frozen P12 models on one validated observed engine prefix (P17-A1)."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.quality import validate_engine_history
from fleetpulse.features import EngineFeatureVector, FeatureConfig, engineer_features
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY, FinalModelSpec
from fleetpulse.services.predictor import PredictionInputError


@dataclass(frozen=True)
class FinalModelPrediction:
    model_name: str
    model_version: str
    stage: str
    raw_rul_cycles: float
    manifest_payload_sha256: str
    weights_sha256: str | None
    feature_config_sha256: str
    training_source_sha256: str
    selection_manifest_sha256: str
    vector: EngineFeatureVector


@dataclass(frozen=True)
class FinalModelFailure:
    model_name: str
    code: str
    reason: str


@dataclass(frozen=True)
class FinalHistoryComparison:
    dataset_partition: str
    unit_id: int
    observed_cycle: int
    input_history_sha256: str
    predictions: tuple[FinalModelPrediction, ...]
    failures: tuple[FinalModelFailure, ...]

    @property
    def raw_min_rul_cycles(self) -> float | None:
        """Smallest successful raw estimate, or None when every model failed."""
        return min((item.raw_rul_cycles for item in self.predictions), default=None)

    @property
    def raw_max_rul_cycles(self) -> float | None:
        """Largest successful raw estimate, or None when every model failed."""
        return max((item.raw_rul_cycles for item in self.predictions), default=None)

    @property
    def raw_spread_cycles(self) -> float | None:
        """Return max(raw)-min(raw) across successes; failed models are excluded.

        This is model disagreement in cycles, not a probability or confidence
        interval. With one successful model the spread is zero; with none it is
        undefined and returned as None. An unrepresentable finite-float range
        raises OverflowError instead of returning infinity.
        """
        minimum = self.raw_min_rul_cycles
        maximum = self.raw_max_rul_cycles
        if minimum is None or maximum is None:
            return None
        spread = maximum - minimum
        if not math.isfinite(spread):
            raise OverflowError("Raw model spread exceeds the finite cycle range.")
        return spread


def _digest(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def _predict_bundle(path: Path, spec: FinalModelSpec, rows, partition, unit):
    # Import optional estimators only when their actual manifest is available.
    if not path.is_file():
        raise FileNotFoundError(f"Final {spec.name} manifest unavailable: {path}")
    if spec.name == "dense-rul-mlp-final":
        from fleetpulse.models.final_mlp_bundle import load_final_mlp_bundle, predict_final_mlp

        bundle = load_final_mlp_bundle(path)
        payload = bundle.payload
        saved = payload["training"]["feature_config"]
        config = FeatureConfig(
            channels=tuple(saved["channels"]), window_cycles=saved["window_cycles"]
        )
        names = tuple(payload["training"]["feature_names"])
        name = payload["model"]["name"]
        version = payload["model"]["version"]
        predict = lambda matrix: predict_final_mlp(bundle, matrix)
    else:
        from fleetpulse.models.final_tree_bundle import (
            load_final_tree_bundle,
            predict_final_tree_bundle,
        )

        bundle = load_final_tree_bundle(path)
        payload, config, names = bundle.payload, bundle.feature_config, bundle.feature_names
        name, version = bundle.name, bundle.version
        predict = lambda matrix: predict_final_tree_bundle(bundle, matrix)
    if (
        name != spec.name
        or version != spec.version
        or payload["artifact_role"] != spec.artifact_role
        or config.version != spec.feature_config_version
        or config.window_cycles != spec.feature_window_cycles
        or len(names) != spec.feature_count
        or names != config.output_schema
    ):
        raise ValueError(
            "Loaded bundle identity, stage or feature schema differs from final registry."
        )
    try:
        vector = engineer_features(rows, dataset_partition=partition, unit_id=unit, config=config)
    except (ValueError, TypeError, OverflowError) as error:
        raise PredictionInputError(f"Cannot form finite saved-schema features: {error}") from error
    values = predict(np.asarray([vector.values], dtype=np.float64))
    if len(values) != 1 or not math.isfinite(float(values[0])):
        raise ValueError("Final bundle did not return one finite raw prediction.")
    training = payload["training"]
    return FinalModelPrediction(
        model_name=name,
        model_version=version,
        stage=spec.stage,
        raw_rul_cycles=float(values[0]),
        manifest_payload_sha256=_digest(payload),
        weights_sha256=payload["model"].get("weights_sha256"),
        feature_config_sha256=_digest(asdict(config)),
        training_source_sha256=training["training_source_sha256"],
        selection_manifest_sha256=training["selection_manifest_sha256"],
        vector=vector,
    )


def compare_final_models(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    artifact_root: str | Path = PROJECT_ROOT,
) -> FinalHistoryComparison:
    """Compare all registered final bundles on exactly the same observed prefix.

    Supply one engine in the normalized 26-column schema, without labels or
    hidden future rows. The final supplied cycle is the observation cutoff.
    Each loader validates its own bundle and uses its saved feature config and
    preprocessing. Raw outputs remain unclipped. A failed/missing model has a
    reason, never a fabricated estimate. An empty predictions tuple means no
    model succeeded. There is no training, development fallback, or caching.
    """
    if not isinstance(dataset_partition, str) or not dataset_partition.strip():
        raise PredictionInputError("dataset_partition must be a nonempty string.")
    if isinstance(unit_id, bool) or not isinstance(unit_id, Integral) or unit_id <= 0:
        raise PredictionInputError("unit_id must be a positive integer.")
    try:
        rows = tuple(history)
    except TypeError as error:
        raise PredictionInputError("history must be an iterable of raw engine records.") from error
    report = validate_engine_history(rows)
    if not report.is_valid:
        codes = ", ".join(sorted({finding.code for finding in report.errors}))
        raise PredictionInputError(f"Invalid observed engine history: {codes}.")
    if {int(row["unit_id"]) for row in rows} != {int(unit_id)}:
        raise PredictionInputError("History must contain only the requested engine's unit_id.")
    # Own copies shared by every model: caller mutation cannot change later inputs.
    normalized = tuple(
        {**row, "unit_id": int(row["unit_id"]), "cycle": int(row["cycle"])} for row in rows
    )
    predictions, failures = [], []
    for spec in FINAL_MODEL_REGISTRY.values():
        try:
            predictions.append(
                _predict_bundle(
                    Path(artifact_root) / spec.manifest_path,
                    spec,
                    normalized,
                    dataset_partition,
                    int(unit_id),
                )
            )
        except PredictionInputError:
            raise
        except OSError as error:
            failures.append(FinalModelFailure(spec.name, "artifact_unavailable", str(error)))
        except ImportError as error:
            failures.append(FinalModelFailure(spec.name, "dependency_unavailable", str(error)))
        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            FloatingPointError,
            EOFError,
        ) as error:
            failures.append(FinalModelFailure(spec.name, "incompatible_artifact", str(error)))
        except (ArithmeticError, AttributeError, IndexError, pickle.UnpicklingError) as error:
            failures.append(FinalModelFailure(spec.name, "prediction_failed", str(error)))
    return FinalHistoryComparison(
        dataset_partition,
        int(unit_id),
        max(row["cycle"] for row in normalized),
        _digest(normalized),
        tuple(predictions),
        tuple(failures),
    )
