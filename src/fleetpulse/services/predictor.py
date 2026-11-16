"""Registered saved-model inference with separate raw and presentation contracts.

Abdullah confirmed agreement with Shaheer and Taha on 13 November 2026:
one observed engine prefix in the normalized 26-column schema, explicit
partition/engine identity and saved manifest path; return only raw_rul_cycles.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import joblib
import sklearn

from fleetpulse.data.quality import validate_engine_history
from fleetpulse.features import EngineFeatureVector, FeatureConfig, engineer_features
from fleetpulse.models.registry import MODEL_REGISTRY


class PredictionInputError(ValueError):
    """The caller's observed history cannot form valid model inputs."""


class PredictionArtifactError(ValueError):
    """The saved model is missing, corrupt, unsupported or incompatible."""


@dataclass(frozen=True)
class RawRULPrediction:
    """Unclipped model output in operating cycles, preserved for evaluation."""

    raw_rul_cycles: float


@dataclass(frozen=True)
class RULPrediction:
    """Raw evaluation value alongside explicitly separate display/context fields."""

    raw_rul_cycles: float
    displayed_rul_cycles: float
    model_name: str
    model_version: str
    observed_cycle: int
    input_window_rows: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Inference:
    raw_rul_cycles: float
    model_name: str
    model_version: str
    vector: EngineFeatureVector
    cycle_gaps: bool


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _require_artifact(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionArtifactError(message)


@lru_cache(maxsize=2)
def _deserialize_model(
    model_bytes: bytes, model_name: str
) -> Any:
    """Reuse an estimator only for exactly the same verified model bytes."""
    spec = MODEL_REGISTRY[model_name]
    estimator_type = spec.estimator_type
    if estimator_type is None and model_name == "xgboost-rul":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise PredictionArtifactError(
                "Loading this candidate requires the optional XGBoost extra; install with "
                "`python -m pip install -e '.[xgboost]'`."
            ) from error
        estimator_type = XGBRegressor
    _require_artifact(estimator_type is not None, "Registered estimator type is unavailable.")
    model = joblib.load(io.BytesIO(model_bytes))
    _require_artifact(
        type(model) is estimator_type,
        f"Saved model is not a {estimator_type.__name__}.",
    )
    return model


def _load_model(
    manifest_path: str | Path,
) -> tuple[FeatureConfig, Any, str, str]:
    """Read a registered saved bundle without requiring any evaluation dataset.

    Only trusted project artifacts should be passed to joblib. Evaluation
    targets/predictions in the hashed P06 manifest are never used for inference.
    """
    path = Path(manifest_path)
    title = "model"  # The artifact is unidentified until its manifest is validated.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document["payload"]
        _require_artifact(
            document["sha256"] == hashlib.sha256(_canonical(payload)).hexdigest(),
            "Saved model manifest content hash mismatch.",
        )
        info, training = payload["model"], payload["training"]
        spec = MODEL_REGISTRY.get(info["name"])
        _require_artifact(spec is not None, "Unsupported model name/version; unregistered model.")
        title = spec.display_name
        _require_artifact(
            info["version"] == spec.version, f"Unsupported {title} model name/version."
        )
        _require_artifact(
            payload["schema_version"] == spec.artifact_schema,
            f"Unsupported {title} artifact schema.",
        )
        if spec.name == "xgboost-rul":
            try:
                runtime_xgboost_version = version("xgboost")
            except PackageNotFoundError as error:
                raise PredictionArtifactError(
                    "Loading this candidate requires the optional XGBoost extra."
                ) from error
            _require_artifact(
                info["xgboost_version"] == runtime_xgboost_version,
                "Saved XGBoost version does not match the runtime.",
            )
        _require_artifact(
            info["scikit_learn_version"] == sklearn.__version__,
            "Saved scikit-learn version does not match the runtime.",
        )
        _require_artifact(
            info["filename"] == spec.filename, f"Expected the sibling model file {spec.filename}."
        )
        saved_config = training["feature_config"]
        _require_artifact(isinstance(saved_config["channels"], list), "Invalid saved channel list.")
        config = FeatureConfig(
            channels=tuple(saved_config["channels"]), window_cycles=saved_config["window_cycles"]
        )
        _require_artifact(
            json.loads(_canonical(asdict(config))) == saved_config,
            "Unsupported or inconsistent saved feature configuration/version.",
        )
        _require_artifact(
            training["feature_names"] == list(config.output_schema),
            "Saved feature names/order do not match the feature configuration.",
        )
        model_path = path.parent / spec.filename
        model_bytes = model_path.read_bytes()
        _require_artifact(
            hashlib.sha256(model_bytes).hexdigest() == info["sha256"],
            f"{title} model file hash mismatch.",
        )
    except PredictionArtifactError:
        raise
    except OSError as error:
        failed_path = error.filename if error.filename is not None else path
        raise PredictionArtifactError(
            f"Cannot read saved {title} bundle at {path}; "
            f"unavailable file {failed_path}: {error.strerror or error}"
        ) from error
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise PredictionArtifactError(f"Invalid {title} manifest at {path}: {error}") from error

    try:
        # Integrity is checked on every call; only deserialization is cached by exact bytes.
        model = _deserialize_model(model_bytes, spec.name)
        _require_artifact(
            model.n_features_in_ == len(config.output_schema)
            and getattr(model, "n_outputs_", 1) == 1,
            "Saved estimator dimensions do not match the feature schema.",
        )
        if hasattr(model, "feature_names_in_"):
            _require_artifact(
                tuple(model.feature_names_in_) == config.output_schema,
                "Saved estimator feature order differs from the manifest.",
            )
        expected_parameters = {**training["parameters"], "random_state": training["random_seed"]}
        _require_artifact(
            all(model.get_params().get(key) == value for key, value in expected_parameters.items()),
            "Saved estimator parameters differ from the manifest.",
        )
    except PredictionArtifactError:
        raise
    except Exception as error:
        raise PredictionArtifactError(f"Cannot load a compatible saved {title}: {error}") from error
    return config, model, info["name"], info["version"]


def _infer(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    manifest_path: str | Path,
) -> _Inference:
    """Validate one observed prefix and return its raw saved-model RUL estimate.

    The final supplied row is the observation cutoff; callers must truncate
    hidden future rows before this call. Require all 26 normalized raw columns
    and reject extra target/lifetime fields. IDs and cycles remain metadata.
    Gaps and short histories follow the reviewed feature contract. No file is
    read except the explicit registered manifest and its sibling saved estimator.
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
    # Normalize validated integer-valued identities on copies, preserving caller inputs.
    normalized = tuple(
        {**row, "unit_id": int(row["unit_id"]), "cycle": int(row["cycle"])} for row in rows
    )
    config, model, model_name, model_version = _load_model(manifest_path)
    try:
        vector = engineer_features(
            normalized, dataset_partition=dataset_partition, unit_id=int(unit_id), config=config
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise PredictionInputError(f"Cannot build finite saved-schema features: {error}") from error
    try:
        predicted = model.predict((vector.values,))
        _require_artifact(
            len(predicted) == 1
            and isinstance(predicted[0], Real)
            and not isinstance(predicted[0], bool)
            and math.isfinite(float(predicted[0])),
            "Saved estimator did not return one finite numeric prediction.",
        )
    except PredictionArtifactError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise PredictionArtifactError(
            f"Saved model cannot predict these features: {error}"
        ) from error
    return _Inference(
        raw_rul_cycles=float(predicted[0]),
        model_name=model_name,
        model_version=model_version,
        vector=vector,
        cycle_gaps=any(finding.code == "cycle_gap" for finding in report.warnings),
    )


def predict_rul(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    manifest_path: str | Path,
) -> RawRULPrediction:
    """Return only the unclipped estimate; preserve the agreed P07-A1 interface.

    Supply one raw observed prefix with no labels/future rows. This API is for
    raw evaluation and does not apply a display policy or add context fields.
    """
    result = _infer(
        history, dataset_partition=dataset_partition, unit_id=unit_id, manifest_path=manifest_path
    )
    return RawRULPrediction(raw_rul_cycles=result.raw_rul_cycles)


def predict_rul_with_context(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    manifest_path: str | Path,
) -> RULPrediction:
    """Return raw and nonnegative display values plus validated model context.

    Same observed-prefix input contract as predict_rul. Warning codes are
    cycle_gaps, short_window (fewer observed rows than configured cycle span),
    and display_clipped_at_zero. Abdullah confirmed both Shaheer and Taha
    agreed these fields on 13 November 2026; the raw API stays unchanged.
    """
    result = _infer(
        history, dataset_partition=dataset_partition, unit_id=unit_id, manifest_path=manifest_path
    )
    warnings = []
    if result.cycle_gaps:
        warnings.append("cycle_gaps")
    if result.vector.window_row_count < result.vector.config.window_cycles:
        warnings.append("short_window")
    if result.raw_rul_cycles < 0:
        warnings.append("display_clipped_at_zero")
    return RULPrediction(
        raw_rul_cycles=result.raw_rul_cycles,
        displayed_rul_cycles=max(0.0, result.raw_rul_cycles),
        model_name=result.model_name,
        model_version=result.model_version,
        observed_cycle=result.vector.selected_cycle,
        input_window_rows=result.vector.window_row_count,
        warnings=tuple(warnings),
    )
