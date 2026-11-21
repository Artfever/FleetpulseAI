"""Validate and extract independently identified engine histories for fleet use."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

from fleetpulse.data.quality import validate_engine_history
from fleetpulse.services.predictor import (
    PredictionArtifactError,
    PredictionInputError,
    RULPrediction,
    predict_rul_with_context,
)


@dataclass(frozen=True, order=True)
class EngineIdentity:
    """Dataset-qualified identity; unit IDs can recur across partitions."""

    dataset_partition: str
    unit_id: int


@dataclass(frozen=True)
class FleetEngineHistory:
    """One validated, ordered observed prefix for a single engine."""

    identity: EngineIdentity
    records: tuple[Mapping[str, object], ...]
    observed_cycle: int


@dataclass(frozen=True)
class FleetEngineHistoryError:
    """Validation failure retained with the engine identity that produced it."""

    identity: EngineIdentity
    codes: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class FleetHistories:
    """Valid engine prefixes and per-engine validation errors from one partition."""

    histories: tuple[FleetEngineHistory, ...]
    errors: tuple[FleetEngineHistoryError, ...]


@dataclass(frozen=True)
class FleetPredictionResult:
    """Prediction or engine-specific failure, always attached to its identity."""

    identity: EngineIdentity
    observed_cycle: int | None
    raw_rul_cycles: float | None
    displayed_rul_cycles: float | None
    model_name: str | None
    model_version: str | None
    input_window_rows: int | None
    warnings: tuple[str, ...]
    error_codes: tuple[str, ...]
    error: str | None


def _prediction_result(
    identity: EngineIdentity,
    prediction: RULPrediction | None = None,
    *,
    observed_cycle: int | None = None,
    error_codes: tuple[str, ...] = (),
    error: str | None = None,
) -> FleetPredictionResult:
    if prediction is None:
        return FleetPredictionResult(
            identity=identity,
            observed_cycle=observed_cycle,
            raw_rul_cycles=None,
            displayed_rul_cycles=None,
            model_name=None,
            model_version=None,
            input_window_rows=None,
            warnings=(),
            error_codes=error_codes,
            error=error,
        )
    return FleetPredictionResult(
        identity=identity,
        observed_cycle=prediction.observed_cycle,
        raw_rul_cycles=prediction.raw_rul_cycles,
        displayed_rul_cycles=prediction.displayed_rul_cycles,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        input_window_rows=prediction.input_window_rows,
        warnings=prediction.warnings,
        error_codes=(),
        error=None,
    )


def _positive_integral_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if isinstance(value, Integral):
        return int(value) if value > 0 else None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return int(numeric)


def extract_latest_engine_histories(
    records: Iterable[Mapping[str, object]], *, dataset_partition: str
) -> FleetHistories:
    """Group, validate, and extract each engine's full ordered observed prefix.

    Histories are grouped without sorting or dropping rows. Each engine is
    validated independently: a malformed engine is returned as an error
    attached to its dataset-qualified identity while valid peers remain usable.
    A partition and positive integral engine ID are required so every outcome
    can retain its original identity. Cycle gaps remain quality warnings; the
    final record in each validated prefix supplies its observed cycle.
    """
    if not isinstance(dataset_partition, str) or not dataset_partition.strip():
        raise PredictionInputError("dataset_partition must be a nonempty string.")
    try:
        rows = tuple(records)
    except TypeError as error:
        raise PredictionInputError("records must be an iterable of engine mappings.") from error
    if not rows:
        raise PredictionInputError("Fleet input must contain at least one engine record.")

    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise PredictionInputError(f"Fleet row {row_number} must be a mapping.")
        unit_id = _positive_integral_id(row.get("unit_id"))
        if unit_id is None:
            raise PredictionInputError(
                f"Fleet row {row_number} has no valid positive integral unit_id; "
                "its engine identity cannot be preserved."
            )
        grouped.setdefault(unit_id, []).append(row)

    histories: list[FleetEngineHistory] = []
    errors: list[FleetEngineHistoryError] = []
    for unit_id in sorted(grouped):
        identity = EngineIdentity(dataset_partition.strip(), unit_id)
        engine_records = tuple(grouped[unit_id])
        report = validate_engine_history(engine_records)
        codes = tuple(sorted({finding.code for finding in report.errors}))
        if codes:
            errors.append(
                FleetEngineHistoryError(
                    identity=identity,
                    codes=codes,
                    message=f"Engine {unit_id} history is invalid: {', '.join(codes)}.",
                )
            )
            continue

        # Quality validation guarantees positive integral, strictly increasing
        # cycles; retain rows as supplied and use the last validated cycle.
        histories.append(
            FleetEngineHistory(
                identity=identity,
                records=engine_records,
                observed_cycle=int(engine_records[-1]["cycle"]),
            )
        )

    return FleetHistories(histories=tuple(histories), errors=tuple(errors))


def predict_fleet(
    records: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    manifest_path: str | Path,
) -> tuple[FleetPredictionResult, ...]:
    """Predict each valid engine prefix and keep results/errors ID-attached.

    Each engine is passed through the same saved-model prediction service as a
    single-engine call. History validation and prediction failures remain local
    to that engine; malformed peers do not change another engine's identity or
    discard its valid estimate. An empty/unidentifiable batch remains a
    batch-level ``PredictionInputError`` from ``extract_latest_engine_histories``.
    """
    batch = extract_latest_engine_histories(records, dataset_partition=dataset_partition)
    results: list[FleetPredictionResult] = []
    for failure in batch.errors:
        results.append(
            _prediction_result(
                failure.identity,
                error_codes=failure.codes,
                error=failure.message,
            )
        )

    for engine in batch.histories:
        try:
            prediction = predict_rul_with_context(
                engine.records,
                dataset_partition=engine.identity.dataset_partition,
                unit_id=engine.identity.unit_id,
                manifest_path=manifest_path,
            )
        except PredictionInputError as error:
            results.append(
                _prediction_result(
                    engine.identity,
                    observed_cycle=engine.observed_cycle,
                    error_codes=("prediction_input_error",),
                    error=str(error),
                )
            )
        except PredictionArtifactError as error:
            results.append(
                _prediction_result(
                    engine.identity,
                    observed_cycle=engine.observed_cycle,
                    error_codes=("prediction_artifact_error",),
                    error=str(error),
                )
            )
        else:
            results.append(_prediction_result(engine.identity, prediction))

    return tuple(sorted(results, key=lambda result: result.identity))
