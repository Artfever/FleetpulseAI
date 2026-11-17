"""Bounded, past-only prediction replay for one engine history."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from numbers import Integral, Real
from pathlib import Path
from threading import RLock

from fleetpulse.data.quality import validate_engine_history
from fleetpulse.services.fleet import EngineIdentity
from fleetpulse.services.predictor import (
    PredictionInputError,
    RULPrediction,
    predict_rul_with_context,
)

MAX_REPLAY_POINTS = 10
MAX_REPLAY_CACHE_POINTS = 128

_REPLAY_CACHE: OrderedDict[tuple[str, ...], RULPrediction] = OrderedDict()
_REPLAY_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class HistoryReplayPoint:
    """One saved-model result formed from rows through a single cutoff."""

    cutoff_cycle: int
    prediction: RULPrediction


@dataclass(frozen=True)
class HistoryReplay:
    """A bounded ordered set of replay points for one identified engine."""

    identity: EngineIdentity
    selected_cycle: int
    points: tuple[HistoryReplayPoint, ...]


def clear_history_replay_cache() -> None:
    """Clear cached replay predictions, primarily for controlled workflows/tests."""
    with _REPLAY_CACHE_LOCK:
        _REPLAY_CACHE.clear()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _model_cache_fingerprints(manifest_path: str | Path) -> tuple[str, str, str] | None:
    """Return manifest/model, model-version and feature-config fingerprints.

    If the local bundle metadata cannot be read or recognized, inference still
    runs uncached so normal prediction errors and non-file manifests retain
    their original behavior.
    """
    path = Path(manifest_path)
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, Mapping):
            return None
        payload = manifest.get("payload", manifest)
        if not isinstance(payload, Mapping):
            return None
        model = payload.get("model", {})
        training = payload.get("training", {})
        if not isinstance(model, Mapping) or not isinstance(training, Mapping):
            return None
        filename = model.get("filename")
        version = model.get("version")
        feature_config = training.get("feature_config")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(version, str)
            or not version
            or not isinstance(feature_config, Mapping)
        ):
            return None
        model_bytes = (path.parent / filename).read_bytes()
        artifact_fingerprint = hashlib.sha256(manifest_bytes + b"\0" + model_bytes).hexdigest()
        feature_fingerprint = hashlib.sha256(_canonical_json(feature_config)).hexdigest()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return artifact_fingerprint, version, feature_fingerprint


def _predict_with_replay_cache(
    prefix: tuple[dict[str, object], ...],
    *,
    identity: EngineIdentity,
    cutoff: int,
    manifest_path: str | Path,
) -> RULPrediction:
    model_fingerprints = _model_cache_fingerprints(manifest_path)
    if model_fingerprints is None:
        return predict_rul_with_context(
            prefix,
            dataset_partition=identity.dataset_partition,
            unit_id=identity.unit_id,
            manifest_path=manifest_path,
        )

    artifact_fingerprint, model_version, feature_fingerprint = model_fingerprints
    history_fingerprint = hashlib.sha256(_canonical_json(prefix)).hexdigest()
    cache_key = (
        identity.dataset_partition,
        str(identity.unit_id),
        str(cutoff),
        history_fingerprint,
        model_version,
        feature_fingerprint,
        artifact_fingerprint,
    )
    with _REPLAY_CACHE_LOCK:
        cached = _REPLAY_CACHE.get(cache_key)
        if cached is not None:
            _REPLAY_CACHE.move_to_end(cache_key)
            return cached

    prediction = predict_rul_with_context(
        prefix,
        dataset_partition=identity.dataset_partition,
        unit_id=identity.unit_id,
        manifest_path=manifest_path,
    )
    with _REPLAY_CACHE_LOCK:
        _REPLAY_CACHE[cache_key] = prediction
        _REPLAY_CACHE.move_to_end(cache_key)
        while len(_REPLAY_CACHE) > MAX_REPLAY_CACHE_POINTS:
            _REPLAY_CACHE.popitem(last=False)
    return prediction


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PredictionInputError(f"{name} must be a positive integer.")
    if isinstance(value, Integral):
        result = int(value)
    else:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise PredictionInputError(f"{name} must be a positive integer.")
        result = int(number)
    if result <= 0:
        raise PredictionInputError(f"{name} must be a positive integer.")
    return result


def replay_engine_history(
    records: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    selected_cycle: int,
    replay_cutoffs: Iterable[int],
    manifest_path: str | Path,
) -> HistoryReplay:
    """Predict at a small grid of observed cycles, never passing later rows.

    Other identified engines in ``records`` are ignored. The requested
    engine's history is validated, and each model call receives a separate
    copy containing only rows whose cycle is at or before that point's
    cutoff. Cutoffs are sorted and capped at :data:`MAX_REPLAY_POINTS` to keep
    UI replay work bounded.
    """
    if not isinstance(dataset_partition, str) or not dataset_partition.strip():
        raise PredictionInputError("dataset_partition must be a nonempty string.")
    selected_unit = _positive_integer(unit_id, "unit_id")
    selected_cycle_value = _positive_integer(selected_cycle, "selected_cycle")
    try:
        all_records = tuple(records)
    except TypeError as error:
        raise PredictionInputError("records must be an iterable of engine mappings.") from error

    engine_records: list[dict[str, object]] = []
    for row_number, record in enumerate(all_records, start=1):
        if not isinstance(record, Mapping):
            raise PredictionInputError(f"History row {row_number} must be a mapping.")
        row_unit = record.get("unit_id")
        try:
            row_unit_id = _positive_integer(row_unit, "history unit_id")
        except PredictionInputError:
            raise PredictionInputError(
                f"History row {row_number} has no valid engine identity; replay cannot verify completeness."
            ) from None
        if row_unit_id == selected_unit:
            engine_records.append(dict(record))
    if not engine_records:
        raise PredictionInputError(f"Engine {selected_unit} is not present in the supplied history.")

    report = validate_engine_history(engine_records)
    if not report.is_valid:
        codes = ", ".join(sorted({finding.code for finding in report.errors}))
        raise PredictionInputError(f"Engine {selected_unit} history is invalid: {codes}.")
    observed_cycles = tuple(int(record["cycle"]) for record in engine_records)
    if selected_cycle_value not in observed_cycles:
        raise PredictionInputError("selected_cycle must be an observed cycle for the selected engine.")

    try:
        cutoffs = tuple(islice(iter(replay_cutoffs), MAX_REPLAY_POINTS + 1))
    except TypeError as error:
        raise PredictionInputError("replay_cutoffs must be an iterable of observed cycles.") from error
    if not cutoffs:
        raise PredictionInputError("replay_cutoffs must contain at least one cycle.")
    if len(cutoffs) > MAX_REPLAY_POINTS:
        raise PredictionInputError(
            f"replay_cutoffs cannot contain more than {MAX_REPLAY_POINTS} points."
        )
    normalized_cutoffs = tuple(
        _positive_integer(cutoff, "replay cutoff") for cutoff in cutoffs
    )
    if len(set(normalized_cutoffs)) != len(normalized_cutoffs):
        raise PredictionInputError("replay_cutoffs cannot contain duplicate cycles.")
    if any(cutoff > selected_cycle_value for cutoff in normalized_cutoffs):
        raise PredictionInputError("replay cutoffs cannot be later than selected_cycle.")
    if any(cutoff not in observed_cycles for cutoff in normalized_cutoffs):
        raise PredictionInputError("Every replay cutoff must be an observed cycle for the selected engine.")

    identity = EngineIdentity(dataset_partition.strip(), selected_unit)
    points: list[HistoryReplayPoint] = []
    for cutoff in sorted(normalized_cutoffs):
        prefix = tuple(
            dict(record)
            for record in engine_records
            if int(record["cycle"]) <= cutoff
        )
        prediction = _predict_with_replay_cache(
            prefix,
            identity=identity,
            cutoff=cutoff,
            manifest_path=manifest_path,
        )
        if prediction.observed_cycle != cutoff:
            raise PredictionInputError(
                f"Replay prediction cutoff mismatch for engine {selected_unit}, cycle {cutoff}."
            )
        points.append(HistoryReplayPoint(cutoff_cycle=cutoff, prediction=prediction))

    return HistoryReplay(identity=identity, selected_cycle=selected_cycle_value, points=tuple(points))
