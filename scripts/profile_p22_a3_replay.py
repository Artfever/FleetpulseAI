"""Profile P22-A3 history replay and compare the optimized row scan to baseline."""

from __future__ import annotations

import cProfile
import hashlib
import json
import math
import pstats
import statistics
import time
from collections.abc import Mapping
from pathlib import Path

import sklearn

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import PredictionInputError
from fleetpulse.services import history as history_service

SOURCE_PATH = PROJECT_ROOT / "data/raw/train_FD001.txt"
MANIFEST_PATH = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
MODEL_PATH = MANIFEST_PATH.parent / "decision_tree_model.joblib"
SOURCE_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
REPLAY_CUTOFFS = (16, 33, 49, 66, 82, 98, 115, 131, 148, 164)
REPEATS = 30
WARM_REPEATS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_select(records, selected_unit: int) -> list[dict[str, object]]:
    """The pre-optimization selection loop, retained for apples-to-apples timing."""
    selected: list[dict[str, object]] = []
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise PredictionInputError(f"History row {row_number} must be a mapping.")
        try:
            row_unit_id = history_service._positive_integer(record.get("unit_id"), "history unit_id")
        except PredictionInputError:
            raise PredictionInputError(
                f"History row {row_number} has no valid engine identity; replay cannot verify completeness."
            ) from None
        if row_unit_id == selected_unit:
            selected.append(dict(record))
    return selected


def _signature(result) -> list[dict[str, object]]:
    return [
        {
            "cutoff_cycle": point.cutoff_cycle,
            "raw_rul_cycles": point.prediction.raw_rul_cycles,
            "displayed_rul_cycles": point.prediction.displayed_rul_cycles,
            "model_name": point.prediction.model_name,
            "model_version": point.prediction.model_version,
            "observed_cycle": point.prediction.observed_cycle,
            "input_window_rows": point.prediction.input_window_rows,
            "warnings": list(point.prediction.warnings),
        }
        for point in result.points
    ]


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[p95_index],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def _replay(records, selector, *, clear_cache: bool):
    previous = history_service._select_replay_engine_records
    history_service._select_replay_engine_records = selector
    try:
        if clear_cache:
            history_service.clear_history_replay_cache()
        return history_service.replay_engine_history(
            records,
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=164,
            replay_cutoffs=REPLAY_CUTOFFS,
            manifest_path=MANIFEST_PATH,
        )
    finally:
        history_service._select_replay_engine_records = previous


def _measure_comparison(records, *, clear_cache: bool, repeats: int):
    selectors = {
        "baseline": _baseline_select,
        "optimized": history_service._select_replay_engine_records,
    }
    durations = {name: [] for name in selectors}
    results = {}
    for iteration in range(repeats):
        order = ("baseline", "optimized") if iteration % 2 == 0 else ("optimized", "baseline")
        for name in order:
            started = time.perf_counter()
            results[name] = _replay(records, selectors[name], clear_cache=clear_cache)
            durations[name].append(time.perf_counter() - started)
    return results, {name: _summary(samples) for name, samples in durations.items()}


def run_profile() -> dict[str, object]:
    for path in (SOURCE_PATH, MANIFEST_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required local P22-A3 profile input is missing: {path}")
    source_hash = _sha256(SOURCE_PATH)
    if source_hash != SOURCE_SHA256:
        raise ValueError("FD001/train source hash does not match the reviewed dataset.")

    full_history = load_engine_history(SOURCE_PATH)
    engine_history = [row for row in full_history if row["unit_id"] == 4]
    # Load and validate the local bundle before timing either selector so the
    # first-call model-load cost cannot skew one side of the comparison.
    _replay(full_history, _baseline_select, clear_cache=True)
    history_service.clear_history_replay_cache()
    cold_results, cold_timings = _measure_comparison(
        full_history, clear_cache=True, repeats=REPEATS
    )
    baseline_result = cold_results["baseline"]
    optimized_result = cold_results["optimized"]
    if _signature(baseline_result) != _signature(optimized_result):
        raise AssertionError("Optimized replay changed one or more raw prediction outputs.")

    _, warm_timings = _measure_comparison(
        full_history, clear_cache=False, repeats=WARM_REPEATS
    )

    history_service.clear_history_replay_cache()
    profiler = cProfile.Profile()
    profiler.enable()
    _replay(full_history, history_service._select_replay_engine_records, clear_cache=True)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    profile_top = []
    for (filename, line, function), (primitive, calls, total, cumulative, _) in stats.stats.items():
        profile_top.append(
            {
                "function": f"{filename}:{line}({function})",
                "calls": calls,
                "self_seconds": total,
                "cumulative_seconds": cumulative,
            }
        )
    profile_top.sort(key=lambda row: row["cumulative_seconds"], reverse=True)

    model_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["payload"]
    return {
        "dataset": "NASA C-MAPSS FD001/train (simulated)",
        "source_sha256": source_hash,
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "model_sha256": _sha256(MODEL_PATH),
        "model_name": model_payload["model"]["name"],
        "model_version": model_payload["model"]["version"],
        "feature_config": model_payload["training"]["feature_config"],
        "scikit_learn_version": sklearn.__version__,
        "profile_sample": {
            "partition": "FD001/train",
            "engine": 4,
            "selected_cycle": 164,
            "cutoffs": list(REPLAY_CUTOFFS),
            "full_fleet_rows": len(full_history),
            "selected_engine_rows": len(engine_history),
            "repeats": REPEATS,
            "warm_cache_repeats": WARM_REPEATS,
        },
        "baseline_full_fleet_cold_replay": cold_timings["baseline"],
        "optimized_full_fleet_cold_replay": cold_timings["optimized"],
        "baseline_full_fleet_warm_cache_replay": warm_timings["baseline"],
        "optimized_full_fleet_warm_cache_replay": warm_timings["optimized"],
        "parity": {
            "exact": True,
            "cutoff_count": len(_signature(optimized_result)),
            "predictions": _signature(optimized_result),
        },
        "profile_top_cumulative": profile_top[:12],
        "limits": [
            "This is a local replay microbenchmark on one packaged simulated training history.",
            "Timings describe this machine/process and do not establish user-visible latency targets.",
            "The baseline selector reproduces the prior row-validation loop; model artifacts and replay logic are otherwise identical.",
            "No training, official-test labels, or maintenance claims are involved.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_profile(), indent=2, sort_keys=True))
