"""Display saved FD001/test model results in the interactive laboratory."""

import csv
import io
import json
from functools import lru_cache

import streamlit as st

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.final_test_display import FinalTestDisplayRow
from fleetpulse.evaluation.final_test_export import FinalTestCsvExports

_ASSETS = PROJECT_ROOT / "app" / "model_arena"


def build_model_arena_payload(
    results: tuple[FinalTestDisplayRow, ...], exports: FinalTestCsvExports | None
) -> dict:
    """Use checksum-checked summaries and only fully verified per-engine values."""
    models = [
        {
            "id": row.name,
            "name": row.display_name,
            "version": row.version,
            "mae": row.mae_cycles,
            "rmse": row.rmse_cycles,
        }
        for row in results
    ]
    engines = {}
    if exports is not None:
        for row in csv.DictReader(io.StringIO(exports.per_engine_csv)):
            engine_id = int(row["Engine ID"])
            observed = int(row["Observed cycle"])
            actual = int(row["True RUL (cycles)"])
            engine = engines.setdefault(
                engine_id, {"id": engine_id, "observed": observed, "actual": actual, "errors": {}}
            )
            if engine["observed"] != observed or engine["actual"] != actual:
                # The engine comparison needs the same observed cutoff and answer for every model.
                engines.clear()
                break
            engine["errors"][row["Model version"]] = {
                "residual": float(row["Residual prediction minus true (cycles)"]),
                "raw": float(row["Raw prediction (cycles)"]),
            }
        expected_versions = {row.version for row in results}
        if len(engines) != results[0].sample_count or any(
            engine["errors"].keys() != expected_versions for engine in engines.values()
        ):
            engines.clear()
    return {
        "sampleCount": results[0].sample_count,
        "models": models,
        "engines": [engines[engine_id] for engine_id in sorted(engines)],
    }


@lru_cache(maxsize=1)
def _document() -> str:
    return (
        (_ASSETS / "index.html").read_text(encoding="utf-8")
        .replace("/* STYLES */", (_ASSETS / "arena.css").read_text(encoding="utf-8"))
        .replace("/* SCRIPT */", (_ASSETS / "arena.js").read_text(encoding="utf-8"))
    )


def render_model_arena(
    results: tuple[FinalTestDisplayRow, ...], exports: FinalTestCsvExports | None
) -> None:
    data = json.dumps(build_model_arena_payload(results, exports), allow_nan=False)
    st.iframe(
        _document().replace("/* DATA */", data.replace("<", r"\u003c")),
        tab_index=0,
        alt="3D held-out model comparison with accessible error scores and per-engine inspection",
    )
