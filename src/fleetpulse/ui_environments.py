"""Render the new subset prediction flow inside the existing RUL page."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import streamlit as st

from fleetpulse import ui
from fleetpulse.data.input_validation import parse_fd001_csv_upload
from fleetpulse.data.loader import load_engine_history
from fleetpulse.services.environments import (
    FEATURES,
    NEW_ENVIRONMENTS,
    group_histories,
    load_environment_model,
    sha256,
)
from fleetpulse.services.history import MAX_REPLAY_POINTS
from fleetpulse.ui_forecast_scene import render_prediction_trace


@st.cache_resource(max_entries=6)
def _model(directory: Path, subset: str, bundle_identity: tuple[str, str]):
    # Both fingerprints are in the cache key so changed artifacts reload.
    return load_environment_model(directory, subset)


@st.cache_data(max_entries=8)
def _histories(path: Path, file_sha256: str):
    return group_histories(load_engine_history(path))


def _clear_planner():
    for key in (
        ui.P15_SELECTED_FLEET_STATE_KEY,
        ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY,
        ui.P20_FLEET_SCENARIO_STATE_KEY,
        ui.P20_FLEET_COMPARISON_STATE_KEY,
        ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY,
    ):
        st.session_state.pop(key, None)


def _take_environment_handoff():
    identity = st.session_state.pop("fp_env_planner_identity", None)
    if identity is None:
        return None
    _, source_hash, engine, cutoff, bundle, partition = identity
    provenance = st.session_state.get(ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY, {})
    selected = st.session_state.get(ui.P15_SELECTED_FLEET_STATE_KEY, [])
    if (
        len(selected) == 1
        and selected[0].get("Dataset") == partition
        and selected[0].get("Engine ID") == engine
        and selected[0].get("Observed cycle") == cutoff
        and provenance.get("source", {}).get("source_file_sha256") == source_hash
        and provenance.get("model", {}).get("manifest_sha256") == bundle[0]
    ):
        return identity
    return None


def clear_environment_handoff():
    if _take_environment_handoff() is not None:
        _clear_planner()


def render_environment_prediction(subset, render_estimate):
    root = ui.PROJECT_ROOT
    directory = root / "artifacts" / "environments" / subset
    source_column, model_column = st.container(key="fp_rul_controls_environment").columns(2)
    options = ["Packaged sample", "Local training data", "Upload CSV"]
    source = source_column.selectbox("History source", options, key=f"{subset}_source")
    model_column.text_input("RUL model", value=f"XGBoost · {subset}", disabled=True)
    previous_identity = _take_environment_handoff()
    try:
        _predict(subset, directory, source, render_estimate, previous_identity)
    finally:
        if previous_identity is not None and "fp_env_planner_identity" not in st.session_state:
            _clear_planner()


def _predict(subset, directory, source, render_estimate, previous_identity):
    try:
        bundle_identity = (sha256(directory / "manifest.json"), sha256(directory / "model.json"))
        model = _model(directory, subset, bundle_identity)
    except (OSError, ImportError, ValueError, KeyError, TypeError) as error:
        st.error(f"{subset} model unavailable. {error}")
        return
    root = ui.PROJECT_ROOT
    try:
        if source == "Upload CSV":
            st.caption(f"Upload observed {subset} readings with the 26 raw columns (up to 1 MiB).")
            upload = st.file_uploader(
                "Engine history CSV", type=["csv"], max_upload_size=1, key=f"{subset}_upload"
            )
            if upload is None:
                return
            contents = upload.getvalue()
            histories = group_histories(parse_fd001_csv_upload(contents))
            source_hash = hashlib.sha256(contents).hexdigest()
            source_path = upload.name
            partition = f"{subset}/upload"
        else:
            if source == "Packaged sample":
                path = root / "data" / "demo" / f"{subset.lower()}_training_prefix.txt"
                st.caption(f"Packaged simulated {subset} training prefix.")
            else:
                filename = f"train_{subset}.txt"
                path = root / "data" / "raw" / filename
                if not path.is_file():
                    path = root / "project data set" / filename
            source_hash = sha256(path)
            histories = _histories(path, source_hash)
            source_path = path.name
            partition = f"{subset}/train"
    except (OSError, ValueError) as error:
        st.error(f"Cannot use this {subset} history. {error}")
        return
    control = st.container(key="fp_rul_controls_history")
    engine_column, cycle_column = control.columns(2)
    engine = engine_column.selectbox(
        "Engine ID", sorted(histories), key=f"{subset}_{source_hash}_engine"
    )
    rows = histories[engine]
    cycles = [row["cycle"] for row in rows]
    cutoff = cycle_column.selectbox(
        "Observed cycle",
        cycles,
        index=len(cycles) - 1,
        key=f"{subset}_{source_hash}_{engine}_cycle",
    )
    visible = [row for row in rows if row["cycle"] <= cutoff]
    identity = (subset, source_hash, engine, cutoff, bundle_identity, partition)
    if previous_identity == identity:
        st.session_state["fp_env_planner_identity"] = identity
    elif previous_identity is not None:
        _clear_planner()
    step = max(1, (len(visible) - 1 + MAX_REPLAY_POINTS - 2) // (MAX_REPLAY_POINTS - 1))
    cutoffs = [row["cycle"] for row in visible[:-1:step]] + [cutoff]
    try:
        predictions = model.predict(visible, cutoffs)
    except (ValueError, RuntimeError) as error:
        clear_environment_handoff()
        st.error(f"{subset} prediction failed. {error}")
        return
    vector, raw = predictions[-1]
    displayed = max(0.0, raw)
    render_estimate(
        engine,
        cutoff,
        displayed,
        f"{subset} XGBoost",
        model.version,
        history_id=f"{subset}:{source_hash}",
    )
    render_prediction_trace(
        [(v.selected_cycle, max(0.0, value)) for v, value in predictions], engine=engine
    )
    warnings = []
    if vector.window_row_count < FEATURES.window_cycles:
        warnings.append("short_window")
    if any(b - a > 1 for a, b in pairwise(cycles[: len(visible)])):
        warnings.append("cycle_gaps")
    if raw < 0:
        warnings.append("display_clipped_at_zero")
    if warnings:
        st.warning("Prediction notes · " + ", ".join(item.replace("_", " ") for item in warnings))
    with st.expander("Prediction details"):
        st.write(
            f"{partition} · {vector.window_row_count} feature rows · Raw estimate {raw:.4f} cycles"
        )
        st.caption("Simulated engine estimate. Use the environment that matches your history.")
    if st.button("Save this prediction for Maintenance Planner", key=f"{subset}_save"):
        _clear_planner()
        st.session_state["fp_env_planner_identity"] = identity
        st.session_state[ui.P15_SELECTED_FLEET_STATE_KEY] = [
            {
                "Dataset": partition,
                "Engine ID": engine,
                "Observed cycle": cutoff,
                "Predicted RUL (cycles)": displayed,
                "Raw RUL (cycles)": raw,
                "Model": "XGBoost",
                "Version": model.version,
                "Feature rows": vector.window_row_count,
                "Warnings": ", ".join(warnings),
                "Error": None,
            }
        ]
        st.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY] = {
            "source": {
                "source_name": f"{subset} {source}",
                "source_path": source_path,
                "source_file_sha256": source_hash,
                "source_record_count": len(visible),
                "source_records_sha256": hashlib.sha256(
                    json.dumps(visible, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "model": {
                "model_name": "XGBoost",
                "model_version": model.version,
                "manifest_sha256": bundle_identity[0],
                "artifact_sha256": bundle_identity[1],
            },
        }
        st.success("Saved for Maintenance Planner.")
    if st.session_state.get("fp_env_planner_identity") == identity:
        st.page_link(
            st.Page(root / "app" / "modules" / "4_Maintenance_Planner.py"),
            label="Open Maintenance Planner",
        )


def render_environment_scores():
    root = ui.PROJECT_ROOT / "artifacts" / "environments"
    with st.expander("Official test results · FD002–FD004"):
        rows = []
        for subset in NEW_ENVIRONMENTS:
            directory = root / subset
            try:
                report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
                if (
                    report["subset"] != subset
                    or report["partition"] != f"{subset}/test"
                    or report["bundle_sha256"] != sha256(directory / "manifest.json")
                    or report["model_sha256"] != sha256(directory / "model.json")
                ):
                    raise ValueError("Evaluation provenance does not match the served model.")
                for kind, name in (("model", "XGBoost"), ("constant_mean", "Constant mean")):
                    scores = report[kind]
                    rows.append(
                        {
                            "Subset": subset,
                            "Model": name,
                            "Model version": report["model_version"] if kind == "model" else None,
                            "Engines": scores["engines"],
                            "MAE (cycles)": scores["mae_cycles"],
                            "RMSE (cycles)": scores["rmse_cycles"],
                            "Mean signed error (cycles)": scores["mean_signed_error_cycles"],
                        }
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.info(f"{subset} results unavailable. {error}")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("One prediction per simulated test engine at its last observed cycle.")
