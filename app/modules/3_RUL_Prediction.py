"""Predict remaining cycles for a selected simulated training-engine history."""

import hashlib
import json
import pickle
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import ui
from fleetpulse.data.input_validation import UploadValidationError, parse_fd001_csv_upload
from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.data.quality import SENSOR_COLUMNS, validate_engine_history
from fleetpulse.deployment import prediction_manifest
from fleetpulse.models.forest_artifact import (
    MANIFEST_FILENAME as FOREST_MANIFEST_FILENAME,
)
from fleetpulse.models.forest_artifact import (
    MODEL_FILENAME as FOREST_MODEL_FILENAME,
)
from fleetpulse.models.mlp_contract import (
    MANIFEST_FILENAME as MLP_MANIFEST_FILENAME,
)
from fleetpulse.models.mlp_contract import (
    WEIGHTS_FILENAME as MLP_WEIGHTS_FILENAME,
)
from fleetpulse.models.usage_scope import classify_training_engine
from fleetpulse.models.xgboost_artifact import (
    MANIFEST_FILENAME as XGBOOST_MANIFEST_FILENAME,
)
from fleetpulse.models.xgboost_artifact import (
    MODEL_FILENAME as XGBOOST_MODEL_FILENAME,
)
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    compare_final_models,
    explain_final_tree_history,
    model_provenance_from_manifest,
    predict_uploaded_csv,
)
from fleetpulse.services.history import MAX_REPLAY_POINTS, replay_engine_history
from fleetpulse.ui_forecast_scene import render_prediction_trace

_app_dir = Path(__file__).resolve().parents[1]
st.html(_app_dir / "rul_prediction.css")
with st.container(key="fp_rul_header"):
    st.title("RUL Prediction", anchor=False)

source_column, model_column = st.container(key="fp_rul_controls_source").columns(2)

default_source = None
if (
    not (ui.PROJECT_ROOT / "data" / "raw" / "train_FD001.txt").is_file()
    and (ui.PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt").is_file()
):
    default_source = "Packaged FD001/train parity subset"
ui.prepare_selection(
    ui.P21_HISTORY_SOURCE_STATE_KEY,
    ui.P21_HISTORY_SOURCE_WIDGET_KEY,
    [
        "Local FD001 training data",
        "Packaged FD001/train parity subset",
        "Illustrative synthetic example",
        "Upload CSV",
    ],
    default=default_source,
)
source = source_column.selectbox(
    "History source",
    [
        "Local FD001 training data",
        "Packaged FD001/train parity subset",
        "Illustrative synthetic example",
        "Upload CSV",
    ],
    key=ui.P21_HISTORY_SOURCE_WIDGET_KEY,
    on_change=ui.remember_selection,
    args=(ui.P21_HISTORY_SOURCE_STATE_KEY, ui.P21_HISTORY_SOURCE_WIDGET_KEY),
)

artifact_dir = ui.PROJECT_ROOT / "artifacts" / "development"
tree_manifest = artifact_dir / "decision_tree_validation.json"
forest_manifest = artifact_dir / FOREST_MANIFEST_FILENAME
forest_model = artifact_dir / FOREST_MODEL_FILENAME
mlp_artifact_dir = artifact_dir / "mlp"
mlp_manifest = mlp_artifact_dir / MLP_MANIFEST_FILENAME
mlp_weights = mlp_artifact_dir / MLP_WEIGHTS_FILENAME
xgboost_manifest = artifact_dir / XGBOOST_MANIFEST_FILENAME
xgboost_model = artifact_dir / XGBOOST_MODEL_FILENAME
model_options = ["Decision Tree"]
if forest_manifest.is_file() and forest_model.is_file():
    model_options.append("Random Forest")
if xgboost_manifest.is_file() and xgboost_model.is_file():
    model_options.append("XGBoost")
if mlp_manifest.is_file() and mlp_weights.is_file():
    model_options.append("Dense MLP (CPU)")
ui.prepare_selection(
    ui.P21_SELECTED_MODEL_STATE_KEY,
    ui.P21_SELECTED_MODEL_WIDGET_KEY,
    model_options,
)
selected_model = model_column.selectbox(
    "RUL model",
    model_options,
    key=ui.P21_SELECTED_MODEL_WIDGET_KEY,
    on_change=ui.remember_selection,
    args=(ui.P21_SELECTED_MODEL_STATE_KEY, ui.P21_SELECTED_MODEL_WIDGET_KEY),
)


def _render_estimate(
    engine: int,
    cycle: int,
    value: float,
    model_name: str,
    model_version: str,
    *,
    history_id: str,
) -> None:
    """Restart the card's reveal when its prediction context changes."""
    signature = (history_id, engine, cycle, value, model_name, model_version)
    card_id = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    with st.container(key=f"fp_rul_estimate_{card_id}"):
        st.metric("Estimated remaining cycles", f"{value:.2f}")
        st.caption(f"Engine {engine} · Cycle {cycle} · {model_name} {model_version}")


def _clear_saved_upload_from_planner() -> None:
    """Invalidate an uploaded planner snapshot and its derived comparisons."""
    selected_rows = st.session_state.get(ui.P15_SELECTED_FLEET_STATE_KEY, [])
    if not any(row.get("Dataset") == "user/upload" for row in selected_rows):
        return
    st.session_state.pop(ui.P15_SELECTED_FLEET_STATE_KEY, None)
    st.session_state.pop(ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY, None)
    st.session_state.pop(ui.P20_FLEET_SCENARIO_STATE_KEY, None)
    st.session_state.pop(ui.P20_FLEET_COMPARISON_STATE_KEY, None)
    st.session_state.pop(ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY, None)


partition = "FD001/train"
if source != "Upload CSV":
    st.session_state.pop(ui.P21_UPLOADED_PREDICTION_STATE_KEY, None)
    _clear_saved_upload_from_planner()
if source == "Local FD001 training data":
    records = ui.render_training_data(compact=True)
elif source == "Packaged FD001/train parity subset":
    records = ui.render_packaged_training_subset(compact=True)
elif source == "Upload CSV":
    records = None
    st.caption("Upload an FD001 history CSV (up to 1 MiB).")
    uploaded_file = st.file_uploader("Engine history CSV", type=["csv"], max_upload_size=1)
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        try:
            uploaded_records = parse_fd001_csv_upload(upload_bytes)
        except UploadValidationError as error:
            st.session_state.pop(ui.P21_UPLOADED_PREDICTION_STATE_KEY, None)
            _clear_saved_upload_from_planner()
            st.error(f"CSV upload rejected ({error.code}): {error}")
        else:
            engine_ids = sorted({int(row["unit_id"]) for row in uploaded_records})
            selected_engine = st.selectbox("Uploaded engine ID", engine_ids)
            selected_history = [
                row for row in uploaded_records if int(row["unit_id"]) == selected_engine
            ]
            configured_manifest = {
                "Decision Tree": tree_manifest,
                "Random Forest": forest_manifest,
                "XGBoost": xgboost_manifest,
                "Dense MLP (CPU)": mlp_manifest,
            }[selected_model]
            configured_artifact = {
                "Decision Tree": artifact_dir / "decision_tree_model.joblib",
                "Random Forest": artifact_dir / FOREST_MODEL_FILENAME,
                "XGBoost": artifact_dir / XGBOOST_MODEL_FILENAME,
                "Dense MLP (CPU)": mlp_weights,
            }[selected_model]
            manifest_identity = (
                hashlib.sha256(configured_manifest.read_bytes()).hexdigest()
                if configured_manifest.is_file()
                else None
            )
            artifact_identity = (
                hashlib.sha256(configured_artifact.read_bytes()).hexdigest()
                if configured_artifact.is_file()
                else None
            )
            upload_identity = {
                "upload_sha256": hashlib.sha256(upload_bytes).hexdigest(),
                "engine_id": selected_engine,
                "model": selected_model,
                "configured_manifest_sha256": manifest_identity,
                "configured_artifact_sha256": artifact_identity,
            }
            saved_upload_prediction = st.session_state.get(
                ui.P21_UPLOADED_PREDICTION_STATE_KEY
            )
            if (
                not isinstance(saved_upload_prediction, dict)
                or saved_upload_prediction.get("identity") != upload_identity
            ):
                _clear_saved_upload_from_planner()
                st.session_state.pop(ui.P21_UPLOADED_PREDICTION_STATE_KEY, None)
                saved_upload_prediction = None
            st.success(f"CSV ready · engine {selected_engine}")
            upload_report = validate_engine_history(selected_history)
            for finding in upload_report.warnings:
                st.warning(finding.message)
            preview_rows = selected_history[:10]
            with st.expander("Preview uploaded rows"):
                st.caption(f"{len(selected_history)} of {len(uploaded_records)} rows · first 10 shown")
                st.dataframe(preview_rows, hide_index=True, width="stretch")
            if st.button("Predict remaining cycles", key="predict_uploaded_csv"):
                try:
                    supplied = None
                    if selected_model == "Decision Tree" and not configured_manifest.exists():
                        try:
                            supplied = st.secrets.get("deployment_model")
                        except StreamlitSecretNotFoundError:
                            pass
                    with prediction_manifest(configured_manifest, supplied) as active_manifest:
                        result = predict_uploaded_csv(
                            upload_bytes,
                            dataset_partition="user/upload",
                            unit_id=selected_engine,
                            manifest_path=active_manifest,
                        )
                        try:
                            model_provenance = model_provenance_from_manifest(active_manifest)
                        except ValueError as error:
                            model_provenance = None
                            provenance_error = str(error)
                        else:
                            provenance_error = None
                except (PredictionArtifactError, PredictionInputError) as error:
                    st.warning(f"Prediction is unavailable: {error}")
                    st.write("Check the saved development model bundle and selected upload.")
                else:
                    if provenance_error:
                        st.warning(f"Planner handoff is unavailable: {provenance_error}")
                    canonical_selected = json.dumps(
                        selected_history, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                    source_provenance = {
                        "source_name": "Uploaded FD001 sensor history",
                        "source_path": uploaded_file.name,
                        "dataset_partition": "user/upload",
                        "source_records_sha256": hashlib.sha256(canonical_selected).hexdigest(),
                        "source_record_count": len(selected_history),
                        "source_file_sha256": upload_identity["upload_sha256"],
                    }
                    st.session_state[ui.P21_UPLOADED_PREDICTION_STATE_KEY] = {
                        "identity": upload_identity,
                        "prediction": {
                            "dataset": "user/upload",
                            "engine_id": selected_engine,
                            "observed_cycle": result.observed_cycle,
                            "displayed_rul_cycles": result.displayed_rul_cycles,
                            "raw_rul_cycles": result.raw_rul_cycles,
                            "model_name": result.model_name,
                            "model_version": result.model_version,
                            "input_window_rows": result.input_window_rows,
                            "warnings": result.warnings,
                        },
                        "source_provenance": source_provenance,
                        "model_provenance": model_provenance,
                    }
                    saved_upload_prediction = st.session_state[
                        ui.P21_UPLOADED_PREDICTION_STATE_KEY
                    ]

            if saved_upload_prediction is not None:
                result = saved_upload_prediction["prediction"]
                _render_estimate(
                    selected_engine,
                    result["observed_cycle"],
                    result["displayed_rul_cycles"],
                    result["model_name"],
                    result["model_version"],
                    history_id=upload_identity["upload_sha256"],
                )
                render_prediction_trace(
                    [(result["observed_cycle"], result["displayed_rul_cycles"])],
                    engine=selected_engine,
                )
                with st.expander("Prediction details"):
                    st.caption(
                        f"{result['input_window_rows']} rows used · "
                        f"Raw estimate: {result['raw_rul_cycles']:.2f} cycles · "
                        "Uploaded history fit membership is not verified."
                    )
                    st.caption(
                        "FD001 estimates are software demonstrations, not maintenance advice; "
                        "other fleets or operating conditions may be outside the model's domain."
                    )
                if "short_window" in result["warnings"]:
                    st.warning(
                        "The available feature window is short; the estimate uses fewer observations."
                    )
                if "cycle_gaps" in result["warnings"]:
                    st.warning("The observed history contains gaps between cycles.")
                if "display_clipped_at_zero" in result["warnings"]:
                    st.warning(
                        "The displayed estimate is clipped to zero; the raw estimate is shown above."
                    )
                if st.button(
                    "Save this prediction for Maintenance Planner",
                    disabled=saved_upload_prediction["model_provenance"] is None,
                ):
                    st.session_state[ui.P15_SELECTED_FLEET_STATE_KEY] = [
                        {
                            "Dataset": result["dataset"],
                            "Engine ID": result["engine_id"],
                            "Observed cycle": result["observed_cycle"],
                            "Predicted RUL (cycles)": result["displayed_rul_cycles"],
                            "Raw RUL (cycles)": result["raw_rul_cycles"],
                            "Model": result["model_name"],
                            "Version": result["model_version"],
                            "Feature rows": result["input_window_rows"],
                            "Warnings": ", ".join(result["warnings"]),
                            "Error": None,
                        }
                    ]
                    st.session_state[ui.P15_SELECTED_FLEET_PROVENANCE_STATE_KEY] = {
                        "source": saved_upload_prediction["source_provenance"],
                        "model": saved_upload_prediction["model_provenance"],
                    }
                    st.session_state.pop(ui.P20_FLEET_SCENARIO_STATE_KEY, None)
                    st.session_state.pop(ui.P20_FLEET_COMPARISON_STATE_KEY, None)
                    st.session_state.pop(ui.P20_FLEET_COMPARISON_ARTIFACT_STATE_KEY, None)
                    st.success("Saved the uploaded prediction for Maintenance Planner.")
                    st.page_link(st.Page(_app_dir / "modules" / "4_Maintenance_Planner.py"), label="Open Maintenance Planner")
    else:
        st.session_state.pop(ui.P21_UPLOADED_PREDICTION_STATE_KEY, None)
        _clear_saved_upload_from_planner()
else:
    partition = "illustrative/synthetic"
    st.caption("Illustrative synthetic data")
    try:
        records = load_engine_history(ui.PROJECT_ROOT / "data/demo/illustrative_history.txt")
        if not validate_engine_history(records).is_valid:
            raise FD001LoadError("The illustrative example has invalid history records.")
    except FD001LoadError as error:
        st.error(f"Illustrative example unavailable: {error}")
        records = None

if records is not None:
    engine_column, cycle_column = st.container(key="fp_rul_controls_engine").columns(2)
    engine_ids = sorted({int(row["unit_id"]) for row in records})
    ui.prepare_selection(
        ui.P21_SELECTED_ENGINE_STATE_KEY,
        ui.P21_SELECTED_ENGINE_PREDICTION_WIDGET_KEY,
        engine_ids,
    )
    engine = engine_column.selectbox(
        "Engine ID",
        engine_ids,
        key=ui.P21_SELECTED_ENGINE_PREDICTION_WIDGET_KEY,
        on_change=ui.remember_selection,
        args=(
            ui.P21_SELECTED_ENGINE_STATE_KEY,
            ui.P21_SELECTED_ENGINE_PREDICTION_WIDGET_KEY,
        ),
    )
    history = sorted(
        (row for row in records if row["unit_id"] == engine), key=lambda row: row["cycle"]
    )
    training_usage = "unknown"
    if partition == "FD001/train":
        split_path = ui.PROJECT_ROOT / "artifacts" / "split_manifest.json"
        try:
            split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
            usage = classify_training_engine(
                split_manifest, dataset_partition=partition, unit_id=engine
            )
        except (OSError, ValueError, TypeError):
            st.warning("Training split unavailable; holdout status is unknown.")
        else:
            training_usage = usage
            if usage == "fitting":
                st.warning("In-sample estimate · this engine was used to fit the model.")
            elif usage == "validation":
                st.caption("Development holdout · not an official test result.")
            else:
                st.warning("Engine fit status unknown · not held-out evidence.")
    else:
        st.warning("Synthetic demo only · not a real engine or maintenance result.")
    cycle_options = [int(row["cycle"]) for row in history]
    ui.prepare_selection(
        ui.P21_SELECTED_CYCLE_STATE_KEY,
        ui.P21_SELECTED_CYCLE_PREDICTION_WIDGET_KEY,
        cycle_options,
        default=cycle_options[-1],
    )
    cycle = cycle_column.select_slider(
        "Observed cycle",
        options=cycle_options,
        key=ui.P21_SELECTED_CYCLE_PREDICTION_WIDGET_KEY,
        on_change=ui.remember_selection,
        args=(
            ui.P21_SELECTED_CYCLE_STATE_KEY,
            ui.P21_SELECTED_CYCLE_PREDICTION_WIDGET_KEY,
        ),
    )
    estimate_slot = st.container()
    visible_history = [row for row in history if int(row["cycle"]) <= cycle]
    with st.expander("Sensor history", expanded=False):
        sensor = st.selectbox("Sensor", SENSOR_COLUMNS)
        st.line_chart(
            {
                "cycle": [int(row["cycle"]) for row in visible_history],
                sensor: [float(row[sensor]) for row in visible_history],
            },
            x="cycle",
            y=sensor,
            y_label=sensor,
            x_label="Cycle",
        )
        st.caption(f"{sensor} through cycle {cycle}")
    replay_point_count = min(len(visible_history), MAX_REPLAY_POINTS)
    if replay_point_count == 1:
        replay_cutoffs = [cycle]
    else:
        last_index = len(visible_history) - 1
        replay_cutoffs = [
            int(visible_history[(point * last_index) // (replay_point_count - 1)]["cycle"])
            for point in range(replay_point_count)
        ]
    manifest = {
        "Decision Tree": tree_manifest,
        "Random Forest": forest_manifest,
        "XGBoost": xgboost_manifest,
        "Dense MLP (CPU)": mlp_manifest,
    }[selected_model]
    try:
        supplied = None
        if selected_model == "Decision Tree" and not manifest.exists():
            try:
                supplied = st.secrets.get("deployment_model")
            except StreamlitSecretNotFoundError:
                pass
        with prediction_manifest(manifest, supplied) as active_manifest:
            replay = replay_engine_history(
                records,
                dataset_partition=partition,
                unit_id=engine,
                selected_cycle=cycle,
                replay_cutoffs=replay_cutoffs,
                manifest_path=active_manifest,
            )
            result = replay.points[-1].prediction
    except (PredictionArtifactError, PredictionInputError) as error:
        estimate_slot.warning(f"Prediction is unavailable: {error}")
    else:
        with estimate_slot:
            _render_estimate(
                engine,
                result.observed_cycle,
                result.displayed_rul_cycles,
                result.model_name,
                result.model_version,
                history_id=source,
            )
            render_prediction_trace(
                [(point.cutoff_cycle, point.prediction.displayed_rul_cycles) for point in replay.points],
                engine=engine,
            )
        with st.expander("Prediction details"):
            st.caption(
                f"{partition} · {result.input_window_rows} rows used · "
                f"Raw estimate: {result.raw_rul_cycles:.2f} cycles"
            )
            if partition == "FD001/train":
                membership = {
                    "fitting": "Seen during fitting",
                    "validation": "Not seen during fitting (development validation engine)",
                    "unknown": "Unknown (not listed or split unavailable)",
                }[training_usage]
                st.caption(f"Engine fit membership: {membership}")
            st.caption(
                "FD001 is simulated; this estimate is not a fault diagnosis or maintenance advice."
            )
        timeline_rows = [
            {
                "Observed cycle": point.cutoff_cycle,
                "Estimated remaining cycles": point.prediction.displayed_rul_cycles,
                "Raw estimate (cycles)": point.prediction.raw_rul_cycles,
                "Warnings": ", ".join(point.prediction.warnings),
            }
            for point in replay.points
        ]
        with st.expander("Prediction over time"):
            st.line_chart(
                {
                    "Observed cycle": [row["Observed cycle"] for row in timeline_rows],
                    "Estimated remaining cycles": [
                        row["Estimated remaining cycles"] for row in timeline_rows
                    ],
                },
                x="Observed cycle",
                y="Estimated remaining cycles",
                x_label="Observed cycle",
                y_label="Estimated remaining cycles",
            )
            st.dataframe(timeline_rows, hide_index=True, width="stretch")
        if "short_window" in result.warnings:
            st.warning(
                "The available feature window is short; the estimate uses fewer observations."
            )
        if "cycle_gaps" in result.warnings:
            st.warning("The observed history contains gaps between cycles.")
        if "display_clipped_at_zero" in result.warnings:
            st.warning(
                "The displayed estimate is clipped to zero; the raw estimate is shown above."
            )
        with st.expander("Explore model explanation"):
            if st.button("Show feature contributions", key="show_final_tree_contributions"):
                try:
                    with st.spinner("Checking the frozen model and calculating contributions..."):
                        final_explanation = explain_final_tree_history(
                            visible_history,
                            dataset_partition=partition,
                            unit_id=engine,
                            artifact_root=ui.PROJECT_ROOT,
                            training_path=ui.PROJECT_ROOT / "data" / "raw" / "train_FD001.txt",
                        )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                    KeyError,
                    EOFError,
                    ImportError,
                    pickle.UnpicklingError,
                ) as error:
                    st.info(f"Final Decision Tree explanation is unavailable: {error}")
                else:
                    st.write(
                        f"Frozen model: `{final_explanation.model_name}` "
                        f"({final_explanation.model_version}); engine {engine}, "
                        f"observed cycle {final_explanation.observed_cycle}."
                    )
                    st.metric("Final raw RUL estimate", f"{final_explanation.raw_rul_cycles:.2f} cycles")
                    st.metric("SHAP reference baseline", f"{final_explanation.base_value_cycles:.2f} cycles")
                    positive = sorted(
                        (item for item in final_explanation.contributions if item.contribution_cycles > 0),
                        key=lambda item: -item.contribution_cycles,
                    )[:5]
                    negative = sorted(
                        (item for item in final_explanation.contributions if item.contribution_cycles < 0),
                        key=lambda item: item.contribution_cycles,
                    )[:5]
                    if positive:
                        st.write("Largest positive contributions")
                        st.dataframe(
                            [
                                {
                                    "Feature": item.feature_name,
                                    "Feature value": item.feature_value,
                                    "Contribution (cycles)": item.contribution_cycles,
                                }
                                for item in positive
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("No positive feature contributions for this explanation.")
                    if negative:
                        st.write("Largest negative contributions")
                        st.dataframe(
                            [
                                {
                                    "Feature": item.feature_name,
                                    "Feature value": item.feature_value,
                                    "Contribution (cycles)": item.contribution_cycles,
                                }
                                for item in negative
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("No negative feature contributions for this explanation.")
                    st.caption(
                        "Baseline plus all signed contributions reconstructs the raw estimate within "
                        f"{final_explanation.additivity_error_cycles:.2g} cycles. "
                        "The result depends on the documented FD001/train reference sample and does not "
                        "measure model accuracy or confidence."
                    )
            if st.button("Compare frozen final models", key="compare_frozen_final_models"):
                try:
                    with st.spinner("Loading verified frozen models and comparing raw estimates..."):
                        comparison = compare_final_models(
                            visible_history,
                            dataset_partition=partition,
                            unit_id=engine,
                            artifact_root=ui.PROJECT_ROOT,
                        )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                    KeyError,
                    EOFError,
                    ImportError,
                    OverflowError,
                    pickle.UnpicklingError,
                ) as error:
                    st.info(f"Frozen model comparison is unavailable: {error}")
                else:
                    st.write(
                        f"{comparison.dataset_partition} engine {comparison.unit_id}, "
                        f"observed cycle {comparison.observed_cycle}. Successful estimates are raw "
                        "and unclipped."
                    )
                    if comparison.predictions:
                        st.dataframe(
                            [
                                {
                                    "Model": prediction.model_name,
                                    "Version": prediction.model_version,
                                    "Raw estimate (cycles)": prediction.raw_rul_cycles,
                                }
                                for prediction in comparison.predictions
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                        spread = comparison.raw_spread_cycles
                        if spread is None:
                            st.info("Model disagreement is unavailable for these results.")
                        else:
                            st.metric("Raw model disagreement (max − min)", f"{spread:.2f} cycles")
                            st.caption(
                                f"Successful-model raw range: {comparison.raw_min_rul_cycles:.2f}–"
                                f"{comparison.raw_max_rul_cycles:.2f} cycles. This range describes "
                                "model disagreement, not prediction uncertainty or confidence."
                            )
                    else:
                        st.info("No frozen final model produced a usable estimate; no spread is available.")
                    if comparison.failures:
                        st.write("Unavailable or incompatible models")
                        st.dataframe(
                            [
                                {
                                    "Model": failure.model_name,
                                    "Status": failure.code,
                                    "Reason": failure.reason,
                                }
                                for failure in comparison.failures
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(
                            "Failed models are listed separately and are excluded from the raw range."
                        )
else:
    if source != "Upload CSV":
        st.write("Choose an engine once the selected history is available.")

with st.expander("Related pages"):
    st.page_link(st.Page(_app_dir / "modules" / "1_Fleet_Overview.py"), label="Continue to Fleet Overview")
    st.page_link(st.Page(_app_dir / "modules" / "2_Engine_Explorer.py"), label="Return to Engine Explorer")
