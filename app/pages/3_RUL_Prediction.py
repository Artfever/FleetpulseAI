"""Predict remaining cycles for a selected simulated training-engine history."""

import json
import pickle

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
from fleetpulse.models.registry import DEVELOPMENT_MODEL_REGISTRY, FINAL_MODEL_REGISTRY
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
    predict_uploaded_csv,
)
from fleetpulse.services.history import MAX_REPLAY_POINTS, replay_engine_history

st.title("RUL Prediction")
st.caption(
    "Estimate remaining useful life in operating cycles using a saved development model. "
    "On request, the separate contribution panel explains the frozen final Decision Tree."
)

source = st.selectbox(
    "History source",
    [
        "Local FD001 training data",
        "Packaged FD001/train parity subset",
        "Illustrative synthetic example",
        "Upload CSV",
    ],
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
else:
    st.info(
        "Random Forest is unavailable in this workspace until its verified model bundle "
        "is present in `artifacts/development/`."
    )
if xgboost_manifest.is_file() and xgboost_model.is_file():
    model_options.append("XGBoost")
else:
    st.info(
        "XGBoost is unavailable in this workspace until its verified model bundle "
        "is present in `artifacts/development/`."
    )
if mlp_manifest.is_file() and mlp_weights.is_file():
    model_options.append("Dense MLP (CPU)")
else:
    st.info(
        "Dense MLP is unavailable in this workspace until its verified checkpoint, "
        "architecture and scaler are present in `artifacts/development/mlp/`."
    )
development_versions = {
    "Decision Tree": DEVELOPMENT_MODEL_REGISTRY["decision-tree-rul"].version,
    "Random Forest": DEVELOPMENT_MODEL_REGISTRY["random-forest-rul"].version,
    "XGBoost": DEVELOPMENT_MODEL_REGISTRY["xgboost-rul"].version,
    "Dense MLP (CPU)": DEVELOPMENT_MODEL_REGISTRY["dense-rul-mlp"].version,
}
final_registry_names = {
    "Decision Tree": "decision-tree-rul",
    "Random Forest": "random-forest-rul",
    "XGBoost": "xgboost-rul",
    "Dense MLP (CPU)": "dense-rul-mlp-final",
}
selected_model = st.selectbox(
    "RUL model",
    model_options,
    format_func=lambda name: f"{name} · Development · {development_versions[name]}",
)
final_model_spec = FINAL_MODEL_REGISTRY[final_registry_names[selected_model]]
st.caption(
    f"Selected: Development candidate `{development_versions[selected_model]}`. "
    f"Final refit `{final_model_spec.version}` is separate; the optional contribution panel uses "
    "the frozen final Decision Tree without replacing this development estimate."
)

partition = "FD001/train"
if source == "Local FD001 training data":
    records = ui.render_training_data()
elif source == "Packaged FD001/train parity subset":
    records = ui.render_packaged_training_subset()
elif source == "Upload CSV":
    records = None
    st.caption(
        "Upload a headered CSV with the 26 raw-history fields. Target/RUL columns are not accepted; "
        "files are limited to 1 MiB."
    )
    uploaded_file = st.file_uploader("Engine history CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_records = parse_fd001_csv_upload(uploaded_file.getvalue())
        except UploadValidationError as error:
            st.error(f"CSV upload rejected ({error.code}): {error}")
        else:
            engine_ids = sorted({int(row["unit_id"]) for row in uploaded_records})
            selected_engine = st.selectbox("Uploaded engine ID", engine_ids)
            selected_history = [
                row for row in uploaded_records if int(row["unit_id"]) == selected_engine
            ]
            st.success(
                f"CSV passed validation. Showing engine {selected_engine} "
                f"({len(selected_history)} of {len(uploaded_records)} rows)."
            )
            upload_report = validate_engine_history(selected_history)
            for finding in upload_report.warnings:
                st.warning(finding.message)
            preview_rows = selected_history[:10]
            st.caption(
                f"Preview: first {len(preview_rows)} rows for engine {selected_engine}; "
                "the file remains unchanged."
            )
            st.dataframe(preview_rows, hide_index=True, width="stretch")
            st.info(
                "Prediction uses the latest observed cycle for the selected engine and the saved "
                "development model. The upload must follow the FD001 sensor contract; other fleets, "
                "sensor definitions or operating conditions may be outside the model's domain. "
                "This is a software estimate, not maintenance advice."
            )
            if st.button("Predict remaining cycles", key="predict_uploaded_csv"):
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
                        result = predict_uploaded_csv(
                            uploaded_file.getvalue(),
                            dataset_partition="user/upload",
                            unit_id=selected_engine,
                            manifest_path=active_manifest,
                        )
                except (PredictionArtifactError, PredictionInputError) as error:
                    st.warning(f"Prediction is unavailable: {error}")
                    st.write("Check the saved development model bundle and selected upload.")
                else:
                    st.metric("Estimated remaining cycles", f"{result.displayed_rul_cycles:.2f}")
                    st.write(f"Model: {result.model_name} ({result.model_version})")
                    st.caption(
                        f"Uploaded engine {selected_engine}, observed cycle {result.observed_cycle}; "
                        f"{result.input_window_rows} observed rows in the feature window. "
                        f"Raw estimate: {result.raw_rul_cycles:.2f} cycles."
                    )
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
else:
    partition = "illustrative/synthetic"
    st.warning(
        "Artificial example, not NASA data or a real engine history. The model was not "
        "validated on these invented readings; its output is only a software demonstration."
    )
    try:
        records = load_engine_history(ui.PROJECT_ROOT / "data/demo/illustrative_history.txt")
        if not validate_engine_history(records).is_valid:
            raise FD001LoadError("The illustrative example has invalid history records.")
    except FD001LoadError as error:
        st.error(f"Illustrative example unavailable: {error}")
        records = None

if records is not None:
    engine_ids = sorted({int(row["unit_id"]) for row in records})
    engine = st.selectbox("Engine ID", engine_ids)
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
        except (OSError, ValueError, TypeError) as error:
            st.warning(
                "Training-set membership for this model cannot be verified from the saved "
                f"development split ({error}). Do not treat this replay as held-out evidence."
            )
        else:
            training_usage = usage
            if usage == "fitting":
                st.warning(
                    "In-sample illustration: according to the saved development split, this engine "
                    "was used to fit the saved development model. Its prediction is not held-out evidence."
                )
            elif usage == "validation":
                st.info(
                    "Held-out development engine: according to the saved development split, this "
                    "engine was not used to fit the saved development model. This is development "
                    "evidence, not official-test performance."
                )
            else:
                st.warning(
                    "This engine is not listed in the saved development split, so its fit/holdout "
                    "status is unknown. Do not treat this replay as held-out evidence."
                )
    else:
        st.info(
            "Illustrative synthetic example: it is not a NASA engine, held-out evaluation, "
            "or maintenance result."
        )
    cycle_options = [int(row["cycle"]) for row in history]
    cycle = st.select_slider(
        "Observed cycle",
        options=cycle_options,
        value=cycle_options[-1],
    )
    sensor = st.selectbox("Sensor history", SENSOR_COLUMNS)
    visible_history = [row for row in history if int(row["cycle"]) <= cycle]
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
    st.caption(
        f"The {sensor} chart ends at cycle {cycle}; the prediction below uses that same observed prefix."
    )
    st.caption(
        "Only readings through the selected cycle are used; this is not a maintenance guarantee."
    )
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
        st.warning(f"Prediction is unavailable: {error}")
        st.write("Check the saved development model bundle and the selected training history.")
    else:
        timeline_rows = [
            {
                "Observed cycle": point.cutoff_cycle,
                "Estimated remaining cycles": point.prediction.displayed_rul_cycles,
                "Raw estimate (cycles)": point.prediction.raw_rul_cycles,
                "Warnings": ", ".join(point.prediction.warnings),
            }
            for point in replay.points
        ]
        st.subheader("Earlier-cycle prediction timeline")
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
        st.caption(
            f"Timeline contains {len(timeline_rows)} observed cutoffs through cycle {cycle}; "
            "later readings are excluded at every cutoff."
        )
        st.metric("Estimated remaining cycles", f"{result.displayed_rul_cycles:.2f}")
        st.write(f"Model: {result.model_name} ({result.model_version})")
        if partition == "FD001/train":
            membership = {
                "fitting": "Seen during fitting",
                "validation": "Not seen during fitting (development validation engine)",
                "unknown": "Unknown (not listed or split unavailable)",
            }[training_usage]
            st.write(f"Engine fit membership: {membership}")
        else:
            st.write("Engine fit membership: Not applicable to this non-FD001/train example.")
        st.caption(
            f"{partition} engine {engine}, observed cycle {result.observed_cycle}; "
            f"{result.input_window_rows} observed rows in the feature window. "
            f"Raw estimate: {result.raw_rul_cycles:.2f} cycles."
        )
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
        st.subheader("Frozen final Decision Tree feature contributions")
        st.caption(
            "This is a separate explanation from the development estimate above. "
            "Positive SHAP contributions move the final tree estimate above its reference baseline; "
            "negative contributions move it below. Values are in RUL cycles, not causal effects."
        )
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
        st.subheader("Frozen final model comparison")
        st.caption(
            "Compare the frozen final models on this same observed history prefix. "
            "The raw spread describes disagreement between models; it is not a confidence interval."
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
    st.write("Choose an engine once the selected history is available.")
