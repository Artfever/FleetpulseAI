"""Predict remaining cycles for a selected simulated training-engine history."""

import json

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from fleetpulse import ui
from fleetpulse.data.input_validation import UploadValidationError, parse_fd001_csv_upload
from fleetpulse.data.loader import FD001LoadError, load_engine_history
from fleetpulse.data.quality import validate_engine_history
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
    predict_rul_with_context,
)

st.title("RUL Prediction")
st.caption(
    "Estimate remaining useful life in operating cycles using an available saved development model. "
    "This page does not load a final refitted model."
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
selected_model = st.selectbox("RUL model", model_options)

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
            st.info("CSV prediction is not available yet. This view validates and previews the upload.")
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
    cycle = st.selectbox("Observed cycle", [int(row["cycle"]) for row in history])
    st.caption(
        "Only readings through the selected cycle are used; this is not a maintenance guarantee."
    )
    if st.button("Predict remaining cycles"):
        prefix = [row for row in history if row["cycle"] <= cycle]
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
                result = predict_rul_with_context(
                    prefix,
                    dataset_partition=partition,
                    unit_id=engine,
                    manifest_path=active_manifest,
                )
        except (PredictionArtifactError, PredictionInputError) as error:
            st.warning(f"Prediction is unavailable: {error}")
            st.write("Check the saved development model bundle and the selected training history.")
        else:
            st.metric("Estimated remaining cycles", f"{result.displayed_rul_cycles:.2f}")
            st.write(f"Model: {result.model_name} ({result.model_version})")
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
else:
    st.write("Choose an engine once the selected history is available.")
