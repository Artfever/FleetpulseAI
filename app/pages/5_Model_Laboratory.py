"""Model Laboratory view."""

import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fleetpulse import ui
from fleetpulse.data.split_audit_view import verified_split_manifest
from fleetpulse.evaluation.model_comparison import (
    load_model_comparison,
    load_verified_forest_scores,
    load_verified_mlp_scores,
)
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.ui import render_page

render_page(
    "Model Laboratory",
    "Inspect model development and evaluation.",
    "Compare saved development models using verified held-out development results.",
    "This page reports held-out development results only; it does not show official test performance.",
)

st.subheader("Development split audit")
st.caption(
    "NASA FD001 simulated training engines only. Official test engines and answers "
    "are separate and are not used here."
)
try:
    manifest = verified_split_manifest(
        ui.PROJECT_ROOT / "artifacts" / "split_manifest.json",
        ui.PROJECT_ROOT / "data" / "raw" / "train_FD001.txt",
    )
except (OSError, ValueError, KeyError, TypeError) as error:
    st.warning(f"Development split audit is unavailable: {error}")
    st.write("Check the saved manifest and its matching local training file.")
else:
    fitting_engines = manifest["fitting_engines"]
    validation_engines = manifest["validation_prefixes"]
    st.caption(
        f"Verified against the local training source: split seed {manifest['split']['seed']}; "
        f"cutoff seed {manifest['cutoff_policy']['seed']}. "
        "These are development settings, not model results."
    )
    cols = st.columns(2)
    cols[0].metric("Fitting engines", len(fitting_engines))
    cols[1].metric("Validation engines", len(validation_engines))
    st.write("Fitting group (dataset-qualified engine IDs)")
    st.dataframe(fitting_engines, hide_index=True)
    st.write("Validation prefixes and known targets — analysis only")
    st.caption(
        "Prefix length counts observed rows through the cutoff. Target RUL and "
        "full lifetime use hidden future information; never use them as model inputs."
    )
    st.dataframe(validation_engines, hide_index=True)
    targets = [row["target_rul"] for row in validation_engines]
    st.write(f"Validation target RUL range: {min(targets)}–{max(targets)} cycles.")

    artifact_dir = ui.PROJECT_ROOT / "artifacts" / "development"
    try:
        expected_manifest_sha = manifest_digest(manifest)
        feature_config = FeatureConfig()
        fitting_path = artifact_dir / "fitting_features.json"
        validation_path = artifact_dir / "validation_features.json"
        fitting = load_feature_dataset(
            fitting_path,
            expected_source_sha256=manifest["source"]["sha256"],
            expected_manifest_sha256=expected_manifest_sha,
            expected_config=feature_config,
        )
        validation_features = load_feature_dataset(
            validation_path,
            expected_source_sha256=manifest["source"]["sha256"],
            expected_manifest_sha256=expected_manifest_sha,
            expected_config=feature_config,
        )
        fitting_sha = json.loads(fitting_path.read_text(encoding="utf-8"))["sha256"]
        validation_sha = json.loads(validation_path.read_text(encoding="utf-8"))["sha256"]
        comparison = load_model_comparison(
            artifact_dir,
            fitting,
            validation_features,
            fitting_dataset_sha256=fitting_sha,
            validation_dataset_sha256=validation_sha,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.info(
            "Verified model comparison is unavailable. Run `python scripts/run_p06_validation.py` "
            f"from the repository root and check the local FD001 setup. Detail: {error}"
        )
    else:
        forest_scores = None
        try:
            forest_scores = load_verified_forest_scores(
                artifact_dir,
                validation_features,
                fitting_dataset_sha256=fitting_sha,
                validation_dataset_sha256=validation_sha,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(
                "Random Forest scores are not shown because its saved development bundle "
                f"could not be verified: {error}"
            )
        if forest_scores is None:
            st.info(
                "Verified Random Forest scores are unavailable until the matching saved "
                "development model bundle is present."
            )
        mlp_scores = None
        try:
            mlp_scores = load_verified_mlp_scores(
                artifact_dir,
                fitting,
                validation_features,
                fitting_dataset_sha256=fitting_sha,
                validation_dataset_sha256=validation_sha,
            )
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            st.warning(
                "Dense MLP results are not shown because its saved development checkpoint "
                f"could not be verified: {error}"
            )
        if mlp_scores is None:
            st.info(
                "Verified Dense MLP results are unavailable until its saved development "
                "checkpoint, architecture and scaler are present."
            )
        model_names = ["Constant baseline", "Decision Tree"]
        if forest_scores is not None:
            model_names.append("Random Forest")
        if mlp_scores is not None:
            model_names.append("Dense MLP")
        section_title = (
            " and ".join(model_names)
            if len(model_names) == 2
            else f"{', '.join(model_names[:-1])}, and {model_names[-1]}"
        )
        st.subheader(section_title)
        st.caption(
            f"All displayed model scores use the same {comparison.validation_engine_count} "
            "held-out FD001 training engines and saved cutoff cycles. These are "
            "development-validation measurements, not official test results."
        )
        metric_rows = [
            {
                "Model": comparison.baseline_name,
                "MAE (cycles)": comparison.baseline_metrics.mae,
                "RMSE (cycles)": comparison.baseline_metrics.rmse,
            },
            {
                "Model": comparison.tree_name,
                "MAE (cycles)": comparison.tree_metrics.mae,
                "RMSE (cycles)": comparison.tree_metrics.rmse,
            },
        ]
        if forest_scores is not None:
            metric_rows.append(
                {
                    "Model": f"Random Forest ({forest_scores.model_version})",
                    "MAE (cycles)": forest_scores.mae_cycles,
                    "RMSE (cycles)": forest_scores.rmse_cycles,
                }
            )
        if mlp_scores is not None:
            metric_rows.append(
                {
                    "Model": f"Dense MLP ({mlp_scores.model_version})",
                    "MAE (cycles)": mlp_scores.mae_cycles,
                    "RMSE (cycles)": mlp_scores.rmse_cycles,
                }
            )
        st.dataframe(metric_rows, hide_index=True, width="stretch")
        st.write(
            f"The constant baseline predicts {comparison.baseline_prediction:.2f} cycles for every engine. "
            "The tree uses the saved sensor and setting features. On this split, the tree has "
            f"{'lower' if comparison.tree_metrics.mae < comparison.baseline_metrics.mae else 'higher or equal'} "
            f"MAE by {abs(comparison.baseline_metrics.mae - comparison.tree_metrics.mae):.2f} cycles and "
            f"{'lower' if comparison.tree_metrics.rmse < comparison.baseline_metrics.rmse else 'higher or equal'} "
            f"RMSE by {abs(comparison.baseline_metrics.rmse - comparison.tree_metrics.rmse):.2f} cycles. "
            "These differences describe this held-out development split only; they do not establish "
            "future or official-test performance."
        )
        if forest_scores is not None:
            st.write(
                f"The verified Random Forest ({forest_scores.model_name}, "
                f"{forest_scores.model_version}) has MAE "
                f"{forest_scores.mae_cycles:.2f} cycles and RMSE "
                f"{forest_scores.rmse_cycles:.2f} cycles across "
                f"{forest_scores.sample_count} held-out development engines. "
                "This is a saved development-validation result, not official-test "
                "performance or evidence of future performance."
            )
        if mlp_scores is not None:
            st.write(
                f"The verified Dense MLP ({mlp_scores.model_name}, "
                f"{mlp_scores.model_version}) has MAE {mlp_scores.mae_cycles:.2f} cycles "
                f"and RMSE {mlp_scores.rmse_cycles:.2f} cycles across "
                f"{mlp_scores.sample_count} held-out development engines. Its checkpoint "
                f"was selected at epoch {mlp_scores.best_epoch} using validation MSE "
                f"{mlp_scores.selected_validation_mse_cycles_squared:.2f} cycles^2. "
                "This is a saved development-validation result, not an untouched "
                "performance estimate or official-test result."
            )
            curve_figure = go.Figure()
            epochs = list(range(1, len(mlp_scores.training_mse_cycles_squared) + 1))
            curve_figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=mlp_scores.training_mse_cycles_squared,
                    mode="lines+markers",
                    name=f"Training MSE — fitting rows ({mlp_scores.fitting_row_count:,})",
                )
            )
            curve_figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=mlp_scores.validation_mse_cycles_squared,
                    mode="lines+markers",
                    name=f"Validation MSE — engine prefixes ({mlp_scores.sample_count})",
                )
            )
            curve_figure.update_layout(
                title="Dense MLP saved learning curves",
                xaxis_title="Training epoch",
                yaxis_title="Mean squared error (cycles²)",
                legend_title="Split and sample unit",
            )
            st.plotly_chart(curve_figure, width="stretch")
            st.caption(
                "Training MSE is averaged over cycle-level fitting rows; validation MSE is "
                "averaged over one prefix per held-out engine. The values share cycles² "
                "units but use different sample units, so compare their trends cautiously. "
                f"Epoch {mlp_scores.best_epoch} was selected by the lowest validation MSE; "
                "this validation curve is checkpoint-selection evidence, not an untouched "
                "final performance estimate."
            )
        figure = px.scatter(
            comparison.predictions,
            x="Actual RUL (cycles)",
            y="Predicted RUL (cycles)",
            color="Model",
            hover_data=["Engine", "Cutoff cycle"],
            title=(
                "Saved model predictions on held-out development engines"
                if mlp_scores is not None
                else "Baseline and Decision Tree predictions on held-out development engines"
            ),
        )
        if mlp_scores is not None:
            figure.add_trace(
                px.scatter(
                    mlp_scores.predictions,
                    x="Actual RUL (cycles)",
                    y="Predicted RUL (cycles)",
                    color="Model",
                    hover_data=["Engine", "Cutoff cycle"],
                ).data[0]
            )
        upper = max(
            max(row["Actual RUL (cycles)"] for row in comparison.predictions),
            max(row["Predicted RUL (cycles)"] for row in comparison.predictions),
        )
        figure.add_trace(
            go.Scatter(
                x=[0, upper],
                y=[0, upper],
                mode="lines",
                name="Ideal prediction",
                line={"dash": "dash", "color": "gray"},
            )
        )
        figure.update_layout(
            xaxis_title="Actual RUL (cycles)",
            yaxis_title="Predicted RUL (cycles)",
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Points closer to the dashed line have smaller absolute error. Each point is one engine at its "
            "saved validation cutoff. The Random Forest aggregate scores appear in the table above. "
            "RUL estimates are not clipped."
        )
