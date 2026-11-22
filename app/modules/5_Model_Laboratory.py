"""Model Laboratory view."""

import base64
import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fleetpulse import ui
from fleetpulse.data.split_audit_view import verified_split_manifest
from fleetpulse.evaluation.final_test_display import (
    FINAL_TEST_REPORT_PATH,
    load_final_test_display,
)
from fleetpulse.evaluation.final_test_export import (
    ENGINE_FILENAME,
    SUMMARY_FILENAME,
    build_final_test_csv_exports,
)
from fleetpulse.evaluation.model_comparison import (
    load_model_comparison,
    load_verified_forest_scores,
    load_verified_mlp_scores,
)
from fleetpulse.evaluation.p22_study_display import (
    load_error_analysis,
    load_sensor_sensitivity,
    load_validation_bootstrap,
)
from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.ui_model_arena import render_model_arena
from fleetpulse.ui_presentation import module_links

st.set_page_config(layout="wide")
st.title("Model Laboratory", anchor=False)
st.markdown("Compare saved results on the **held-out FD001 test set**.")
arena_panel = st.container()
_app_dir = Path(__file__).resolve().parents[1]

with st.expander("Development validation and selection"):
    st.subheader("Development validation and selection (FD001/train)")
    st.subheader("P11 candidate selection comparison")
    selection_path = ui.PROJECT_ROOT / "artifacts" / "selection_manifest.json"
    try:
        selection = load_selection_manifest(selection_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.info(f"Saved P11 selection comparison is unavailable: {error}")
    else:
        st.caption(
            "Six saved FD001/train candidate runs on the same 20 development-validation "
            "engine prefixes. These scores were used during model selection and are not "
            "an untouched final performance estimate."
        )
        selection_rows = [
            {
                "Selection candidate": candidate["name"],
                "Selection MAE (cycles)": candidate["mae_cycles"],
                "Selection RMSE (cycles)": candidate["rmse_cycles"],
                "Configuration": json.dumps(candidate["configuration"], sort_keys=True),
            }
            for candidate in selection["candidates"]
        ]
        st.dataframe(selection_rows, hide_index=True, width="stretch")
        decision = selection["decision"]
        selected_artifact = decision["selected_artifact"]
        selected_config = decision["selected_configuration"]
        mlp_refit_rule = decision["mlp_refit_rule"]
        st.write(
            f"Owner-delegated P11-T3 selection: **{decision['selected_candidate']}**. "
            f"Saved model: `{selected_artifact['name']}` / `{selected_artifact['version']}`. "
            f"Lowest MAE: {decision['lowest_mae_candidate']}; lowest RMSE: "
            f"{decision['lowest_rmse_candidate']}. {decision['reason']}"
        )
        st.caption(
            f"Selected input contract: {selected_config['feature_config']['window_cycles']}-cycle "
            f"window, {selected_config['engineered_feature_count']} past-only features across "
            f"{len(selected_config['feature_config']['channels'])} channels. MLP (128, 64) "
            f"alternative refit rule, if needed: exactly {mlp_refit_rule['frozen_epoch_count']} "
            "epochs at seed 42, without reselecting epochs on this reused validation cohort."
        )
        st.caption(
            "The run files were independently checked against the same saved engine/cutoff "
            "identities, targets, source, and split. The compact manifest records their "
            "result-file hashes; the ignored run files are needed to reproduce the audit. "
            "These results do not establish individual-engine accuracy or maintenance readiness."
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
                    line={"dash": "dash", "color": "#697586"},
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

with st.expander("Research studies"):
    st.subheader("P22 saved model studies")
    st.caption(
        "These are fixed experiment outputs, separate from the current prediction and planner inputs. "
        "They summarize simulated FD001 evidence; none is an aircraft fault diagnosis or maintenance claim."
    )

    error_analysis_path = ui.PROJECT_ROOT / "docs/figures/p22_s1_error_analysis.json"
    try:
        error_analysis = load_error_analysis(error_analysis_path)
    except (ValueError, TypeError) as error:
        st.info(f"Saved P22 error diagnostics are unavailable: {error}")
    else:
        st.markdown("**P22-S1 — Official FD001/test error diagnostics**")
        st.caption(
            "One point per engine at its greatest observed cycle; raw predictions are paired with the "
            "separate test answers. Bins are analysis ranges, not maintenance thresholds."
        )
        error_rows = []
        for model in error_analysis["models"].values():
            summary = model["summary"]
            error_rows.append(
                {
                    "Model": model["display_name"],
                    "Version": model["version"],
                    "Low MAE (cycles)": summary["by_true_rul_bin"]["low"]["mae_cycles"],
                    "Medium MAE (cycles)": summary["by_true_rul_bin"]["medium"]["mae_cycles"],
                    "High MAE (cycles)": summary["by_true_rul_bin"]["high"]["mae_cycles"],
                    "Overall MAE (cycles)": summary["overall"]["mae_cycles"],
                    "Overall RMSE (cycles)": summary["overall"]["rmse_cycles"],
                }
            )
        st.dataframe(error_rows, hide_index=True, width="stretch")
        figure_paths = (
            ("p22_s1_actual_vs_predicted.svg", "Raw prediction versus true RUL"),
            ("p22_s1_residuals.svg", "Raw prediction minus true RUL, by true RUL"),
            ("p22_s1_rul_bins.svg", "MAE within the declared true-RUL bins"),
        )
        figure_dir = ui.PROJECT_ROOT / "docs/figures"
        for filename, description in figure_paths:
            figure_path = figure_dir / filename
            try:
                svg_bytes = figure_path.read_bytes()
                if b"<svg" not in svg_bytes[:512] or b"<script" in svg_bytes.lower():
                    raise ValueError("the saved figure is not a safe SVG image")
            except (OSError, ValueError) as error:
                st.info(f"P22-S1 figure {filename} is unavailable: {error}")
            else:
                encoded = base64.b64encode(svg_bytes).decode("ascii")
                st.markdown(
                    f'<img alt="{description}" src="data:image/svg+xml;base64,{encoded}" '
                    'style="display:block;max-width:100%;height:auto;" />',
                    unsafe_allow_html=True,
                )
                st.caption(description)
        st.caption(
            "P22-S1 uses five frozen model versions and 100 FD001/test engines. Its results are "
            "descriptive summaries of this simulated test set, not tuning evidence or per-engine intervals."
        )

    bootstrap_path = ui.PROJECT_ROOT / "docs/p22_s2_validation_bootstrap.json"
    try:
        bootstrap = load_validation_bootstrap(bootstrap_path)
    except (ValueError, TypeError) as error:
        st.info(f"Saved P22 validation bootstrap is unavailable: {error}")
    else:
        st.markdown("**P22-S2 — Validation MAE variability across engines**")
        bootstrap_rows = []
        for model in bootstrap["models"].values():
            interval = model["percentile_95_ci_mae_cycles"]
            bootstrap_rows.append(
                {
                    "Model": model["display_name"],
                    "Version": model["model_version"],
                    "Validation engines": model["engine_count"],
                    "Observed MAE (cycles)": model["observed_mae_cycles"],
                    "95% interval for validation MAE (cycles)": f"{interval[0]:.2f}–{interval[1]:.2f}",
                    "Bootstrap SE (cycles)": model["bootstrap_standard_error_cycles"],
                }
            )
        st.dataframe(bootstrap_rows, hide_index=True, width="stretch")
        st.caption(
            "10,000 paired whole-engine resamples, seed 42. The interval summarizes variation in "
            "aggregate validation MAE conditional on fixed models; it is not an interval for an engine's RUL."
        )

    sensitivity_path = ui.PROJECT_ROOT / "docs/p22_s3_sensor_sensitivity.json"
    try:
        sensitivity = load_sensor_sensitivity(sensitivity_path)
    except (ValueError, TypeError) as error:
        st.info(f"Saved P22 sensor sensitivity study is unavailable: {error}")
    else:
        st.markdown("**P22-S3 — Artificial sensor-change sensitivity**")
        sensitivity_rows = []
        for result in sensitivity["aggregate_summary"]:
            sensitivity_rows.append(
                {
                    "Model version": result["model_version"],
                    "Relative sensor change": f"±{result['relative_change_magnitude']:.0%}",
                    "Engine cases": result["engine_case_count"],
                    "Mean absolute raw RUL movement (cycles)": result[
                        "mean_absolute_prediction_change_cycles"
                    ],
                    "95th percentile movement (cycles)": result[
                        "p95_nearest_rank_absolute_prediction_change_cycles"
                    ],
                    "Maximum movement (cycles)": result["max_absolute_prediction_change_cycles"],
                    "Changed outputs": result["changed_output_count"],
                }
            )
        st.dataframe(sensitivity_rows, hide_index=True, width="stretch")
        st.caption(
            "One channel at a time across 20 FD001/train validation prefixes; synthetic ±1%/±5% "
            "changes are not observed noise or fault injections. Raw movement is sensitivity, not accuracy or causation."
        )

official_results = ()
csv_exports = None
with st.expander("Official test details and downloads"):
    st.subheader("Official held-out test results (FD001/test)")
    st.caption(
        "This is a distinct final reporting section for the official FD001 test engines. "
        "These scores were not used to fit, select, or tune the models. They are error measures in cycles, not accuracy percentages."
    )
    official_results_path = ui.PROJECT_ROOT / FINAL_TEST_REPORT_PATH
    try:
        official_results = load_final_test_display(official_results_path)
    except FileNotFoundError:
        st.info(
            "Verified official test results are unavailable in this deployment. "
            "The local P13 report is not part of the public source checkout."
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"Official test results are hidden because the saved P13 report failed verification: {error}")
    else:
        st.dataframe(
            [
                {
                    "Model": result.display_name,
                    "Model version": result.version,
                    "Sample count (engines)": result.sample_count,
                    "MAE (cycles)": result.mae_cycles,
                    "RMSE (cycles)": result.rmse_cycles,
                }
                for result in official_results
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "MAE is mean absolute error; RMSE is the square root of mean squared error. "
            "Both are error magnitudes in operating cycles; lower means smaller errors on this 100-engine test sample. "
            "Neither is an accuracy percentage or a guarantee about future engines. "
            "The figures come from the checksum-verified local P13 report and were not used for model selection."
        )
        try:
            csv_exports = build_final_test_csv_exports(official_results_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"CSV exports are unavailable because the saved per-engine results failed verification: {error}")
        else:
            st.download_button(
                "Download summary CSV",
                data=csv_exports.summary_csv,
                file_name=SUMMARY_FILENAME,
                mime="text/csv",
                key="p13_test_summary_csv",
            )
            st.download_button(
                "Download per-engine CSV",
                data=csv_exports.per_engine_csv,
                file_name=ENGINE_FILENAME,
                mime="text/csv",
                key="p13_test_per_engine_csv",
            )
            st.caption(
                f"Exports contain {csv_exports.summary_row_count} model summary rows and "
                f"{csv_exports.per_engine_row_count} per-engine rows. The values are copied from the "
                "verified report; missing or inconsistent fields make the exports unavailable."
            )

with arena_panel:
    if official_results:
        render_model_arena(official_results, csv_exports)
    else:
        st.info("The arena needs a verified FD001/test report. See Official test details and downloads.")

with st.expander("Related pages"):
    module_links(
        (_app_dir / "modules" / "3_RUL_Prediction.py", "Continue to RUL Prediction", ":material/query_stats:"),
        (_app_dir / "views" / "home.py", "Return Home", ":material/home:"),
    )
