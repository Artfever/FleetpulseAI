"""Raw prediction and explicitly separate presentation/context services."""

from fleetpulse.services.comparison_artifact import (
    build_paired_comparison_artifact,
    model_provenance_from_manifest,
)
from fleetpulse.services.explanations import (
    BACKGROUND_SELECTION_POLICY,
    ExplanationConsistency,
    FeatureContribution,
    FinalTreeExplanation,
    GlobalFeatureImportance,
    TrainingFeatureBackground,
    build_fd001_training_background,
    check_final_explanation_consistency,
    explain_final_tree_prediction,
)
from fleetpulse.services.final_comparison import (
    FinalHistoryComparison,
    FinalModelFailure,
    FinalModelPrediction,
    compare_final_models,
)
from fleetpulse.services.fleet import (
    EngineIdentity,
    FleetEngineHistory,
    FleetEngineHistoryError,
    FleetHistories,
    FleetPredictionResult,
    extract_latest_engine_histories,
    predict_fleet,
)
from fleetpulse.services.history import (
    MAX_REPLAY_CACHE_POINTS,
    MAX_REPLAY_POINTS,
    HistoryReplay,
    HistoryReplayPoint,
    clear_history_replay_cache,
    replay_engine_history,
)
from fleetpulse.services.planner import PlannerJobAssumption, build_planning_scenario
from fleetpulse.services.predictor import (
    PredictionArtifactError,
    PredictionInputError,
    RawRULPrediction,
    RULPrediction,
    predict_rul,
    predict_rul_with_context,
)
from fleetpulse.services.schedule_comparison import (
    ScheduleComparisonResult,
    ScheduleMethodResult,
    compare_schedules,
)
from fleetpulse.services.uploaded_history import predict_uploaded_csv

__all__ = [
    "BACKGROUND_SELECTION_POLICY",
    "MAX_REPLAY_CACHE_POINTS",
    "MAX_REPLAY_POINTS",
    "EngineIdentity",
    "ExplanationConsistency",
    "FeatureContribution",
    "FinalHistoryComparison",
    "FinalModelFailure",
    "FinalModelPrediction",
    "FinalTreeExplanation",
    "FleetEngineHistory",
    "FleetEngineHistoryError",
    "FleetHistories",
    "FleetPredictionResult",
    "GlobalFeatureImportance",
    "HistoryReplay",
    "HistoryReplayPoint",
    "PlannerJobAssumption",
    "PredictionArtifactError",
    "PredictionInputError",
    "RULPrediction",
    "RawRULPrediction",
    "ScheduleComparisonResult",
    "ScheduleMethodResult",
    "TrainingFeatureBackground",
    "build_fd001_training_background",
    "build_paired_comparison_artifact",
    "build_planning_scenario",
    "check_final_explanation_consistency",
    "clear_history_replay_cache",
    "compare_final_models",
    "compare_schedules",
    "explain_final_tree_prediction",
    "extract_latest_engine_histories",
    "model_provenance_from_manifest",
    "predict_fleet",
    "predict_rul",
    "predict_rul_with_context",
    "predict_uploaded_csv",
    "replay_engine_history",
]
