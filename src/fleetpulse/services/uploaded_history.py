"""Validated CSV upload handoff to the shared prediction service."""

from collections.abc import Mapping
from pathlib import Path

from fleetpulse.data.input_validation import parse_fd001_csv_upload
from fleetpulse.services.predictor import (
    PredictionInputError,
    RULPrediction,
    predict_rul_with_context,
)


def predict_uploaded_csv(
    contents: bytes,
    *,
    dataset_partition: str,
    unit_id: int,
    manifest_path: str | Path,
) -> RULPrediction:
    """Validate an uploaded CSV before passing one selected engine to prediction.

    Parsing and all quality checks happen before the prediction service can
    load a model. The caller must explicitly select an engine when a file
    contains more than one. The shared service validates the selected history
    again and applies the same saved model and feature contract as built-in
    histories.
    """
    records = parse_fd001_csv_upload(contents)
    selected: list[Mapping[str, int | float]] = [
        row for row in records if row["unit_id"] == unit_id
    ]
    if not selected:
        raise PredictionInputError(f"Uploaded history does not contain engine {unit_id}.")
    return predict_rul_with_context(
        selected,
        dataset_partition=dataset_partition,
        unit_id=unit_id,
        manifest_path=manifest_path,
    )
