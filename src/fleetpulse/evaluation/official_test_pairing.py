"""Pair frozen FD001/test predictions with the separate official RUL vector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetpulse.data.loader import load_rul_values
from fleetpulse.evaluation.test import DEFAULT_OUTPUT_PATH, PREDICTION_SCHEMA

DEFAULT_RUL_PATH = Path("data/raw/RUL_FD001.txt")
DEFAULT_PAIRED_OUTPUT_PATH = Path("artifacts/final/p13/official_test_pairings.json")
EXPECTED_TEST_HISTORY_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
EXPECTED_RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
PAIRING_SCHEMA = "p13-official-test-pairings-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pair_test_predictions(
    prediction_document: Mapping[str, Any],
    rul_values: Sequence[int],
    *,
    rul_source_sha256: str,
    rul_source_name: str,
    expected_engine_ids: Sequence[int] | None = None,
    expected_model_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map the RUL vector onto sorted engine IDs and enforce exact coverage.

    The mapping is explicit: RUL line 1 goes to the lowest test `unit_id`, line
    2 to the next, and so on. The returned artifact records every such mapping.
    No metric or error is calculated here; P13-S3 owns result presentation.
    """
    if not isinstance(prediction_document, Mapping):
        raise TypeError("prediction_document must be a JSON mapping.")
    payload = prediction_document.get("payload")
    if not isinstance(payload, Mapping) or prediction_document.get("sha256") != _canonical_sha256(
        payload
    ):
        raise ValueError("Prediction artifact payload hash is missing or invalid.")
    if (
        payload.get("schema_version") != PREDICTION_SCHEMA
        or payload.get("dataset_partition") != "FD001/test"
        or payload.get("labels_read") is not False
    ):
        raise ValueError("Prediction artifact is not an unscored FD001/test P13-S1 report.")
    engine_count = payload.get("engine_count")
    if type(engine_count) is not int or engine_count < 1:
        raise ValueError("Prediction artifact engine_count is invalid.")
    engines = payload.get("engines")
    if not isinstance(engines, list) or len(engines) != engine_count:
        raise ValueError("Prediction artifact engine inventory is incomplete.")
    engine_by_id: dict[int, dict[str, Any]] = {}
    for row in engines:
        if not isinstance(row, Mapping):
            raise TypeError("Every engine metadata row must be an object.")
        engine_id = row.get("unit_id")
        cutoff = row.get("observed_cycle")
        if type(engine_id) is not int or engine_id < 1 or type(cutoff) is not int or cutoff < 1:
            raise ValueError("Prediction artifact contains invalid engine identity/cutoff.")
        if engine_id in engine_by_id:
            raise ValueError(f"Duplicate test engine {engine_id} in prediction metadata.")
        engine_by_id[engine_id] = dict(row)
    sorted_engine_ids = sorted(engine_by_id)
    if expected_engine_ids is not None and sorted_engine_ids != list(expected_engine_ids):
        raise ValueError("Test engine IDs differ from the expected official identity sequence.")
    if payload.get("source_sha256") != EXPECTED_TEST_HISTORY_SHA256:
        raise ValueError("Prediction artifact was generated from an unexpected FD001/test file.")

    if len(rul_values) != len(sorted_engine_ids):
        raise ValueError(
            f"RUL answer count {len(rul_values)} does not match engine count "
            f"{len(sorted_engine_ids)}."
        )
    labels: dict[int, dict[str, int]] = {}
    for answer_line, (engine_id, true_rul) in enumerate(
        zip(sorted_engine_ids, rul_values, strict=True), start=1
    ):
        if type(true_rul) is not int or true_rul < 0:
            raise ValueError(f"RUL answer line {answer_line} must be a nonnegative integer.")
        labels[engine_id] = {"answer_line": answer_line, "true_rul_cycles": true_rul}

    saved_models = payload.get("models")
    if not isinstance(saved_models, list) or not saved_models:
        raise ValueError("Prediction artifact has no model outputs.")
    if expected_model_versions is not None and len(saved_models) != len(expected_model_versions):
        raise ValueError("Prediction artifact does not contain every frozen final model.")
    names: set[str] = set()
    paired_models: list[dict[str, Any]] = []
    for model in saved_models:
        if not isinstance(model, Mapping):
            raise TypeError("Every model record must be an object.")
        name = model.get("name")
        version = model.get("version")
        predictions = model.get("predictions")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Model names must be nonempty and unique.")
        names.add(name)
        if not isinstance(version, str) or not version:
            raise ValueError(f"Model {name} has no version.")
        if expected_model_versions is not None and expected_model_versions.get(name) != version:
            raise ValueError(f"Model {name} version differs from the frozen final registry.")
        if model.get("prediction_count") != len(sorted_engine_ids):
            raise ValueError(f"Model {name} prediction_count does not match test engine count.")
        if not isinstance(predictions, list) or len(predictions) != len(sorted_engine_ids):
            raise ValueError(f"Model {name} does not have exactly one prediction per engine.")
        prediction_by_id: dict[int, Mapping[str, Any]] = {}
        for item in predictions:
            if not isinstance(item, Mapping):
                raise TypeError(f"Model {name} contains a malformed prediction row.")
            engine_id = item.get("unit_id")
            prediction = item.get("raw_rul_cycles")
            if type(engine_id) is not int or engine_id not in engine_by_id:
                raise ValueError(f"Model {name} contains an unknown test engine ID.")
            if engine_id in prediction_by_id:
                raise ValueError(f"Model {name} has duplicate prediction for engine {engine_id}.")
            if type(prediction) not in (int, float) or not math.isfinite(prediction):
                raise ValueError(f"Model {name} prediction for engine {engine_id} is not finite.")
            if item.get("observed_cycle") != engine_by_id[engine_id]["observed_cycle"]:
                raise ValueError(f"Model {name} cutoff differs for engine {engine_id}.")
            prediction_by_id[engine_id] = item
        if set(prediction_by_id) != set(sorted_engine_ids):
            raise ValueError(f"Model {name} does not cover every test engine exactly once.")
        paired_rows = []
        for engine_id in sorted_engine_ids:
            prediction = prediction_by_id[engine_id]
            label = labels[engine_id]
            paired_rows.append(
                {
                    "unit_id": engine_id,
                    "observed_cycle": engine_by_id[engine_id]["observed_cycle"],
                    "answer_line": label["answer_line"],
                    "true_rul_cycles": label["true_rul_cycles"],
                    "raw_prediction_cycles": float(prediction["raw_rul_cycles"]),
                }
            )
        paired_models.append(
            {
                "name": name,
                "version": version,
                "paired_count": len(paired_rows),
                "rows": paired_rows,
            }
        )

    return {
        "schema_version": PAIRING_SCHEMA,
        "dataset_partition": "FD001/test",
        "prediction_payload_sha256": prediction_document["sha256"],
        "test_history_sha256": payload["source_sha256"],
        "answer_file": rul_source_name,
        "answer_file_sha256": rul_source_sha256,
        "engine_count": len(sorted_engine_ids),
        "label_alignment_policy": (
            "RUL line i maps to sorted FD001/test unit_id at position i; "
            "the 1-indexed line-to-engine mapping is recorded explicitly."
        ),
        "scores_computed": False,
        "models": paired_models,
    }


def _write_immutable_pairing(payload: Mapping[str, Any], path: str | Path) -> str:
    digest = _canonical_sha256(payload)
    content = json.dumps(
        {"payload": payload, "sha256": digest},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"Refusing to overwrite different pairings at {output}; use a new versioned path."
            )
        return digest
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return digest


def run_official_test_pairing(
    prediction_path: str | Path = DEFAULT_OUTPUT_PATH,
    rul_path: str | Path = DEFAULT_RUL_PATH,
    output_path: str | Path = DEFAULT_PAIRED_OUTPUT_PATH,
) -> tuple[dict[str, Any], str]:
    """Verify the pinned original files, load labels, and save their explicit pairing."""
    prediction_document = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    if not isinstance(prediction_document, Mapping):
        raise TypeError("Prediction artifact must be a JSON object.")
    rul_file = Path(rul_path)
    rul_hash = hashlib.sha256(rul_file.read_bytes()).hexdigest()
    if rul_hash != EXPECTED_RUL_SHA256:
        raise ValueError("RUL_FD001.txt differs from the pinned original source checksum.")
    labels = load_rul_values(rul_file)

    from fleetpulse.models.registry import FINAL_MODEL_REGISTRY

    model_versions = {name: spec.version for name, spec in FINAL_MODEL_REGISTRY.items()}
    paired = pair_test_predictions(
        prediction_document,
        labels,
        rul_source_sha256=rul_hash,
        rul_source_name=rul_file.name,
        expected_engine_ids=tuple(range(1, 101)),
        expected_model_versions=model_versions,
    )
    digest = _write_immutable_pairing(paired, output_path)
    return paired, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair frozen FD001/test predictions with the separate official RUL vector."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rul-answers", type=Path, default=DEFAULT_RUL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAIRED_OUTPUT_PATH)
    arguments = parser.parse_args()
    report, digest = run_official_test_pairing(
        arguments.predictions, arguments.rul_answers, arguments.output
    )
    print(f"dataset_partition: {report['dataset_partition']}")
    print(f"engine_count: {report['engine_count']}")
    print(f"model_count: {len(report['models'])}")
    print(f"paired_prediction_count: {sum(model['paired_count'] for model in report['models'])}")
    print(f"answer_file_sha256: {report['answer_file_sha256']}")
    print(f"scores_computed: {report['scores_computed']}")
    for row in report["models"][0]["rows"][:3]:
        print(
            f"example unit_id={row['unit_id']} answer_line={row['answer_line']} "
            f"true_rul={row['true_rul_cycles']} prediction={row['raw_prediction_cycles']:.8f} cycles"
        )
    print(f"report: {arguments.output}")
    print(f"payload_sha256: {digest}")


if __name__ == "__main__":
    main()
