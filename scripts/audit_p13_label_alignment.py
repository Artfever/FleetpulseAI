"""Independently audit the FD001 test-engine/RUL-line pairing."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TEST_PATH = Path("data/raw/test_FD001.txt")
RUL_PATH = Path("data/raw/RUL_FD001.txt")
PAIRED_PATH = Path("artifacts/final/p13/official_test_pairings.json")
OUTPUT_PATH = Path("artifacts/final/p13/a1_alignment_audit.json")
EXPECTED_TEST_SHA256 = "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
EXPECTED_RUL_SHA256 = "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
PAIRING_SCHEMA = "p13-official-test-pairings-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sorted_id_line_mapping(engine_ids: Sequence[int], labels: Sequence[int]) -> list[dict[str, int]]:
    """Map vector line i to sorted distinct engine ID i (1-indexed)."""
    if any(type(engine_id) is not int or engine_id < 1 for engine_id in engine_ids):
        raise ValueError("Engine IDs must be positive integers.")
    ordered_ids = sorted(set(engine_ids))
    if len(ordered_ids) != len(engine_ids):
        raise ValueError("Engine ID inventory contains duplicates.")
    if len(ordered_ids) != len(labels):
        raise ValueError(
            f"RUL answer count {len(labels)} does not match engine count {len(ordered_ids)}."
        )
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("RUL answers must be nonnegative integers.")
    return [
        {"answer_line": line, "unit_id": engine_id, "true_rul_cycles": labels[line - 1]}
        for line, engine_id in enumerate(ordered_ids, start=1)
    ]


def _read_test_histories(path: str | Path) -> tuple[list[int], dict[int, int]]:
    """Read actual engine block order and final observed cycles independently."""
    file_order: list[int] = []
    last_cycle: dict[int, int] = {}
    closed_ids: set[int] = set()
    previous_id: int | None = None
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 26:
            raise ValueError(f"Test history line {line_number} must contain 26 fields.")
        try:
            engine_id, cycle = int(values[0]), int(values[1])
        except ValueError as error:
            raise ValueError(f"Test history line {line_number} has invalid identity fields.") from error
        if engine_id < 1 or cycle < 1:
            raise ValueError(f"Test history line {line_number} has a nonpositive identity.")
        if engine_id != previous_id:
            if engine_id in closed_ids:
                raise ValueError("Test history engine trajectories are interleaved.")
            if previous_id is not None:
                closed_ids.add(previous_id)
            file_order.append(engine_id)
            previous_id = engine_id
        if engine_id in last_cycle and cycle <= last_cycle[engine_id]:
            raise ValueError(f"Test history cycles are not increasing for engine {engine_id}.")
        last_cycle[engine_id] = cycle
    if not file_order:
        raise ValueError("Test history contains no engine rows.")
    return file_order, last_cycle


def _read_rul_values(path: str | Path) -> list[int]:
    answers: list[int] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(f"RUL line {line_number} must contain one integer answer.")
        try:
            answer = int(values[0])
        except ValueError as error:
            raise ValueError(f"RUL line {line_number} is not an integer.") from error
        if answer < 0:
            raise ValueError(f"RUL line {line_number} is negative.")
        answers.append(answer)
    return answers


def verify_model_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, int]],
    last_cycles: Mapping[int, int],
) -> int:
    """Independently verify a model's per-ID label/cutoff pairing."""
    expected = {entry["unit_id"]: entry for entry in mapping}
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Paired result rows must be mappings.")
        engine_id = row.get("unit_id")
        if type(engine_id) is not int or engine_id not in expected:
            raise ValueError("Paired row has an unknown engine ID.")
        if engine_id in by_id:
            raise ValueError(f"Duplicate paired row for engine {engine_id}.")
        by_id[engine_id] = row
    if set(by_id) != set(expected):
        raise ValueError("Paired rows do not cover the full engine set exactly once.")
    for engine_id, expected_row in expected.items():
        row = by_id[engine_id]
        if (
            type(row.get("answer_line")) is not int
            or row.get("answer_line") != expected_row["answer_line"]
            or type(row.get("true_rul_cycles")) is not int
            or row.get("true_rul_cycles") != expected_row["true_rul_cycles"]
            or type(row.get("observed_cycle")) is not int
            or row.get("observed_cycle") != last_cycles[engine_id]
        ):
            raise ValueError(f"Label line or observed cycle is misaligned for engine {engine_id}.")
        prediction = row.get("raw_prediction_cycles")
        if type(prediction) not in (int, float) or not math.isfinite(prediction):
            raise ValueError(f"Prediction for engine {engine_id} is not finite.")
    return len(by_id)


def audit_alignment(
    test_path: str | Path = TEST_PATH,
    rul_path: str | Path = RUL_PATH,
    paired_path: str | Path = PAIRED_PATH,
) -> dict[str, Any]:
    test_file, rul_file, paired_file = Path(test_path), Path(rul_path), Path(paired_path)
    test_bytes, rul_bytes, paired_bytes = (
        test_file.read_bytes(),
        rul_file.read_bytes(),
        paired_file.read_bytes(),
    )
    test_hash, rul_hash = _sha256(test_bytes), _sha256(rul_bytes)
    if test_hash != EXPECTED_TEST_SHA256 or rul_hash != EXPECTED_RUL_SHA256:
        raise ValueError("FD001 source file checksums differ from the pinned provenance.")
    document = json.loads(paired_bytes.decode("utf-8"))
    payload = document.get("payload")
    if (
        not isinstance(payload, Mapping)
        or document.get("sha256") != _sha256(_canonical(payload))
        or payload.get("schema_version") != PAIRING_SCHEMA
        or payload.get("dataset_partition") != "FD001/test"
    ):
        raise ValueError("Paired prediction artifact schema or checksum is invalid.")

    file_order, last_cycles = _read_test_histories(test_file)
    labels = _read_rul_values(rul_file)
    mapping = sorted_id_line_mapping(file_order, labels)
    sorted_ids = [entry["unit_id"] for entry in mapping]
    if file_order != sorted_ids:
        raise ValueError("Actual FD001 test trajectories are not in ascending engine-ID order.")
    if sorted_ids != list(range(1, 101)):
        raise ValueError("Actual FD001 test engine identity set differs from 1 through 100.")
    if (
        payload.get("test_history_sha256") != test_hash
        or payload.get("answer_file_sha256") != rul_hash
        or payload.get("engine_count") != len(sorted_ids)
    ):
        raise ValueError("Paired artifact source hashes or engine count do not match local files.")

    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 5:
        raise ValueError("Expected five frozen model pair tables.")
    paired_row_count = 0
    model_names: list[str] = []
    for model in models:
        if not isinstance(model, Mapping) or not isinstance(model.get("rows"), list):
            raise TypeError("Paired model table is malformed.")
        count = verify_model_pair_rows(model["rows"], mapping, last_cycles)
        if model.get("paired_count") != count:
            raise ValueError("Paired model row count metadata is inconsistent.")
        paired_row_count += count
        model_names.append(str(model.get("name")))
    if len(set(model_names)) != 5:
        raise ValueError("Paired artifact has duplicate model names.")
    return {
        "schema_version": "p13-a1-independent-alignment-audit-v1",
        "test_history_sha256": test_hash,
        "rul_answer_sha256": rul_hash,
        "paired_payload_sha256": document["sha256"],
        "paired_file_sha256": _sha256(paired_bytes),
        "test_engine_file_order": file_order,
        "engine_count": len(file_order),
        "answer_count": len(labels),
        "line_to_engine_mapping": [
            {"answer_line": row["answer_line"], "unit_id": row["unit_id"]}
            for row in mapping
        ],
        "model_names": model_names,
        "paired_rows_verified": paired_row_count,
        "shuffled_toy_fixture_required": True,
        "metrics_recalculated": False,
        "tuning_performed": False,
    }


def write_audit_report(report: Mapping[str, Any], output_path: str | Path = OUTPUT_PATH) -> str:
    digest = _sha256(_canonical(report))
    content = json.dumps(
        {"payload": report, "sha256": digest}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different audit evidence at {output}.")
        return digest
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return digest


def main() -> None:
    report = audit_alignment()
    digest = write_audit_report(report)
    mapping = report["line_to_engine_mapping"]
    print(f"test_engine_count: {report['engine_count']}")
    print(f"rul_answer_count: {report['answer_count']}")
    print(f"file_order_matches_sorted_ids: {report['test_engine_file_order'] == sorted(report['test_engine_file_order'])}")
    print(f"paired_model_count: {len(report['model_names'])}")
    print(f"paired_rows_verified: {report['paired_rows_verified']}")
    print(f"first_mapping: answer line {mapping[0]['answer_line']} -> engine {mapping[0]['unit_id']}")
    print(f"last_mapping: answer line {mapping[-1]['answer_line']} -> engine {mapping[-1]['unit_id']}")
    print(f"report: {OUTPUT_PATH}")
    print(f"report_payload_sha256: {digest}")


if __name__ == "__main__":
    main()
