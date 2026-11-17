"""Bounded CSV upload parsing and FD001 raw-history validation."""

import csv
import io

import pytest

from fleetpulse.data.input_validation import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    parse_fd001_csv_upload,
)
from fleetpulse.data.loader import HISTORY_COLUMNS


def _record(unit_id=910001, cycle=1):
    return {
        field: unit_id if field == "unit_id" else cycle if field == "cycle" else 0.25
        for field in HISTORY_COLUMNS
    }


def _csv_bytes(headers=HISTORY_COLUMNS, rows=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    for row in rows if rows is not None else [_record()]:
        writer.writerow([row.get(field, "") for field in headers])
    return stream.getvalue().encode("utf-8")


def test_valid_csv_maps_by_header_and_preserves_row_order():
    headers = tuple(reversed(HISTORY_COLUMNS))
    first = _record(cycle=1)
    second = _record(cycle=2)

    records = parse_fd001_csv_upload(_csv_bytes(headers, [first, second]))

    assert [record["cycle"] for record in records] == [1.0, 2.0]
    assert [record["unit_id"] for record in records] == [910001.0, 910001.0]
    assert tuple(records[0]) == headers
    assert "rul" not in records[0]


@pytest.mark.parametrize(
    ("contents", "code"),
    [
        (b"", "empty_upload"),
        (_csv_bytes((*HISTORY_COLUMNS[:-1],)), "invalid_header"),
        (_csv_bytes((*HISTORY_COLUMNS, "rul")), "invalid_header"),
        (_csv_bytes((*HISTORY_COLUMNS[:-1], HISTORY_COLUMNS[-2])), "duplicate_header"),
        (b"\xff", "invalid_encoding"),
    ],
)
def test_invalid_csv_headers_and_encodings_are_rejected(contents, code):
    with pytest.raises(UploadValidationError) as error:
        parse_fd001_csv_upload(contents)

    assert error.value.code == code


def test_upload_byte_limit_is_enforced_before_csv_parsing():
    with pytest.raises(UploadValidationError, match="upload limit") as error:
        parse_fd001_csv_upload(b"x" * (MAX_UPLOAD_BYTES + 1))

    assert error.value.code == "file_too_large"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("sensor_1", "not-a-number", "invalid_number"),
        ("sensor_1", "nan", "non_finite_reading"),
        ("cycle", "0", "invalid_cycle"),
        ("cycle", "1.5", "invalid_cycle"),
        ("unit_id", "-1", "invalid_identifier"),
    ],
)
def test_invalid_numbers_and_identifiers_are_rejected(field, value, code):
    row = _record()
    row[field] = value

    with pytest.raises(UploadValidationError) as error:
        parse_fd001_csv_upload(_csv_bytes(rows=[row]))

    assert code in {finding.code for finding in error.value.findings}


def test_duplicate_engine_cycle_is_rejected():
    row = _record(cycle=1)

    with pytest.raises(UploadValidationError) as error:
        parse_fd001_csv_upload(_csv_bytes(rows=[row, row]))

    assert "duplicate_engine_cycle" in {finding.code for finding in error.value.findings}


def test_ragged_csv_row_is_rejected_with_row_number():
    contents = _csv_bytes() + b"1,2,3\n"

    with pytest.raises(UploadValidationError) as error:
        parse_fd001_csv_upload(contents)

    assert error.value.code == "invalid_row_width"
    assert error.value.row_number == 3
