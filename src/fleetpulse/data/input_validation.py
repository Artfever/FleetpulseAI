"""Validate bounded UTF-8 CSV uploads against the raw FD001 history schema."""

from __future__ import annotations

import csv
import io
from collections import Counter
from numbers import Integral

from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.data.quality import QualityFinding, validate_engine_history

MAX_UPLOAD_BYTES = 1_048_576


class UploadValidationError(ValueError):
    """A CSV upload cannot safely become normalized engine-history records."""

    def __init__(self, findings: tuple[QualityFinding, ...]):
        if not findings:
            raise ValueError("UploadValidationError requires at least one finding.")
        self.findings = findings
        self.code = findings[0].code
        self.row_number = findings[0].row_number
        self.field = findings[0].field
        super().__init__("; ".join(finding.message for finding in findings))


def _error(code: str, message: str, *, row: int | None = None, field: str | None = None):
    return QualityFinding(code, "error", row, field, message)


def parse_fd001_csv_upload(
    contents: bytes, *, max_upload_bytes: int = MAX_UPLOAD_BYTES
) -> list[dict[str, int | float]]:
    """Parse and validate a headered CSV upload without reordering its records.

    Headers must contain exactly the normalized 26 raw-history names from
    ``HISTORY_COLUMNS``. Their order may vary because each row is mapped by
    header name. Target RUL columns and any other extra fields are rejected.
    Cycle gaps do not reject parsing; callers can use ``validate_engine_history``
    to surface warning findings. All quality errors reject the upload.
    """
    if not isinstance(contents, bytes):
        raise TypeError("contents must be bytes from the uploaded file.")
    if (
        isinstance(max_upload_bytes, bool)
        or not isinstance(max_upload_bytes, Integral)
        or max_upload_bytes < 1
    ):
        raise ValueError("max_upload_bytes must be a positive integer.")
    if len(contents) > max_upload_bytes:
        raise UploadValidationError(
            (_error("file_too_large", f"CSV exceeds the {max_upload_bytes}-byte upload limit."),)
        )
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UploadValidationError(
            (_error("invalid_encoding", "CSV must use UTF-8 encoding."),)
        ) from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    records: list[dict[str, int | float]] = []
    parse_findings: list[QualityFinding] = []
    try:
        headers = next(reader, None)
        if headers is None or not headers:
            raise UploadValidationError(
                (_error("empty_upload", "CSV must include a header and at least one history row."),)
            )
        duplicates = sorted(name for name, count in Counter(headers).items() if count > 1)
        if duplicates:
            raise UploadValidationError(
                (
                    _error(
                        "duplicate_header",
                        f"CSV header contains duplicate field names: {', '.join(duplicates)}.",
                    ),
                )
            )
        missing = sorted(set(HISTORY_COLUMNS) - set(headers))
        extra = sorted(set(headers) - set(HISTORY_COLUMNS))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unexpected {', '.join(extra)}")
            raise UploadValidationError(
                (
                    _error(
                        "invalid_header",
                        "CSV fields do not match the raw-history schema: "
                        + "; ".join(details)
                        + ". Include only the required raw-history fields; remove "
                        "unexpected or target fields. Column meanings are never guessed.",
                    ),
                )
            )

        for row_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(headers):
                parse_findings.append(
                    _error(
                        "invalid_row_width",
                        f"Row has {len(values)} values; expected {len(headers)}.",
                        row=row_number,
                    )
                )
                continue
            record: dict[str, int | float] = {}
            for field, value in zip(headers, values, strict=True):
                try:
                    record[field] = float(value)
                except (OverflowError, ValueError):
                    parse_findings.append(
                        _error(
                            "invalid_number",
                            f"{field} must contain a numeric value.",
                            row=row_number,
                            field=field,
                        )
                    )
            if len(record) == len(headers):
                records.append(record)
    except csv.Error as error:
        raise UploadValidationError(
            (_error("malformed_csv", f"CSV syntax is invalid: {error}"),)
        ) from error

    report = validate_engine_history(records)
    findings = (*parse_findings, *report.errors)
    if findings:
        raise UploadValidationError(findings)
    return records
