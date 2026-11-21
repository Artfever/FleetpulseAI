from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.audit_submission_archive import audit_submission_archive


def _write_archive(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, content in entries.items():
            archive.writestr(member, content)


def test_clean_portable_archive_passes(tmp_path: Path) -> None:
    archive_path = tmp_path / "submission.zip"
    _write_archive(
        archive_path,
        {
            "README.md": "Install from the project root.",
            ".env.example": "APP_MODE=demo\n",
            "src/fleetpulse/__init__.py": "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n",
            "artifacts/final/tree/bundles/decision-tree-rul_manifest.json": "{}",
        },
    )

    report = audit_submission_archive(archive_path)

    assert report["passed"] is True
    assert report["finding_count"] == 0
    assert len(report["sha256"]) == 64


@pytest.mark.parametrize(
    ("member", "expected_code"),
    [
        (".venv/Lib/site.py", "local_environment"),
        ("tests/__pycache__/test_a.pyc", "local_environment"),
        (".env.local", "environment_file"),
        ("secrets.json", "secret_filename"),
        ("data/raw/train_FD001.txt", "raw_dataset"),
        ("artifacts/development/model.joblib", "local_artifact"),
        ("C:/" + "Users/abdullah/project/README.md", "absolute_member_path"),
        ("../outside.txt", "path_traversal"),
    ],
)
def test_forbidden_archive_members_are_reported(
    tmp_path: Path, member: str, expected_code: str
) -> None:
    archive_path = tmp_path / "submission.zip"
    _write_archive(archive_path, {member: "content"})

    report = audit_submission_archive(archive_path)

    assert report["passed"] is False
    assert expected_code in {item["code"] for item in report["findings"]}


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        ("Executed from C:/" + "Users/alice/Temp/clone", "machine_specific_content"),
        ("Local data at /" + "home/alice/fleetpulse/data/raw", "machine_specific_content"),
        ("-----BEGIN " + "PRIVATE KEY-----", "secret_content"),
        ("token=ghp_" + "123456789012345678901234567890123456", "secret_content"),
    ],
)
def test_machine_paths_and_secret_material_are_reported(
    tmp_path: Path, content: str, expected_code: str
) -> None:
    archive_path = tmp_path / "submission.zip"
    _write_archive(archive_path, {"docs/evidence.md": content})

    report = audit_submission_archive(archive_path)

    assert report["passed"] is False
    assert expected_code in {item["code"] for item in report["findings"]}
