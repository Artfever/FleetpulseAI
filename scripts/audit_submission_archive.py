"""Reject local-only files and machine-specific paths in a submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

FORBIDDEN_COMPONENTS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MACHINE_PATH = re.compile(
    r"(?i)(?:\b[A-Z]:[\\/](?:Users|Documents and Settings|home|workspace|workspaces|src|mount)[\\/][^\s\"'<>]+|/(?:home|Users|mnt|mount|tmp|private/tmp)/[^\s\"'<>]+)"
)
SECRET_TEXT = re.compile(
    r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"
)
SECRET_NAMES = {"id_rsa", "id_ed25519", ".netrc", "credentials.json", "secrets.json"}
SCRATCH_NAMES = {".ds_store", "desktop.ini", "thumbs.db"}


@dataclass(frozen=True)
class Finding:
    code: str
    member: str
    detail: str


def _path_findings(member: str) -> list[Finding]:
    normalized = member.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part.casefold() for part in path.parts]
    findings: list[Finding] = []
    if path.is_absolute() or re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("//"):
        findings.append(Finding("absolute_member_path", member, "ZIP member name must be relative."))
    if ".." in parts:
        findings.append(Finding("path_traversal", member, "ZIP member path contains '..'."))
    for part in parts:
        if part in FORBIDDEN_COMPONENTS:
            findings.append(Finding("local_environment", member, f"Forbidden environment/cache component: {part}."))
    name = path.name.casefold()
    if (name == ".env" or (name.startswith(".env.") and name != ".env.example")):
        findings.append(Finding("environment_file", member, "Environment files are excluded; only .env.example is allowed."))
    if name in SECRET_NAMES or name.endswith((".pem", ".p12", ".pfx", ".key")):
        findings.append(Finding("secret_filename", member, "Filename resembles a private key or credential file."))
    if name in SCRATCH_NAMES or name.endswith((".tmp", ".bak", ".log", "~")):
        findings.append(Finding("scratch_file", member, "Temporary, backup, or OS scratch file is excluded."))
    if len(parts) >= 2 and parts[:2] == ["data", "raw"] and name != ".gitkeep":
        findings.append(Finding("raw_dataset", member, "Full local raw datasets are provisioned separately."))
    if len(parts) >= 2 and parts[0] == "artifacts" and parts[1] in {"development", "experiments"} and name != ".gitkeep":
        findings.append(Finding("local_artifact", member, "Development or experiment artifacts are not submission inputs."))
    return findings


def audit_submission_archive(archive_path: Path) -> dict[str, object]:
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = info.filename
                if info.is_dir():
                    continue
                findings.extend(_path_findings(member))
                suffix = PurePosixPath(member.replace("\\", "/")).suffix.casefold()
                if suffix in TEXT_SUFFIXES and info.file_size <= 5 * 1024 * 1024:
                    try:
                        content = archive.read(info).decode("utf-8")
                    except (UnicodeDecodeError, RuntimeError, zipfile.BadZipFile):
                        continue
                    if MACHINE_PATH.search(content):
                        findings.append(Finding("machine_specific_content", member, "Text file contains an absolute machine/development path."))
                    if SECRET_TEXT.search(content):
                        findings.append(Finding("secret_content", member, "Text file contains a private-key or token pattern."))
    except (OSError, zipfile.BadZipFile) as exc:
        findings.append(Finding("invalid_archive", str(archive_path), str(exc)))

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest() if archive_path.is_file() else None
    return {
        "schema_version": "p26-submission-archive-audit-v1",
        "archive": archive_path.name,
        "sha256": digest,
        "passed": not findings,
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Submission ZIP to inspect")
    args = parser.parse_args()
    report = audit_submission_archive(args.archive)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
