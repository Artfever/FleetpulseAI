"""Load and validate the compact P11 model-selection comparison manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "p11-selection-manifest-v1"


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    """Return an integrity-checked comparison manifest, or raise on invalid data."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if document.get("sha256") != digest:
        raise ValueError("selection manifest content hash mismatch")
    _validate(payload)
    return payload


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported selection manifest schema")
    if payload.get("dataset_partition") != "FD001/train":
        raise ValueError("selection manifest must describe FD001/train")
    if payload.get("validation_sample_unit") != "engine-prefix":
        raise ValueError("selection manifest sample unit mismatch")
    if payload.get("validation_engine_count") != 20:
        raise ValueError("selection manifest must describe 20 validation prefixes")
    if payload.get("official_test_data_opened") is not False:
        raise ValueError("selection manifest must explicitly exclude official test data")
    for key in ("training_source_sha256", "split_manifest_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid {key}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("selection manifest must contain the six declared comparison rows")
    names = set()
    for candidate in candidates:
        name = candidate["name"]
        if not isinstance(name, str) or name in names:
            raise ValueError("selection candidate names must be unique")
        names.add(name)
        if candidate.get("sample_count") != 20:
            raise ValueError(f"selection candidate sample count mismatch: {name}")
        if candidate.get("training_source_sha256") != payload["training_source_sha256"]:
            raise ValueError(f"selection candidate source mismatch: {name}")
        if candidate.get("split_manifest_sha256") != payload["split_manifest_sha256"]:
            raise ValueError(f"selection candidate split mismatch: {name}")
        for metric in ("mae_cycles", "rmse_cycles"):
            value = candidate.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {metric} for {name}")
        run_hash = candidate.get("result_artifact_sha256")
        if not isinstance(run_hash, str) or len(run_hash) != 64:
            raise ValueError(f"invalid result artifact hash for {name}")
    decision = payload.get("decision", {})
    if decision.get("status") != "owner_selected_candidate":
        raise ValueError("selection manifest must record the owner-delegated model choice")
    candidate_names = {candidate["name"] for candidate in candidates}
    for field in ("selected_candidate", "lowest_mae_candidate", "lowest_rmse_candidate"):
        if decision.get(field) not in candidate_names:
            raise ValueError(f"selection decision references an unknown candidate: {field}")
    if not decision.get("selection_authority") or not decision.get("reason") or not decision.get(
        "tradeoff"
    ):
        raise ValueError("selection decision must record its authority, rationale, and trade-off")
    selected_artifact = decision.get("selected_artifact", {})
    if not selected_artifact.get("name") or not selected_artifact.get("version"):
        raise ValueError("selected artifact must include its model name and version")
    if selected_artifact.get("sha256") != selected_artifact.get(
        "matched_p11_run_checkpoint_sha256"
    ):
        raise ValueError("selected model checkpoint does not match its P11 comparison run")
    for field in ("sha256", "matched_p11_run_checkpoint_sha256"):
        value = selected_artifact.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid selected artifact {field}")
