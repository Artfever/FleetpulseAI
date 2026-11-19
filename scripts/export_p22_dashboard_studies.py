"""Materialize the verified P22-A1/A2 scheduler runs as UI-readable snapshots.

Requires the ignored local P13 report. The source runners read only raw RUL
predictions and never use official-test labels for scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.p22_study_display import P13_REPORT_SHA256

if __package__:
    from .run_p22_a1_fleet_scenarios import load_raw_predictions, run_experiment
    from .run_p22_a2_seed_comparison import run_comparison
else:
    from run_p22_a1_fleet_scenarios import load_raw_predictions, run_experiment
    from run_p22_a2_seed_comparison import run_comparison

P13_REPORT_PATH = PROJECT_ROOT / "artifacts/final/p13/final_test_results.json"


def _sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_snapshot(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "sha256": _sha256(payload)}
    path.write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_studies(output_dir: Path) -> tuple[Path, Path]:
    if not P13_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Verified local P13 prediction report is unavailable: {P13_REPORT_PATH}")
    predictions, provenance = load_raw_predictions(P13_REPORT_PATH)
    if provenance.get("report_sha256") != P13_REPORT_SHA256:
        raise ValueError("Local P13 prediction report differs from the frozen P22 input source")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
    }

    fixed = run_experiment(predictions)
    fixed["input_provenance"] = provenance
    fixed["benchmark_environment"] = environment
    fixed_path = output_dir / "p22_a1_fixed_fleet_scenarios.json"
    _write_snapshot(fixed, fixed_path)

    seeds = run_comparison(predictions)
    seeds["input_provenance"] = provenance
    seeds["benchmark_environment"] = environment
    seed_path = output_dir / "p22_a2_seed_comparison.json"
    _write_snapshot(seeds, seed_path)
    return fixed_path, seed_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs")
    args = parser.parse_args()
    for path in export_studies(args.output_dir):
        print(f"Wrote verified study snapshot to {path}")


if __name__ == "__main__":
    main()
