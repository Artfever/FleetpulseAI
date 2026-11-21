"""Audit the local Model Laboratory scores against the frozen P13 report/bundles."""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.final_test_display import (
    FINAL_TEST_REPORT_PATH,
    load_final_test_display,
)
from scripts.audit_p13_bundles import (
    MLP_DIR,
    RESULTS_PATH,
    TREE_DIR,
    TREE_INDEX_PATH,
    audit_bundles,
)


def audit_metric_display() -> dict[str, object]:
    """Require the rendered table, sealed P13 scores, and frozen bundles to agree."""
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app/pages/5_Model_Laboratory.py"), default_timeout=60
    ).run()
    if app.exception:
        raise RuntimeError(f"Model Laboratory raised an exception: {app.exception[0].message}")

    official_tables = [
        item.value
        for item in app.dataframe
        if hasattr(item.value, "columns")
        and {"MAE (cycles)", "RMSE (cycles)"}.issubset(item.value.columns)
        and len(item.value) == 5
    ]
    if len(official_tables) != 1:
        raise ValueError(f"Expected one five-model official table; found {len(official_tables)}.")

    rendered = official_tables[0].to_dict(orient="records")
    expected = [
        {
            "Model": row.display_name,
            "Model version": row.version,
            "Sample count (engines)": row.sample_count,
            "MAE (cycles)": row.mae_cycles,
            "RMSE (cycles)": row.rmse_cycles,
        }
        for row in load_final_test_display(PROJECT_ROOT / FINAL_TEST_REPORT_PATH)
    ]
    if rendered != expected:
        raise ValueError("The Model Laboratory table differs from the verified P13 report.")

    bundle_audit = audit_bundles(TREE_INDEX_PATH, TREE_DIR, MLP_DIR, RESULTS_PATH)
    if not bundle_audit["evaluated_models_match_frozen_inventory"]:
        raise ValueError("P13 evaluated models differ from the frozen final bundle inventory.")

    return {
        "page_exceptions": 0,
        "display_rows_match_verified_report": True,
        "frozen_bundle_audit_passed": True,
        "models": rendered,
        "bundle_files_checked": bundle_audit["bundle_files_checked"],
        "report_payload_sha256": bundle_audit["reported_results_payload_sha256"],
        "tree_index_payload_sha256": bundle_audit["tree_bundle_index_payload_sha256"],
        "test_artifact_used_for_training_or_tuning": False,
    }


def main() -> None:
    print(json.dumps(audit_metric_display(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
