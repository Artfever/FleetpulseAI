"""Build the checksummed feature-only FD001 reference used by hosted SHAP."""

from __future__ import annotations

from fleetpulse import PROJECT_ROOT
from fleetpulse.models.final_tree_bundle import load_final_tree_bundle
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services.explanations import (
    BACKGROUND_ARTIFACT_PATH,
    DECISION_TREE_KEY,
    build_fd001_training_background,
    save_training_feature_background,
)


def main() -> None:
    model = FINAL_MODEL_REGISTRY[DECISION_TREE_KEY]
    bundle = load_final_tree_bundle(PROJECT_ROOT / model.manifest_path)
    source = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
    background = build_fd001_training_background(
        source,
        expected_training_source_sha256=bundle.payload["training"]["training_source_sha256"],
        feature_config=bundle.feature_config,
    )
    digest = save_training_feature_background(
        background,
        PROJECT_ROOT / BACKGROUND_ARTIFACT_PATH,
    )
    output = PROJECT_ROOT / BACKGROUND_ARTIFACT_PATH
    print(f"output: {output.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"bytes: {output.stat().st_size}")
    print(f"rows: {len(background.values)}")
    print(f"features: {len(background.feature_names)}")
    print(f"selection_policy: {background.selection_policy}")
    print(f"training_source_sha256: {background.training_source_sha256}")
    print(f"background_sha256: {background.background_sha256}")
    print(f"artifact_payload_sha256: {digest}")


if __name__ == "__main__":
    main()
