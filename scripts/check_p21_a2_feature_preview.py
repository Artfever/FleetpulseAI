"""Reproduce the packaged FD001/train Engine Explorer feature-preview example."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import build_engine_feature_preview

EXPECTED_SOURCE_SHA256 = "953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7"
EXPECTED_FEATURES = {
    "sensor_1_latest": 518.67,
    "sensor_1_mean": 518.67,
    "sensor_1_std": 0.0,
    "sensor_1_slope": 0.0,
    "sensor_4_latest": 1411.81,
    "sensor_4_mean": 1411.7286666666666,
    "sensor_4_std": 4.397945530459507,
    "sensor_4_slope": 0.2859310344827574,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "data" / "demo" / "fd001_training_prefixes.txt"
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Packaged FD001/train example changed; review expected audit values.")

    records = load_engine_history(source_path)
    engine_rows = [row for row in records if row["unit_id"] == 4]
    preview = build_engine_feature_preview(
        engine_rows,
        dataset_partition="FD001/train",
        unit_id=4,
        selected_cycle=164,
    )
    if (
        len(preview.visible_history) != 164
        or preview.vector.window_row_count != 30
        or len(preview.vector.values) != 96
        or preview.vector.engine_key != ("FD001/train", 4)
    ):
        raise RuntimeError("Packaged feature-preview identity or dimensions changed.")

    features = preview.vector.as_dict()
    for name, expected in EXPECTED_FEATURES.items():
        if not math.isclose(features[name], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected {name}: {features[name]!r}.")
        print(f"{name}={features[name]:.16g}")
    print(f"source_sha256={source_sha256}")
    print("engine=FD001/train:4 selected_cycle=164 window_rows=30 feature_count=96")


if __name__ == "__main__":
    main()
