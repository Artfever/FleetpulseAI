"""Audit saved-model replay identities using the packaged FD001/train subset."""

from __future__ import annotations

import json
import math
from pathlib import Path

from fleetpulse.data.loader import load_engine_history
from fleetpulse.services.history import (
    _REPLAY_CACHE,
    clear_history_replay_cache,
    replay_engine_history,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = load_engine_history(root / "data" / "demo" / "fd001_training_prefixes.txt")
    manifest_path = root / "artifacts" / "development" / "decision_tree_validation.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))["payload"]
    expected_version = payload["model"]["version"]
    source_sha256 = payload["training"]["source_sha256"]
    model_sha256 = payload["model"]["sha256"]

    clear_history_replay_cache()
    for cutoff in (162, 164):
        replay = replay_engine_history(
            records,
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=cutoff,
            replay_cutoffs=[cutoff],
            manifest_path=manifest_path,
        )
        point = replay.points[0]
        prediction = point.prediction
        if (
            replay.identity.dataset_partition != "FD001/train"
            or replay.identity.unit_id != 4
            or point.cutoff_cycle != cutoff
            or prediction.model_version != expected_version
            or not math.isfinite(prediction.raw_rul_cycles)
        ):
            raise RuntimeError(f"Unexpected replay identity or output at cycle {cutoff}.")
        # Repeating the same exact request must reuse the one cache entry.
        replay_engine_history(
            records,
            dataset_partition="FD001/train",
            unit_id=4,
            selected_cycle=cutoff,
            replay_cutoffs=[cutoff],
            manifest_path=manifest_path,
        )
        if len(_REPLAY_CACHE) != len((162, 164)[: (1 if cutoff == 162 else 2)]):
            raise RuntimeError("Replay cache did not retain exactly one entry per distinct cutoff.")
        print(
            f"engine=4 cutoff={cutoff} model={prediction.model_name} "
            f"version={prediction.model_version} raw_rul_cycles={prediction.raw_rul_cycles:.14g}"
        )

    print(f"source_sha256={source_sha256}")
    print(f"model_sha256={model_sha256}")
    print(f"cache_entries={len(_REPLAY_CACHE)}")


if __name__ == "__main__":
    main()
