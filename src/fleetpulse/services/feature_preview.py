"""UI-free orchestration for selected-engine feature previews."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral

from fleetpulse.features import EngineFeatureVector, FeatureConfig, engineer_features


@dataclass(frozen=True)
class EngineFeaturePreview:
    """Visible past-only input rows and their ordered feature vector."""

    visible_history: tuple[dict[str, object], ...]
    vector: EngineFeatureVector


def build_engine_feature_preview(
    history: Iterable[Mapping[str, object]],
    *,
    dataset_partition: str,
    unit_id: int,
    selected_cycle: int,
    config: FeatureConfig | None = None,
) -> EngineFeaturePreview:
    """Build one engine's feature preview from rows through an observed cutoff.

    The service materializes a private snapshot, delegates all numeric feature
    work to the shared engineering function, and returns only rows at or before
    the selected cycle for display. It has no Streamlit dependency.
    """
    rows: list[dict[str, object]] = []
    for row_number, record in enumerate(history, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"History row {row_number} must be a mapping.")
        rows.append(dict(record))

    if (
        isinstance(selected_cycle, bool)
        or not isinstance(selected_cycle, Integral)
        or selected_cycle < 1
    ):
        raise ValueError("selected_cycle must be a positive integer.")
    prefix: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=1):
        cycle = row.get("cycle")
        if isinstance(cycle, bool) or not isinstance(cycle, Integral) or cycle < 1:
            raise ValueError(f"History row {row_number} cycle must be a positive integer.")
        if cycle <= selected_cycle:
            prefix.append(row)

    vector = engineer_features(
        prefix,
        dataset_partition=dataset_partition,
        unit_id=unit_id,
        selected_cycle=selected_cycle,
        config=config,
    )
    return EngineFeaturePreview(visible_history=tuple(prefix), vector=vector)
