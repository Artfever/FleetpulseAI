"""UI-free summaries for observed sensor histories."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real


@dataclass(frozen=True)
class SensorHistorySummary:
    """Descriptive range and direction-change counts for one sensor trace."""

    pattern: str
    minimum: float
    maximum: float
    direction_reversals: int


def summarize_sensor_history(readings: Iterable[Real]) -> SensorHistorySummary:
    """Summarize observed values without assigning physical or fault meaning."""
    values = tuple(float(value) for value in readings)
    if not values:
        raise ValueError("At least one sensor reading is required.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Sensor readings must be finite.")

    differences = tuple(right - left for left, right in pairwise(values))
    reversals = sum(
        (left < 0 < right) or (left > 0 > right)
        for left, right in pairwise(differences)
    )
    minimum = min(values)
    maximum = max(values)
    return SensorHistorySummary(
        pattern="flat across observed cycles" if minimum == maximum else "varies across observed cycles",
        minimum=minimum,
        maximum=maximum,
        direction_reversals=reversals,
    )
