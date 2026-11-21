"""Classify whether a training-engine replay was seen during model fitting."""

from collections.abc import Mapping
from numbers import Integral


def classify_training_engine(
    split_manifest: Mapping[str, object], *, dataset_partition: str, unit_id: int
) -> str:
    """Return ``fitting``, ``validation``, or ``unknown`` for a saved split.

    Validation identities are taken from the saved validation-prefix table;
    target values are not read or needed to determine model-fit membership.
    """
    if isinstance(unit_id, bool) or not isinstance(unit_id, Integral) or unit_id <= 0:
        raise ValueError("unit_id must be a positive integer.")
    if not isinstance(dataset_partition, str) or not dataset_partition.strip():
        raise ValueError("dataset_partition must be a nonempty string.")

    fitting = _identities(split_manifest, "fitting_engines", dataset_partition)
    validation = _identities(split_manifest, "validation_prefixes", dataset_partition)
    if fitting & validation:
        raise ValueError("Fitting and validation engine identities overlap.")
    key = (dataset_partition, int(unit_id))
    if key in fitting:
        return "fitting"
    if key in validation:
        return "validation"
    return "unknown"


def _identities(
    manifest: Mapping[str, object], field: str, dataset_partition: str
) -> set[tuple[str, int]]:
    rows = manifest.get(field)
    if not isinstance(rows, list):
        raise TypeError(f"Split manifest {field} must be a list.")
    identities = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"Split manifest {field} contains an invalid identity.")
        partition, unit = row.get("dataset_partition"), row.get("unit_id")
        if (
            not isinstance(partition, str)
            or isinstance(unit, bool)
            or not isinstance(unit, Integral)
        ):
            raise TypeError(f"Split manifest {field} contains an invalid identity.")
        if not partition.strip() or unit <= 0:
            raise ValueError(f"Split manifest {field} contains an invalid identity.")
        key = (partition, int(unit))
        if key in identities:
            raise ValueError(f"Split manifest {field} contains a duplicate identity.")
        identities.add(key)
    return identities
