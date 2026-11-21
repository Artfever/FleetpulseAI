"""Export one tracked simulated FD001 prefix as a headered upload CSV."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import HISTORY_COLUMNS, load_engine_history


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a headered CSV from the tracked FD001/train demo subset. "
            "The output contains raw history only, with no target or RUL label."
        )
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        default=4,
        help="training engine to export (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output CSV path (default: a named file in the system temp directory)",
    )
    args = parser.parse_args()

    source = PROJECT_ROOT / "data" / "demo" / "fd001_training_prefixes.txt"
    records = [
        row
        for row in load_engine_history(source)
        if row["unit_id"] == args.unit_id
    ]
    if not records:
        parser.error(f"engine {args.unit_id} is not present in the tracked demo subset")

    output = args.output or (
        Path(tempfile.gettempdir()) / f"fleetpulse-fd001-train-engine-{args.unit_id}.csv"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    last_cycle = records[-1]["cycle"]
    print(
        f"Wrote {len(records)} raw-history rows for FD001/train engine "
        f"{args.unit_id} through cycle {last_cycle} to {output}. "
        "No target or RUL label was included."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
