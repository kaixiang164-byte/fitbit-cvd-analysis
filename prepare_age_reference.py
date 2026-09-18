#!/usr/bin/env python3
"""Package the extracted M1 export for the unchanged age-sensitivity reader.

This new compatibility adapter does not fit a model or reconstruct study data.
It reads m1_primary_pairwise_results_python.csv from an explicit runtime
directory and writes manuscript_multinomial_pairwise_results.csv there.
Only the model label changes: 'M1 primary' becomes 'M1_primary'. All rows,
column order, and other cell strings are preserved, including numerical text.
The resulting reference contains M1 only; it is not a historical combined
M1/M2 export. Runtime files must remain outside the code-release directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

SOURCE_FILENAME = "m1_primary_pairwise_results_python.csv"
DESTINATION_FILENAME = "manuscript_multinomial_pairwise_results.csv"
NUMERIC_COLUMNS = ("log_OR", "SE", "OR", "CI_lower", "CI_upper", "p_value", "n")


def prepare_reference(directory: str | Path) -> tuple[Path, bool]:
    """Return (destination, created); refuse to replace a different reference."""
    runtime_directory = Path(directory).expanduser().resolve(strict=True)
    code_directory = Path(__file__).resolve().parent
    if runtime_directory == code_directory or code_directory in runtime_directory.parents:
        raise ValueError("Choose a runtime directory outside the code release.")
    if not runtime_directory.is_dir():
        raise ValueError("The runtime directory must already exist.")
    source = runtime_directory / SOURCE_FILENAME
    destination = runtime_directory / DESTINATION_FILENAME
    if destination.is_symlink():
        raise ValueError("The destination must not be a symbolic link.")
    with source.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.reader(handle))
    if not records or len(records) < 2:
        raise ValueError("The generated M1 export must contain a header and data rows.")
    header, rows = records[0], records[1:]
    required = {"model", "term", "contrast", *NUMERIC_COLUMNS}
    if len(header) != len(set(header)) or not required.issubset(header):
        raise ValueError("The generated M1 export has duplicate or missing required columns.")
    positions = {name: index for index, name in enumerate(header)}
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"Invalid column count in M1 export row {row_number}.")
        if row[positions["model"]] != "M1 primary":
            raise ValueError(f"Expected model='M1 primary' in row {row_number}.")
        key = (row[positions["term"]], row[positions["contrast"]])
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate term/contrast in row {row_number}.")
        seen.add(key)
        for name in NUMERIC_COLUMNS:
            try:
                finite = math.isfinite(float(row[positions[name]]))
            except ValueError:
                finite = False
            if not finite:
                raise ValueError(f"Non-finite or invalid {name} in row {row_number}.")
        row[positions["model"]] = "M1_primary"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    expected = buffer.getvalue().encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != expected:
            raise FileExistsError("A different age reference exists; no file was changed.")
        return destination, False
    # Exclusive creation protects a concurrently created destination.
    with destination.open("xb") as handle:
        handle.write(expected)
    return destination, True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path.cwd(),
        help="Runtime directory containing generated M1 results (default: working directory).",
    )
    args = parser.parse_args(argv)
    destination, created = prepare_reference(args.directory)
    print(("Created " if created else "Verified identical existing ") + destination.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
