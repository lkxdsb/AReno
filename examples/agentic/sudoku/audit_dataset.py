"""Audit Sudoku JSONL splits for leakage, metadata drift, and optional uniqueness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import clues_are_preserved, count_solutions, is_solved  # noqa: E402


def audit(paths: list[Path], *, verify_solutions: bool = False) -> dict:
    """Return a machine-readable audit and fail-safe error list."""

    errors = []
    files = {}
    seen_ids: dict[str, str] = {}
    seen_puzzles: dict[str, str] = {}
    seen_solutions: dict[str, str] = {}
    observed_splits: dict[str, str] = {}
    for path in paths:
        rows = _read_rows(path)
        difficulties = Counter(str(row.get("difficulty")) for row in rows)
        splits = {str(row.get("split")) for row in rows}
        files[str(path)] = {
            "records": len(rows),
            "splits": sorted(splits),
            "difficulties": dict(sorted(difficulties.items())),
        }
        if len(splits) != 1 or "None" in splits:
            errors.append(f"{path}: expected exactly one declared split")
        else:
            split = next(iter(splits))
            previous = observed_splits.get(split)
            if previous is not None:
                errors.append(f"{path}: split {split!r} is already supplied by {previous}")
            observed_splits[split] = str(path)

        for line_number, row in enumerate(rows, start=1):
            location = f"{path}:{line_number}"
            puzzle = str(row.get("puzzle", ""))
            solution = str(row.get("solution", ""))
            _record_unique(errors, seen_ids, str(row.get("id", "")), location, "id")
            _record_unique(errors, seen_puzzles, puzzle, location, "puzzle")
            _record_unique(errors, seen_solutions, solution, location, "solution")
            _check_hash(errors, row, "puzzle_hash", puzzle, location)
            _check_hash(errors, row, "solution_hash", solution, location)
            if verify_solutions:
                try:
                    if not is_solved(solution):
                        errors.append(f"{location}: solution is not a valid completed board")
                    elif not clues_are_preserved(puzzle, solution):
                        errors.append(f"{location}: solution does not preserve givens")
                    elif count_solutions(puzzle, limit=2) != 1:
                        errors.append(f"{location}: puzzle is not uniquely solvable")
                except (TypeError, ValueError) as exc:
                    errors.append(f"{location}: invalid board: {exc}")
    return {"ok": not errors, "files": files, "errors": errors}


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _record_unique(
    errors: list[str],
    seen: dict[str, str],
    value: str,
    location: str,
    field: str,
) -> None:
    if not value:
        errors.append(f"{location}: missing {field}")
        return
    previous = seen.get(value)
    if previous is not None:
        errors.append(f"{location}: duplicate {field}; first seen at {previous}")
    else:
        seen[value] = location


def _check_hash(errors: list[str], row: dict, field: str, value: str, location: str) -> None:
    expected = row.get(field)
    actual = hashlib.sha256(value.encode("ascii", errors="ignore")).hexdigest()[:16]
    if expected != actual:
        errors.append(f"{location}: {field} mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--verify-solutions", action="store_true")
    args = parser.parse_args()
    result = audit(args.paths, verify_solutions=args.verify_solutions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
