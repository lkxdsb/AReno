"""Dataset loader for the Sudoku agentic training example."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    DEFAULT_MAX_ACTIONS,
    DIFFICULTIES,
    MAX_ACTIONS_LIMIT,
    SolveStats,
    clues_are_preserved,
    count_solutions,
    encode_board,
    is_solved,
    make_prompt,
)


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize and fully validate puzzle records before model initialization."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        record_id = str(record.get("id", f"sudoku-{index:05d}"))
        try:
            puzzle = encode_board(record["puzzle"])
            solution = encode_board(record["solution"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Sudoku record {record_id} has an invalid puzzle or solution: {exc}") from exc
        difficulty = str(record.get("difficulty", ""))
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Sudoku record {record_id} difficulty must be one of {', '.join(DIFFICULTIES)}")
        max_actions = record.get("max_actions", DEFAULT_MAX_ACTIONS)
        if (
            not isinstance(max_actions, int)
            or isinstance(max_actions, bool)
            or not 1 <= max_actions <= MAX_ACTIONS_LIMIT
        ):
            raise ValueError(
                f"Sudoku record {record_id} max_actions must be an integer between 1 and {MAX_ACTIONS_LIMIT}"
            )
        if not is_solved(solution):
            raise ValueError(f"Sudoku record {record_id} solution is not a completed valid board")
        if is_solved(puzzle):
            raise ValueError(f"Sudoku record {record_id} puzzle must contain at least one empty cell")
        if not clues_are_preserved(puzzle, solution):
            raise ValueError(f"Sudoku record {record_id} solution does not preserve puzzle givens")
        stats = SolveStats()
        if count_solutions(puzzle, limit=2, stats=stats) != 1:
            raise ValueError(f"Sudoku record {record_id} puzzle must have exactly one solution")
        _validate_generated_metadata(record_id, record, puzzle, solution, difficulty, stats)

        record.update(
            {
                "id": record_id,
                "puzzle": puzzle,
                "solution": solution,
                "difficulty": difficulty,
                "max_actions": max_actions,
            }
        )
        record["prompt"] = make_prompt(record)
        if solution in record["prompt"]:
            raise ValueError(f"Sudoku record {record_id} prompt exposes the solution")
        records.append(record)
    return records


def _validate_generated_metadata(
    record_id: str,
    record: dict,
    puzzle: str,
    solution: str,
    difficulty: str,
    stats: SolveStats,
) -> None:
    split = record.get("split")
    if split is not None and split not in {"train", "validation", "test"}:
        raise ValueError(f"Sudoku record {record_id} split must be train, validation, or test")
    for field, value in (("puzzle_hash", puzzle), ("solution_hash", solution)):
        expected = record.get(field)
        actual = hashlib.sha256(value.encode("ascii")).hexdigest()[:16]
        if expected is not None and expected != actual:
            raise ValueError(f"Sudoku record {record_id} {field} does not match its board")

    clue_count = sum(cell != "0" for cell in puzzle)
    if "clue_count" in record and record["clue_count"] != clue_count:
        raise ValueError(f"Sudoku record {record_id} clue_count does not match its puzzle")
    method = record.get("difficulty_method")
    if method == "curriculum_empty_cells_v1":
        curriculum_empty_cells = record.get("curriculum_empty_cells")
        if (
            not isinstance(curriculum_empty_cells, int)
            or isinstance(curriculum_empty_cells, bool)
            or curriculum_empty_cells != puzzle.count("0")
        ):
            raise ValueError(f"Sudoku record {record_id} curriculum_empty_cells does not match its puzzle")
        _validate_solver_stats(record_id, record, puzzle, stats)
        return
    if method != "clue_count_and_uniqueness_search_v2":
        return

    expected_clues = {"easy": 40, "medium": 34, "hard": 28}[difficulty]
    if clue_count != expected_clues:
        raise ValueError(f"Sudoku record {record_id} clue count does not match v2 difficulty")
    empty_cells = len(puzzle) - clue_count
    search_overhead = stats.visited_nodes - (empty_cells + 1)
    expected_difficulty = "easy" if search_overhead == 0 else "medium" if 5 <= search_overhead < 100 else "hard"
    if 0 < search_overhead < 5:
        raise ValueError(f"Sudoku record {record_id} search overhead falls outside v2 difficulty bands")
    if difficulty != expected_difficulty:
        raise ValueError(f"Sudoku record {record_id} difficulty does not match v2 search effort")
    _validate_solver_stats(record_id, record, puzzle, stats)


def _validate_solver_stats(
    record_id: str,
    record: dict,
    puzzle: str,
    stats: SolveStats,
) -> None:
    expected_stats = {
        "solver_search_nodes": stats.visited_nodes,
        "solver_search_overhead": stats.visited_nodes - (puzzle.count("0") + 1),
        "solver_guesses": stats.guesses,
        "solver_backtracks": stats.backtracks,
    }
    for field, actual in expected_stats.items():
        if field in record and record[field] != actual:
            raise ValueError(f"Sudoku record {record_id} {field} does not match deterministic analysis")
