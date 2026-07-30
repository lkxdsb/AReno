"""Generate near-complete Sudoku records for short agentic curriculum runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_generator import SPLITS, _fingerprint, _random_solution  # noqa: E402
from game import (  # noqa: E402
    DEFAULT_MAX_ACTIONS,
    MAX_ACTIONS_LIMIT,
    SolveStats,
    count_solutions,
    encode_board,
)


def generate_curriculum_records(
    count: int = 300,
    *,
    seed: int = 2026,
    split: str = "train",
    empty_cells: int = 3,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> list[dict]:
    """Generate deterministic unique puzzles with a small fixed number of blanks."""

    if count <= 0:
        raise ValueError("count must be positive")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if not 1 <= empty_cells <= 64:
        raise ValueError("empty_cells must be between 1 and 64")
    if not 1 <= max_actions <= MAX_ACTIONS_LIMIT:
        raise ValueError(f"max_actions must be between 1 and {MAX_ACTIONS_LIMIT}")

    rng = random.Random(seed)
    records = []
    seen_puzzles: set[str] = set()
    seen_solutions: set[str] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not generate enough unique curriculum puzzles")
        solution = _random_solution(rng)
        solution_text = encode_board(solution)
        if solution_text in seen_solutions:
            continue
        puzzle = [list(row) for row in solution]
        for position in rng.sample(range(81), empty_cells):
            row, col = divmod(position, 9)
            puzzle[row][col] = 0
        puzzle_text = encode_board(puzzle)
        stats = SolveStats()
        if puzzle_text in seen_puzzles or count_solutions(puzzle, limit=2, stats=stats) != 1:
            continue

        seen_puzzles.add(puzzle_text)
        seen_solutions.add(solution_text)
        index = len(records) + 1
        records.append(
            {
                "id": f"sudoku-{split}-curriculum-e{empty_cells}-{index:05d}",
                "split": split,
                "puzzle": puzzle_text,
                "solution": solution_text,
                "puzzle_hash": _fingerprint(puzzle_text),
                "solution_hash": _fingerprint(solution_text),
                "difficulty": "easy",
                "difficulty_method": "curriculum_empty_cells_v1",
                "curriculum_empty_cells": empty_cells,
                "clue_count": 81 - empty_cells,
                "solver_search_nodes": stats.visited_nodes,
                "solver_search_overhead": stats.visited_nodes - (empty_cells + 1),
                "solver_guesses": stats.guesses,
                "solver_backtracks": stats.backtracks,
                "max_actions": max_actions,
                "seed": seed,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--empty-cells", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()

    records = generate_curriculum_records(
        args.count,
        seed=args.seed,
        split=args.split,
        empty_cells=args.empty_cells,
        max_actions=args.max_actions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
