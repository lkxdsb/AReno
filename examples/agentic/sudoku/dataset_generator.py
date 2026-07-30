"""Generate deterministic, uniquely solvable Sudoku training records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    BOARD_SIZE,
    DEFAULT_MAX_ACTIONS,
    DIFFICULTIES,
    EMPTY,
    MAX_ACTIONS_LIMIT,
    SolveStats,
    count_solutions,
    encode_board,
    is_solved,
)

TARGET_CLUES = {
    "easy": 40,
    "medium": 34,
    "hard": 28,
}
SPLITS = ("train", "validation", "test")


def generate_records(
    count: int = 300,
    *,
    seed: int = 2026,
    split: str = "train",
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> list[dict]:
    """Generate a balanced deterministic split across three audited difficulty levels."""

    if count <= 0:
        raise ValueError("count must be positive")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if not 1 <= max_actions <= MAX_ACTIONS_LIMIT:
        raise ValueError(f"max_actions must be between 1 and {MAX_ACTIONS_LIMIT}")
    rng = random.Random(seed)
    records = []
    seen: set[str] = set()
    for index in range(count):
        difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
        puzzle, solution, stats = generate_puzzle(difficulty, rng=rng, seen=seen)
        puzzle_text = encode_board(puzzle)
        solution_text = encode_board(solution)
        empty_cells = puzzle_text.count("0")
        records.append(
            {
                "id": f"sudoku-{split}-{difficulty}-{index + 1:05d}",
                "split": split,
                "puzzle": puzzle_text,
                "solution": solution_text,
                "puzzle_hash": _fingerprint(puzzle_text),
                "solution_hash": _fingerprint(solution_text),
                "difficulty": difficulty,
                "difficulty_method": "clue_count_and_uniqueness_search_v2",
                "clue_count": TARGET_CLUES[difficulty],
                "solver_search_nodes": stats.visited_nodes,
                "solver_search_overhead": stats.visited_nodes - (empty_cells + 1),
                "solver_guesses": stats.guesses,
                "solver_backtracks": stats.backtracks,
                "max_actions": max_actions,
                "seed": seed,
            }
        )
    return records


def generate_puzzle(
    difficulty: str,
    *,
    rng: random.Random,
    seen: set[str] | None = None,
    max_attempts: int = 200,
) -> tuple[list[list[int]], list[list[int]], SolveStats]:
    """Generate one unique puzzle matching deterministic clue and search thresholds."""

    if difficulty not in TARGET_CLUES:
        raise ValueError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
    target_clues = TARGET_CLUES[difficulty]
    seen = seen if seen is not None else set()
    for _ in range(max_attempts):
        solution = _random_solution(rng)
        puzzle = _remove_clues(solution, target_clues=target_clues, rng=rng)
        encoded = encode_board(puzzle)
        stats = SolveStats()
        solution_count = count_solutions(puzzle, limit=2, stats=stats)
        empty_cells = BOARD_SIZE * BOARD_SIZE - _clue_count(puzzle)
        if (
            solution_count == 1
            and _clue_count(puzzle) == target_clues
            and _matches_difficulty(difficulty, stats, empty_cells=empty_cells)
            and encoded not in seen
        ):
            seen.add(encoded)
            return puzzle, solution, stats
    raise RuntimeError(f"could not generate a unique {difficulty} puzzle after {max_attempts} attempts")


def _random_solution(rng: random.Random) -> list[list[int]]:
    """Return a diverse valid board using randomized MRV backtracking."""

    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    row_masks = [0] * BOARD_SIZE
    col_masks = [0] * BOARD_SIZE
    box_masks = [0] * BOARD_SIZE
    if not _fill_random_solution(board, row_masks, col_masks, box_masks, rng, remaining=81):
        raise RuntimeError("randomized Sudoku fill unexpectedly failed")
    if not is_solved(board):
        raise RuntimeError("randomized Sudoku fill produced an invalid solution")
    return board


def _remove_clues(
    solution: list[list[int]],
    *,
    target_clues: int,
    rng: random.Random,
) -> list[list[int]]:
    puzzle = [list(row) for row in solution]
    positions = list(range(BOARD_SIZE * BOARD_SIZE))
    rng.shuffle(positions)
    clues = BOARD_SIZE * BOARD_SIZE
    for position in positions:
        if clues <= target_clues:
            break
        row, col = divmod(position, BOARD_SIZE)
        original = puzzle[row][col]
        puzzle[row][col] = EMPTY
        if count_solutions(puzzle, limit=2) == 1:
            clues -= 1
        else:
            puzzle[row][col] = original
    return puzzle


def _clue_count(board: list[list[int]]) -> int:
    return sum(value != EMPTY for row in board for value in row)


def _matches_difficulty(difficulty: str, stats: SolveStats, *, empty_cells: int) -> bool:
    search_overhead = stats.visited_nodes - (empty_cells + 1)
    if difficulty == "easy":
        return search_overhead == 0 and stats.guesses == 0 and stats.backtracks == 0
    if difficulty == "medium":
        return 5 <= search_overhead < 100
    return search_overhead >= 100


def _fill_random_solution(
    board: list[list[int]],
    row_masks: list[int],
    col_masks: list[int],
    box_masks: list[int],
    rng: random.Random,
    *,
    remaining: int,
) -> bool:
    if remaining == 0:
        return True
    all_digits_mask = (1 << BOARD_SIZE) - 1
    best_size = BOARD_SIZE + 1
    best_cells: list[tuple[int, int, int]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            box = (row // 3) * 3 + col // 3
            allowed_mask = all_digits_mask & ~(row_masks[row] | col_masks[col] | box_masks[box])
            allowed_count = allowed_mask.bit_count()
            if allowed_count == 0:
                return False
            if allowed_count < best_size:
                best_size = allowed_count
                best_cells = [(row, col, allowed_mask)]
            elif allowed_count == best_size:
                best_cells.append((row, col, allowed_mask))

    row, col, allowed_mask = rng.choice(best_cells)
    digits = [digit for digit in range(1, BOARD_SIZE + 1) if allowed_mask & (1 << (digit - 1))]
    rng.shuffle(digits)
    box = (row // 3) * 3 + col // 3
    for digit in digits:
        bit = 1 << (digit - 1)
        board[row][col] = digit
        row_masks[row] |= bit
        col_masks[col] |= bit
        box_masks[box] |= bit
        if _fill_random_solution(
            board,
            row_masks,
            col_masks,
            box_masks,
            rng,
            remaining=remaining - 1,
        ):
            return True
        board[row][col] = EMPTY
        row_masks[row] ^= bit
        col_masks[col] ^= bit
        box_masks[box] ^= bit
    return False


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed, split=args.split, max_actions=args.max_actions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
