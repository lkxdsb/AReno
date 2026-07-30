"""Deterministic Sudoku rules and bounded tool environment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

BOARD_SIZE = 9
BOX_SIZE = 3
EMPTY = 0
DIGITS = frozenset(range(1, BOARD_SIZE + 1))
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_MAX_ACTIONS = 96
MAX_ACTIONS_LIMIT = 256

Board = list[list[int]]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_candidates",
            "description": "Return every digit currently allowed by row, column, and box constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "minimum": 1, "maximum": 9},
                    "col": {"type": "integer", "minimum": 1, "maximum": 9},
                },
                "required": ["row", "col"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_digit",
            "description": "Place one locally legal digit in an empty, non-given cell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "minimum": 1, "maximum": 9},
                    "col": {"type": "integer", "minimum": 1, "maximum": 9},
                    "digit": {"type": "integer", "minimum": 1, "maximum": 9},
                },
                "required": ["row", "col", "digit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo the most recent successful place_digit action.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def normalize_board(value: str | Sequence[Sequence[int]] | Sequence[int]) -> Board:
    """Return a validated 9x9 integer board without checking Sudoku conflicts."""

    if isinstance(value, str):
        compact = "".join(char for char in value if not char.isspace())
        if len(compact) != BOARD_SIZE * BOARD_SIZE:
            raise ValueError("Sudoku board string must contain exactly 81 cells")
        if any(char not in "0123456789." for char in compact):
            raise ValueError("Sudoku board cells must be digits, 0, or .")
        flat = [EMPTY if char in {"0", "."} else int(char) for char in compact]
    else:
        values = list(value)
        if len(values) == BOARD_SIZE and all(_is_row(row) for row in values):
            flat = [cell for row in values for cell in row]
        else:
            flat = values
        if len(flat) != BOARD_SIZE * BOARD_SIZE:
            raise ValueError("Sudoku board must contain exactly 81 cells")
        if any(not _is_int_cell(cell) for cell in flat):
            raise ValueError("Sudoku board cells must be integers between 0 and 9")
        flat = [int(cell) for cell in flat]
    return [flat[index : index + BOARD_SIZE] for index in range(0, len(flat), BOARD_SIZE)]


def encode_board(board: str | Sequence[Sequence[int]] | Sequence[int]) -> str:
    """Encode a board as an 81-character row-major string."""

    return "".join(str(cell) for row in normalize_board(board) for cell in row)


def format_board(board: str | Sequence[Sequence[int]] | Sequence[int]) -> str:
    """Render a compact human-readable board with 1-based coordinates."""

    normalized = normalize_board(board)
    lines = ["    1 2 3   4 5 6   7 8 9"]
    for row_index, row in enumerate(normalized, start=1):
        if row_index in {4, 7}:
            lines.append("  +-------+-------+-------+")
        cells = ["." if value == EMPTY else str(value) for value in row]
        lines.append(f"{row_index} | {' '.join(cells[:3])} | {' '.join(cells[3:6])} | {' '.join(cells[6:])} |")
    return "\n".join(lines)


def make_prompt(record: dict[str, Any]) -> str:
    """Build a task prompt without exposing the stored solution."""

    board = normalize_board(record["puzzle"])
    max_actions = int(record.get("max_actions", DEFAULT_MAX_ACTIONS))
    difficulty = str(record.get("difficulty", "unknown"))
    return (
        f"Solve this {difficulty} 9x9 Sudoku in at most {max_actions} tool actions.\n"
        "Rows and columns use coordinates 1 through 9. Given cells cannot be changed. "
        "Use inspect_candidates, place_digit, and undo. Candidate inspection reports only digits "
        "allowed by the current row, column, and box; it does not reveal the hidden solution. "
        "Call exactly one tool per turn until the board is solved or the action budget is exhausted.\n\n"
        f"{format_board(board)}"
    )


def candidates(board: str | Sequence[Sequence[int]] | Sequence[int], row: int, col: int) -> list[int]:
    """Return locally legal digits for one empty cell using 1-based coordinates."""

    normalized = normalize_board(board)
    row_index, col_index = _coordinate(row, col)
    if normalized[row_index][col_index] != EMPTY:
        raise ValueError("candidate inspection requires an empty cell")
    used = set(normalized[row_index])
    used.update(normalized[index][col_index] for index in range(BOARD_SIZE))
    box_row = (row_index // BOX_SIZE) * BOX_SIZE
    box_col = (col_index // BOX_SIZE) * BOX_SIZE
    used.update(
        normalized[box_row + row_offset][box_col + col_offset]
        for row_offset in range(BOX_SIZE)
        for col_offset in range(BOX_SIZE)
    )
    return sorted(DIGITS - used)


def is_consistent(board: str | Sequence[Sequence[int]] | Sequence[int]) -> bool:
    """Return whether every filled cell respects row, column, and box constraints."""

    normalized = normalize_board(board)
    units: list[list[int]] = [list(row) for row in normalized]
    units.extend([[normalized[row][col] for row in range(BOARD_SIZE)] for col in range(BOARD_SIZE)])
    units.extend(
        [
            [
                normalized[box_row + row_offset][box_col + col_offset]
                for row_offset in range(BOX_SIZE)
                for col_offset in range(BOX_SIZE)
            ]
            for box_row in range(0, BOARD_SIZE, BOX_SIZE)
            for box_col in range(0, BOARD_SIZE, BOX_SIZE)
        ]
    )
    return all(_nonzero_values_are_unique(unit) for unit in units)


def is_solved(board: str | Sequence[Sequence[int]] | Sequence[int]) -> bool:
    """Return whether the board is a complete valid Sudoku solution."""

    normalized = normalize_board(board)
    return all(value != EMPTY for row in normalized for value in row) and is_consistent(normalized)


def clues_are_preserved(
    puzzle: str | Sequence[Sequence[int]] | Sequence[int],
    solution: str | Sequence[Sequence[int]] | Sequence[int],
) -> bool:
    """Return whether every nonempty puzzle cell matches the solution."""

    puzzle_board = normalize_board(puzzle)
    solution_board = normalize_board(solution)
    return all(
        puzzle_board[row][col] in {EMPTY, solution_board[row][col]}
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
    )


def solve(
    board: str | Sequence[Sequence[int]] | Sequence[int],
    *,
    stats: SolveStats | None = None,
) -> Board | None:
    """Return one deterministic solution using MRV backtracking."""

    normalized = normalize_board(board)
    if not is_consistent(normalized):
        return None
    working = [list(row) for row in normalized]
    if _search_one(working, stats or SolveStats()):
        return working
    return None


def count_solutions(
    board: str | Sequence[Sequence[int]] | Sequence[int],
    *,
    limit: int = 2,
    stats: SolveStats | None = None,
) -> int:
    """Count solutions up to a positive limit and optionally record full-search effort."""

    if limit <= 0:
        raise ValueError("solution count limit must be positive")
    normalized = normalize_board(board)
    if not is_consistent(normalized):
        return 0
    working = [list(row) for row in normalized]
    return _count_solutions(working, limit, stats)


@dataclass(slots=True)
class SolveStats:
    """Deterministic search-effort counters for diagnostics."""

    visited_nodes: int = 0
    guesses: int = 0
    backtracks: int = 0


@dataclass(slots=True)
class SudokuGame:
    """Stateful bounded environment for the three public Sudoku tools."""

    puzzle: str | Sequence[Sequence[int]] | Sequence[int]
    max_actions: int = DEFAULT_MAX_ACTIONS
    initial_board: Board = field(init=False)
    board: Board = field(init=False)
    actions_used: int = field(init=False, default=0)
    invalid_actions: int = field(init=False, default=0)
    history: list[tuple[int, int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        board = normalize_board(self.puzzle)
        if not is_consistent(board):
            raise ValueError("Sudoku puzzle givens conflict")
        if (
            not isinstance(self.max_actions, int)
            or isinstance(self.max_actions, bool)
            or not 1 <= self.max_actions <= MAX_ACTIONS_LIMIT
        ):
            raise ValueError(f"max_actions must be an integer between 1 and {MAX_ACTIONS_LIMIT}")
        self.initial_board = [list(row) for row in board]
        self.board = [list(row) for row in board]
        self.actions_used = 0
        self.invalid_actions = 0
        self.history = []

    @property
    def remaining_actions(self) -> int:
        return max(self.max_actions - self.actions_used, 0)

    @property
    def solved(self) -> bool:
        return is_solved(self.board)

    @property
    def exhausted(self) -> bool:
        return self.actions_used >= self.max_actions

    @property
    def terminal(self) -> bool:
        return self.solved or self.exhausted

    def execute(self, name: object, arguments: object) -> dict[str, Any]:
        """Execute one tool call and return a solution-free structured result."""

        if self.solved:
            result = self._invalid("game is already solved", consume_action=False)
            result.update(self._status())
            return result
        if self.exhausted:
            result = self._invalid("action budget exhausted", consume_action=False)
            result.update(self._status())
            return result

        self.actions_used += 1
        if not isinstance(arguments, dict):
            result = self._invalid("tool arguments must be a JSON object")
        elif name == "inspect_candidates":
            result = self._inspect_candidates(arguments)
        elif name == "place_digit":
            result = self._place_digit(arguments)
        elif name == "undo":
            result = self._undo(arguments)
        else:
            result = self._invalid(f"unknown tool: {name}")
        result.update(self._status())
        return result

    def _inspect_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            row = _strict_int(arguments.get("row"), "row")
            col = _strict_int(arguments.get("col"), "col")
            allowed = candidates(self.board, row, col)
        except ValueError as exc:
            return self._invalid(str(exc))
        return {"valid": True, "row": row, "col": col, "candidates": allowed}

    def _place_digit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            row = _strict_int(arguments.get("row"), "row")
            col = _strict_int(arguments.get("col"), "col")
            digit = _strict_int(arguments.get("digit"), "digit")
            row_index, col_index = _coordinate(row, col)
            if digit not in DIGITS:
                raise ValueError("digit must be between 1 and 9")
            if self.initial_board[row_index][col_index] != EMPTY:
                raise ValueError("given cells cannot be changed")
            if self.board[row_index][col_index] != EMPTY:
                raise ValueError("place_digit requires an empty cell")
            allowed = candidates(self.board, row, col)
            if digit not in allowed:
                raise ValueError("digit conflicts with the current row, column, or box")
        except ValueError as exc:
            return self._invalid(str(exc))
        self.board[row_index][col_index] = digit
        self.history.append((row_index, col_index))
        return {
            "valid": True,
            "row": row,
            "col": col,
            "digit": digit,
            "board": encode_board(self.board),
        }

    def _undo(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return self._invalid("undo does not accept arguments")
        if not self.history:
            return self._invalid("cannot undo at history start")
        row_index, col_index = self.history.pop()
        digit = self.board[row_index][col_index]
        self.board[row_index][col_index] = EMPTY
        return {
            "valid": True,
            "undone": {"row": row_index + 1, "col": col_index + 1, "digit": digit},
            "board": encode_board(self.board),
        }

    def _invalid(self, error: str, *, consume_action: bool = True) -> dict[str, Any]:
        if consume_action:
            self.invalid_actions += 1
        return {"valid": False, "error": error}

    def _status(self) -> dict[str, Any]:
        return {
            "solved": self.solved,
            "terminal": self.terminal,
            "actions_used": self.actions_used,
            "remaining_actions": self.remaining_actions,
            "invalid_actions": self.invalid_actions,
        }


def _search_one(board: Board, stats: SolveStats) -> bool:
    stats.visited_nodes += 1
    selected = _select_unfilled_cell(board)
    if selected is None:
        return True
    row, col, allowed = selected
    if not allowed:
        stats.backtracks += 1
        return False
    if len(allowed) > 1:
        stats.guesses += 1
    for digit in allowed:
        board[row][col] = digit
        if _search_one(board, stats):
            return True
        board[row][col] = EMPTY
    stats.backtracks += 1
    return False


def _count_solutions(board: Board, limit: int, stats: SolveStats | None) -> int:
    if stats is not None:
        stats.visited_nodes += 1
    selected = _select_unfilled_cell(board)
    if selected is None:
        return 1
    row, col, allowed = selected
    if not allowed:
        if stats is not None:
            stats.backtracks += 1
        return 0
    if stats is not None and len(allowed) > 1:
        stats.guesses += 1
    total = 0
    for digit in allowed:
        board[row][col] = digit
        total += _count_solutions(board, limit - total, stats)
        board[row][col] = EMPTY
        if total >= limit:
            break
    return total


def _select_unfilled_cell(board: Board) -> tuple[int, int, list[int]] | None:
    best: tuple[int, int, list[int]] | None = None
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            allowed = _zero_based_candidates(board, row, col)
            if not allowed:
                return row, col, []
            if best is None or len(allowed) < len(best[2]):
                best = (row, col, allowed)
                if len(allowed) == 1:
                    return best
    return best


def _zero_based_candidates(board: Board, row: int, col: int) -> list[int]:
    used = set(board[row])
    used.update(board[index][col] for index in range(BOARD_SIZE))
    box_row = (row // BOX_SIZE) * BOX_SIZE
    box_col = (col // BOX_SIZE) * BOX_SIZE
    used.update(
        board[box_row + row_offset][box_col + col_offset]
        for row_offset in range(BOX_SIZE)
        for col_offset in range(BOX_SIZE)
    )
    return sorted(DIGITS - used)


def _coordinate(row: object, col: object) -> tuple[int, int]:
    row_value = _strict_int(row, "row")
    col_value = _strict_int(col, "col")
    if not 1 <= row_value <= BOARD_SIZE or not 1 <= col_value <= BOARD_SIZE:
        raise ValueError("row and col must be between 1 and 9")
    return row_value - 1, col_value - 1


def _strict_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _is_int_cell(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= BOARD_SIZE


def _is_row(value: object) -> bool:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return False
    row = list(value)
    return len(row) == BOARD_SIZE and all(_is_int_cell(cell) for cell in row)


def _nonzero_values_are_unique(values: Sequence[int]) -> bool:
    filled = [value for value in values if value != EMPTY]
    return len(filled) == len(set(filled))
