"""Outcome and process reward for Sudoku agent trajectories."""

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _load_game_module():
    """Load the sibling module without claiming the process-global name `game`."""

    module_name = "_areno_agentic_sudoku_game"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name("game.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Sudoku game module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_game = _load_game_module()
EMPTY = _game.EMPTY
SudokuGame = _game.SudokuGame
encode_board = _game.encode_board
normalize_board = _game.normalize_board


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Replay-derived metrics for one Sudoku trajectory."""

    difficulty: str
    solved: bool
    reward: float
    correct_progress: float
    actions_used: int
    invalid_actions: int
    invalid_action_rate: float
    redundant_actions: int
    redundant_action_rate: float
    multi_call_violations: int
    exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reward_fn(record) -> float:
    """Return a bounded reward by replaying the exact parsed tool calls."""

    return score_episode(
        dict(record.source_record),
        list(record.tool_calls),
        messages=list(record.messages),
    ).reward


def score_episode(
    source: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    *,
    messages: list[dict[str, Any]] | None = None,
) -> EpisodeMetrics:
    """Replay one episode and compute reward plus evaluation metrics."""

    game = SudokuGame(source["puzzle"], max_actions=int(source["max_actions"]))
    seen_state_actions: set[tuple[str, str, str]] = set()
    redundant_actions = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            game.execute(None, None)
            continue
        name = call.get("name")
        arguments = _arguments(call.get("arguments"))
        signature = (
            encode_board(game.board),
            str(name),
            json.dumps(arguments, sort_keys=True) if isinstance(arguments, dict) else repr(arguments),
        )
        if signature in seen_state_actions:
            redundant_actions += 1
        seen_state_actions.add(signature)
        game.execute(name, arguments)

    solution = encode_board(source["solution"])
    solved = encode_board(game.board) == solution
    progress = _correct_progress(game.initial_board, game.board, normalize_board(solution))
    invalid_rate = game.invalid_actions / max(game.actions_used, 1)
    redundant_rate = redundant_actions / max(game.actions_used, 1)
    multi_call_violations = _multi_call_violations(messages or [])
    reward = _episode_reward(
        solved=solved,
        progress=progress,
        actions_used=game.actions_used,
        max_actions=game.max_actions,
        invalid_rate=invalid_rate,
        redundant_rate=redundant_rate,
        multi_call_rate=multi_call_violations / max(game.actions_used, 1),
        exhausted=game.exhausted,
        has_actions=bool(tool_calls),
    )
    return EpisodeMetrics(
        difficulty=str(source.get("difficulty", "unknown")),
        solved=solved,
        reward=reward,
        correct_progress=progress,
        actions_used=game.actions_used,
        invalid_actions=game.invalid_actions,
        invalid_action_rate=invalid_rate,
        redundant_actions=redundant_actions,
        redundant_action_rate=redundant_rate,
        multi_call_violations=multi_call_violations,
        exhausted=game.exhausted,
    )


def summarize_by_difficulty(episodes: list[EpisodeMetrics]) -> dict[str, dict[str, float | int]]:
    """Aggregate solve and invalid-action rates for each observed difficulty."""

    summary: dict[str, dict[str, float | int]] = {}
    for difficulty in sorted({episode.difficulty for episode in episodes}):
        group = [episode for episode in episodes if episode.difficulty == difficulty]
        actions = sum(episode.actions_used for episode in group)
        invalid = sum(episode.invalid_actions for episode in group)
        redundant = sum(episode.redundant_actions for episode in group)
        multi_call_violations = sum(episode.multi_call_violations for episode in group)
        summary[difficulty] = {
            "episodes": len(group),
            "solve_rate": sum(episode.solved for episode in group) / len(group),
            "invalid_action_rate": invalid / max(actions, 1),
            "redundant_action_rate": redundant / max(actions, 1),
            "multi_call_violation_rate": multi_call_violations / max(actions, 1),
            "mean_reward": sum(episode.reward for episode in group) / len(group),
            "mean_correct_progress": sum(episode.correct_progress for episode in group) / len(group),
        }
    return summary


def _episode_reward(
    *,
    solved: bool,
    progress: float,
    actions_used: int,
    max_actions: int,
    invalid_rate: float,
    redundant_rate: float,
    multi_call_rate: float,
    exhausted: bool,
    has_actions: bool,
) -> float:
    if not has_actions:
        return -1.0
    efficiency = 1.0 - actions_used / max_actions
    if solved:
        score = 0.8 + 0.2 * efficiency
        score -= 0.3 * invalid_rate + 0.15 * redundant_rate + 0.1 * multi_call_rate
    else:
        score = -0.25 + 0.25 * progress
        score -= 0.1 * (actions_used / max_actions)
        score -= 0.5 * invalid_rate + 0.2 * redundant_rate + 0.1 * multi_call_rate
        if exhausted:
            score -= 0.1
    return round(max(-1.0, min(1.0, score)), 6)


def _correct_progress(initial: list[list[int]], current: list[list[int]], solution: list[list[int]]) -> float:
    empty_cells = [
        (row, col) for row in range(len(initial)) for col in range(len(initial[row])) if initial[row][col] == EMPTY
    ]
    if not empty_cells:
        return 1.0
    correct = sum(current[row][col] == solution[row][col] for row, col in empty_cells)
    return correct / len(empty_cells)


def _arguments(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _multi_call_violations(messages: list[dict[str, Any]]) -> int:
    violations = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list) and len(calls) > 1:
            violations += len(calls) - 1
    return violations
