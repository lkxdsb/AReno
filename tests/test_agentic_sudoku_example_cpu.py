from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "sudoku"

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    module_name = f"agentic_sudoku_{name}_for_tests"
    previous_game = sys.modules.pop("game", None)
    previous_reward = sys.modules.pop("reward", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop(module_name, None)
        sys.modules.pop("game", None)
        sys.modules.pop("reward", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if previous_reward is not None:
            sys.modules["reward"] = previous_reward
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


def _record(**updates):
    record = {
        "id": "known",
        "puzzle": PUZZLE,
        "solution": SOLUTION,
        "difficulty": "easy",
        "max_actions": 96,
    }
    record.update(updates)
    return record


def _solution_calls():
    calls = []
    for index, (puzzle_cell, solution_cell) in enumerate(zip(PUZZLE, SOLUTION, strict=True)):
        if puzzle_cell != "0":
            continue
        row, col = divmod(index, 9)
        calls.append(
            {
                "name": "place_digit",
                "arguments": json.dumps({"row": row + 1, "col": col + 1, "digit": int(solution_cell)}),
            }
        )
    return calls


def test_candidate_query_is_rule_derived_and_solution_free():
    game = _load_module("game")

    assert game.candidates(PUZZLE, 1, 3) == [1, 2, 4]
    assert int(SOLUTION[2]) == 4
    assert game.candidates(PUZZLE, 1, 3) != [4]
    for digit in game.candidates(PUZZLE, 1, 3):
        board = game.normalize_board(PUZZLE)
        board[0][2] = digit
        assert game.is_consistent(board)


def test_solver_handles_unique_multiple_and_conflicting_boards():
    game = _load_module("game")
    stats = game.SolveStats()

    assert game.encode_board(game.solve(PUZZLE, stats=stats)) == SOLUTION
    assert game.count_solutions(PUZZLE, limit=2) == 1
    assert game.count_solutions("0" * 81, limit=2) == 2
    assert game.solve("55" + "0" * 79) is None
    assert game.count_solutions("55" + "0" * 79, limit=2) == 0
    assert stats.visited_nodes > 0


def test_game_rejects_boundaries_and_restores_state_on_undo():
    game = _load_module("game")
    env = game.SudokuGame(PUZZLE, max_actions=8)

    assert env.execute("undo", {})["error"] == "cannot undo at history start"
    assert env.execute("inspect_candidates", {"row": 0, "col": 1})["valid"] is False
    assert env.execute("place_digit", {"row": 1, "col": 1, "digit": 9})["error"] == "given cells cannot be changed"
    before = game.encode_board(env.board)
    placed = env.execute("place_digit", {"row": 1, "col": 3, "digit": 1})
    assert placed["valid"] is True
    assert placed["solved"] is False
    undone = env.execute("undo", {})
    assert undone["valid"] is True
    assert game.encode_board(env.board) == before
    assert env.actions_used == 5
    assert env.invalid_actions == 3


def test_action_budget_exhaustion_is_terminal_and_bounded():
    game = _load_module("game")
    env = game.SudokuGame(PUZZLE, max_actions=1)

    result = env.execute("inspect_candidates", {"row": 1, "col": 3})

    assert result["valid"] is True
    assert result["terminal"] is True
    assert result["remaining_actions"] == 0
    rejected = env.execute("place_digit", {"row": 1, "col": 3, "digit": 4})
    assert rejected["valid"] is False
    assert rejected["error"] == "action budget exhausted"
    assert rejected["terminal"] is True
    assert rejected["actions_used"] == 1

    malformed = game.SudokuGame(PUZZLE, max_actions=1).execute("place_digit", None)
    assert malformed["valid"] is False
    assert malformed["terminal"] is True
    assert malformed["invalid_actions"] == 1


def test_generator_is_reproducible_balanced_diverse_and_uniquely_solvable():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(6, seed=7)

    assert records == generator.generate_records(6, seed=7)
    assert [record["difficulty"] for record in records] == ["easy", "medium", "hard"] * 2
    assert len({record["puzzle"] for record in records}) == 6
    assert len({record["solution_hash"] for record in records}) == 6
    assert all(record["split"] == "train" for record in records)
    for record in records:
        assert record["clue_count"] == generator.TARGET_CLUES[record["difficulty"]]
        assert game.count_solutions(record["puzzle"], limit=2) == 1
        assert _exact_cover_solution_count(record["puzzle"], limit=2) == 1
        assert game.is_solved(record["solution"])
        assert game.clues_are_preserved(record["puzzle"], record["solution"])
        stats = game.SolveStats(
            visited_nodes=record["solver_search_nodes"],
            guesses=record["solver_guesses"],
            backtracks=record["solver_backtracks"],
        )
        assert generator._matches_difficulty(
            record["difficulty"],
            stats,
            empty_cells=record["puzzle"].count("0"),
        )
        assert record["solver_search_overhead"] == stats.visited_nodes - (record["puzzle"].count("0") + 1)


def test_curriculum_generator_produces_short_unique_reproducible_puzzles():
    game = _load_module("game")
    generator = _load_module("curriculum_generator")

    records = generator.generate_curriculum_records(6, seed=17, empty_cells=3, max_actions=8)

    assert records == generator.generate_curriculum_records(6, seed=17, empty_cells=3, max_actions=8)
    assert len({record["puzzle_hash"] for record in records}) == 6
    assert len({record["solution_hash"] for record in records}) == 6
    for record in records:
        assert record["difficulty_method"] == "curriculum_empty_cells_v1"
        assert record["curriculum_empty_cells"] == 3
        assert record["puzzle"].count("0") == 3
        assert record["max_actions"] == 8
        assert game.count_solutions(record["puzzle"], limit=2) == 1


def test_loader_validates_records_and_never_exposes_solution():
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    rows = generator.generate_records(3, seed=11)

    records = loader.load_training_dataset("unused", default_loader=lambda _: rows)

    assert len(records) == 3
    assert all(record["solution"] not in record["prompt"] for record in records)
    assert all("Candidate inspection" in record["prompt"] for record in records)

    invalid = [dict(rows[0], puzzle="0" * 81)]
    try:
        loader.load_training_dataset("unused", default_loader=lambda _: invalid)
    except ValueError as exc:
        assert "exactly one solution" in str(exc)
    else:
        raise AssertionError("multi-solution puzzle was accepted")

    tampered = [dict(rows[0], puzzle_hash="0" * 16)]
    try:
        loader.load_training_dataset("unused", default_loader=lambda _: tampered)
    except ValueError as exc:
        assert "puzzle_hash" in str(exc)
    else:
        raise AssertionError("tampered generated metadata was accepted")

    curriculum_generator = _load_module("curriculum_generator")
    curriculum = curriculum_generator.generate_curriculum_records(1, seed=23)
    loaded_curriculum = loader.load_training_dataset("unused", default_loader=lambda _: curriculum)
    assert loaded_curriculum[0]["puzzle"].count("0") == 3


def test_reward_replay_separates_success_partial_invalid_and_empty_paths():
    reward = _load_module("reward")
    source = _record()

    solved = reward.score_episode(source, _solution_calls())
    partial = reward.score_episode(source, _solution_calls()[:5])
    invalid = reward.score_episode(
        source,
        [{"name": "place_digit", "arguments": json.dumps({"row": 1, "col": 1, "digit": 9})}],
    )
    empty = reward.score_episode(source, [])

    assert solved.solved is True
    assert solved.reward > partial.reward > invalid.reward
    assert solved.correct_progress == 1.0
    assert partial.reward <= 0.0
    assert invalid.invalid_action_rate == 1.0
    assert empty.reward == -1.0

    summary = reward.summarize_by_difficulty([solved, partial, invalid])
    assert summary["easy"]["episodes"] == 3
    assert summary["easy"]["solve_rate"] == 1 / 3
    assert summary["easy"]["invalid_action_rate"] > 0


def test_reward_loads_through_runtime_file_loader():
    from areno.api.rewards import load_reward_fn

    previous_game = sys.modules.pop("game", None)
    try:
        loaded = load_reward_fn(str(EXAMPLE_DIR / "reward.py"))
        assert "game" not in sys.modules
    finally:
        if previous_game is not None:
            sys.modules["game"] = previous_game

    assert callable(loaded)


def test_reward_penalizes_repeated_state_actions_and_multi_call_turns():
    reward = _load_module("reward")
    source = _record()
    distinct_calls = [
        {"name": "inspect_candidates", "arguments": {"row": 1, "col": 3}},
        {"name": "inspect_candidates", "arguments": {"row": 1, "col": 4}},
    ]
    repeated_calls = [distinct_calls[0], distinct_calls[0]]
    normal_messages = [
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "tool_calls": [{"id": "2"}]},
    ]
    multi_messages = [
        {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
    ]

    distinct = reward.score_episode(source, distinct_calls, messages=normal_messages)
    short = reward.score_episode(source, distinct_calls[:1], messages=normal_messages[:1])
    repeated = reward.score_episode(source, repeated_calls, messages=normal_messages)
    multi = reward.score_episode(source, distinct_calls, messages=multi_messages)

    assert distinct.redundant_actions == 0
    assert distinct.reward < short.reward
    assert repeated.redundant_actions == 1
    assert repeated.reward < distinct.reward
    assert multi.multi_call_violations == 1
    assert multi.reward < distinct.reward


def test_dataset_audit_rejects_cross_split_leakage(tmp_path):
    audit = _load_module("audit_dataset")
    generator = _load_module("dataset_generator")

    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    write(train_path, generator.generate_records(3, seed=21, split="train"))
    write(test_path, generator.generate_records(3, seed=21, split="test"))

    leaked = audit.audit([train_path, test_path], verify_solutions=True)

    assert leaked["ok"] is False
    assert any("duplicate puzzle" in error for error in leaked["errors"])

    write(test_path, generator.generate_records(3, seed=22, split="test"))
    clean = audit.audit([train_path, test_path], verify_solutions=True)
    assert clean["ok"] is True


def test_evaluator_loads_replayable_rows_and_reports_by_difficulty(tmp_path):
    evaluate = _load_module("evaluate")
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        json.dumps({"source_record": _record(), "tool_calls": _solution_calls()}) + "\n",
        encoding="utf-8",
    )

    episodes = evaluate.load_episodes(path)
    summary = evaluate.summarize_by_difficulty(episodes)

    assert len(episodes) == 1
    assert summary["easy"]["solve_rate"] == 1.0
    assert summary["easy"]["invalid_action_rate"] == 0.0


def test_tool_schemas_are_closed_and_do_not_contain_solution_fields():
    game = _load_module("game")

    assert [tool["function"]["name"] for tool in game.TOOLS] == [
        "inspect_candidates",
        "place_digit",
        "undo",
    ]
    for tool in game.TOOLS:
        parameters = tool["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert "solution" not in parameters.get("properties", {})


def test_agent_preserves_tool_order_and_stops_at_budget_without_fabricating_calls():
    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    def response(*, tool_name=None, arguments=None, content=None):
        calls = []
        if tool_name is not None:
            calls = [
                SimpleNamespace(
                    id="call-1",
                    type="function",
                    function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments or {})),
                )
            ]
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FakeCompletions(
        [
            response(tool_name="inspect_candidates", arguments={"row": 1, "col": 3}),
            response(content="The action budget was exhausted."),
        ]
    )
    item = SimpleNamespace(prompt="solve", record=_record(max_actions=1))
    turns = asyncio.run(run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=completions))))

    assert len(turns) == 2
    finish_messages = completions.messages[1]
    assert finish_messages[-3]["role"] == "assistant"
    assert finish_messages[-2]["role"] == "tool"
    assert finish_messages[-2]["tool_call_id"] == "call-1"
    assert json.loads(finish_messages[-2]["content"])["terminal"] is True
    assert finish_messages[-1]["role"] == "user"

    failed = FakeCompletions(
        [
            response(content="I will not call a tool."),
            response(content="I still will not call a tool."),
        ]
    )
    failed_turns = asyncio.run(run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=failed))))
    assert len(failed_turns) == 1
    assert len(failed.messages) == 2
    assert "not an executable tool call" in failed.messages[1][-2]["content"]
    assert failed.messages[0][-1]["role"] == "user"

    recovered = FakeCompletions(
        [
            response(content="I should inspect."),
            response(tool_name="inspect_candidates", arguments={"row": 1, "col": 3}),
            response(content="Episode complete."),
        ]
    )
    recovered_turns = asyncio.run(
        run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=recovered)))
    )
    assert len(recovered_turns) == 2
    assert recovered_turns[0].response.choices[0].message.tool_calls


def test_agent_episode_stops_when_the_environment_reports_solved():
    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    responses = []
    for index, call in enumerate(_solution_calls()):
        responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id=f"call-{index}",
                                    type="function",
                                    function=SimpleNamespace(
                                        name=call["name"],
                                        arguments=call["arguments"],
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        )
    responses.append(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Solved.", tool_calls=[]),
                )
            ]
        )
    )
    completions = FakeCompletions(responses)
    item = SimpleNamespace(prompt="solve", record=_record())

    turns = asyncio.run(run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=completions))))

    assert len(turns) == len(_solution_calls()) + 1
    finish_messages = completions.messages[-1]
    final_tool_result = json.loads(finish_messages[-2]["content"])
    assert final_tool_result["solved"] is True
    assert final_tool_result["terminal"] is True


def _exact_cover_solution_count(puzzle: str, *, limit: int) -> int:
    """Independent Algorithm X oracle for generator cross-checks."""

    row_constraints = {}
    columns = {}
    for row in range(9):
        for col in range(9):
            given = int(puzzle[row * 9 + col])
            digits = [given] if given else range(1, 10)
            for digit in digits:
                candidate = (row, col, digit)
                constraints = (
                    ("cell", row, col),
                    ("row", row, digit),
                    ("col", col, digit),
                    ("box", (row // 3) * 3 + col // 3, digit),
                )
                row_constraints[candidate] = constraints
                for constraint in constraints:
                    columns.setdefault(constraint, set()).add(candidate)

    def search(active_columns):
        if not active_columns:
            return 1
        constraint = min(active_columns, key=lambda key: len(active_columns[key]))
        total = 0
        for candidate in active_columns[constraint]:
            covered_constraints = row_constraints[candidate]
            conflicting_rows = set()
            for covered in covered_constraints:
                conflicting_rows.update(active_columns.get(covered, ()))
            next_columns = {}
            dead_end = False
            for key, rows in active_columns.items():
                if key in covered_constraints:
                    continue
                remaining = rows - conflicting_rows
                if not remaining:
                    dead_end = True
                    break
                next_columns[key] = remaining
            if not dead_end:
                total += search(next_columns)
                if total >= limit:
                    return total
        return total

    return search(columns)
