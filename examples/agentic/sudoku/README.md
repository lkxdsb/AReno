# Agentic Sudoku Example

This example trains a policy to solve standard 9x9 Sudoku through three
stateful tools:

- `inspect_candidates(row, col)` reports every digit currently allowed by the
  row, column, and 3x3 box. It never consults or reveals the stored solution.
- `place_digit(row, col, digit)` accepts a locally legal digit in an empty,
  non-given cell.
- `undo()` removes the most recent successful placement.

Every tool call consumes one action. The environment terminates when the board
is solved or the action budget is exhausted. A locally legal placement may
still lead to a dead end; the model must detect that from later observations
and use `undo`.

## Data contract

Records contain an 81-character puzzle and solution, with `0` representing an
empty cell:

```json
{
  "id": "sudoku-train-easy-00001",
  "split": "train",
  "puzzle": "000064079002350400...",
  "solution": "531864279972351486...",
  "puzzle_hash": "d6c4...",
  "solution_hash": "af35...",
  "difficulty": "easy",
  "difficulty_method": "clue_count_and_uniqueness_search_v2",
  "clue_count": 40,
  "solver_search_nodes": 42,
  "solver_search_overhead": 0,
  "solver_guesses": 0,
  "solver_backtracks": 0,
  "max_actions": 96
}
```

The loader validates board shape, solution validity, preserved givens,
positive bounded action budget, generated metadata, and exactly one puzzle
solution before model initialization. The stored solution remains in
`source_record` for reward verification but is omitted from the model prompt
and every tool result.

## Generate deterministic splits

Use different seeds for each frozen split:

```bash
python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/sudoku-train.jsonl \
  --count 3000 --seed 2026 --split train

python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/sudoku-validation.jsonl \
  --count 600 --seed 2027 --split validation

python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/sudoku-test.jsonl \
  --count 600 --seed 2028 --split test
```

Generation uses randomized MRV backtracking to construct completed grids, then
removes clues only while uniqueness is preserved. Each accepted puzzle is
checked by a deterministic MRV solver that counts up to two solutions.

Difficulty is a reproducible computational label, not a human rating:

| Level | Clues | Uniqueness-search overhead |
| --- | ---: | ---: |
| easy | 40 | 0 extra nodes |
| medium | 34 | 5-99 extra nodes |
| hard | 28 | at least 100 extra nodes |

Search overhead is `visited_nodes - (empty_cells + 1)`. It measures work beyond
the direct no-branch solution path while proving uniqueness. The generator
stores the observed counters, and the loader recalculates them.

Audit split declarations, exact duplicate boards and solutions, hashes, givens,
solution validity, and uniqueness before training:

```bash
python examples/agentic/sudoku/audit_dataset.py \
  /tmp/sudoku-train.jsonl \
  /tmp/sudoku-validation.jsonl \
  /tmp/sudoku-test.jsonl \
  --verify-solutions
```

The audit detects exact leakage. It does not claim to identify every puzzle
related by row, column, digit, rotation, or reflection symmetries; see
[`ADVERSARIAL_REVIEW.md`](ADVERSARIAL_REVIEW.md).

## Inspect normalized records

```bash
python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path /tmp/sudoku-train.jsonl \
  --loader examples/agentic/sudoku/dataset_loader.py \
  --algo gspo
```

Do not start training unless the inspection reports `"ok": true`.

## Train

Full training requires a supported CUDA environment:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path /tmp/sudoku-train.jsonl \
  --dataset-loader-fn examples/agentic/sudoku/dataset_loader.py \
  --reward-fn-path examples/agentic/sudoku/reward.py \
  --agent-fn examples/agentic/sudoku/run_agent.py \
  --algo gspo --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --mini-bs 1 \
  --max-running-prompts 2 \
  --max-new-tokens 64 --max-context-len 16384 \
  --attn-backend native \
  --max-steps 1
```

Start with a small generated dataset and one real trainer step before
increasing the workload. Sudoku episodes can contain many model requests, so
dataset row count is not the same as trajectory cost. A reproducible Kaggle
checklist and staged commands are in [`KAGGLE.md`](KAGGLE.md).

## Reward and anti-gaming rules

`reward.py` replays parsed tool calls from the original puzzle instead of
trusting model text. The hidden solution is used only after the episode to
verify the final board and count correct placements.

- A solved trajectory receives the dominant outcome reward plus a bounded
  efficiency term.
- An unfinished trajectory cannot receive a positive reward merely by placing
  a few correct digits and stopping early.
- Unfinished trajectories pay a small action-cost penalty, so repeatedly
  inspecting distinct cells is not reward-neutral.
- Invalid calls, repeated calls against the same board state, and multiple tool
  calls in one assistant turn are penalized.
- An exhausted unfinished episode receives an additional penalty.
- A trajectory with no tool call receives `-1.0`.

All final rewards are clipped to `[-1, 1]`. The weights are explicit heuristics
and should be monitored on real rollouts rather than treated as universally
optimal.

## Evaluate recorded episodes

`evaluate.py` accepts JSONL rows containing either the source fields directly
or a `source_record` object, plus parsed `tool_calls`. Include `messages` to
measure multiple tool calls per assistant turn:

```json
{
  "source_record": {
    "puzzle": "000064079002350400...",
    "solution": "531864279972351486...",
    "difficulty": "easy",
    "max_actions": 96
  },
  "tool_calls": [
    {
      "name": "place_digit",
      "arguments": "{\"row\":1,\"col\":1,\"digit\":5}"
    }
  ],
  "messages": []
}
```

Summarize solve rate, invalid-action rate, redundant-action rate, reward, and
correct progress by difficulty:

```bash
python examples/agentic/sudoku/evaluate.py /tmp/sudoku-episodes.jsonl
python examples/agentic/sudoku/evaluate.py /tmp/sudoku-episodes.jsonl --json
```

Validation and test files must remain frozen and must not be passed to
`areno train`. Evaluation needs replayable episode JSONL exported from the
rollout pipeline; dataset inspection alone is not model evaluation.

## CPU validation

```bash
pytest -q tests/test_agentic_sudoku_example_cpu.py
ruff check examples/agentic/sudoku tests/test_agentic_sudoku_example_cpu.py
```
