"""Summarize Sudoku solve rate and invalid-action rate from replayable JSONL episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import score_episode, summarize_by_difficulty  # noqa: E402


def load_episodes(path: Path):
    """Load rows containing source fields plus parsed tool calls."""

    episodes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_record", row)
            tool_calls = row.get("tool_calls", [])
            messages = row.get("messages", [])
            if not isinstance(source, dict) or not isinstance(tool_calls, list):
                raise ValueError(f"line {line_number} must contain an object source and a tool_calls list")
            if not isinstance(messages, list):
                raise ValueError(f"line {line_number} messages must be a list when provided")
            episodes.append(score_episode(source, tool_calls, messages=messages))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="JSONL rows with puzzle, solution, difficulty, and tool_calls.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of a table.")
    args = parser.parse_args()

    summary = summarize_by_difficulty(load_episodes(args.episodes))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print("difficulty  episodes  solve_rate  invalid_rate  redundant_rate  mean_reward  mean_progress")
    for difficulty, metrics in summary.items():
        print(
            f"{difficulty:<10}  {metrics['episodes']:>8}  {metrics['solve_rate']:>10.3f}  "
            f"{metrics['invalid_action_rate']:>12.3f}  {metrics['redundant_action_rate']:>14.3f}  "
            f"{metrics['mean_reward']:>11.3f}  "
            f"{metrics['mean_correct_progress']:>13.3f}"
        )


if __name__ == "__main__":
    main()
