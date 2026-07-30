"""Bounded multi-turn tool agent for Sudoku."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import TOOLS, SudokuGame  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
MAX_NO_TOOL_RESPONSES = 2

SYSTEM_PROMPT = (
    "You are a careful Sudoku agent. Maintain the board from the complete tool history. "
    "Call exactly one provided tool on every action turn. Candidate lists contain every locally legal digit, "
    "not the hidden answer. Use undo when a locally legal placement leads to a dead end. "
    "Never claim success unless a tool result says solved=true."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent Sudoku episodes and preserve raw model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Sudoku agentic example requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("Sudoku agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        grouped = await asyncio.gather(*(_run_episode(item, client) for item in items))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    game = SudokuGame(item.record["puzzle"], max_actions=int(item.record["max_actions"]))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.prompt}]
    turns = []
    consecutive_no_tool_responses = 0

    while not game.terminal:
        turn_messages = [
            *messages,
            {
                "role": "user",
                "content": (f"Action {game.actions_used + 1} of {game.max_actions}: call exactly one Sudoku tool now."),
            },
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=TOOLS,
            tool_choice="required",
            stream=False,
        )
        turn = AgentTrajectoryTurn(
            item=item,
            messages=turn_messages,
            response=response,
            tools=TOOLS,
            tool_choice="required",
        )
        assistant_message = _assistant_message(response)
        if not assistant_message["tool_calls"]:
            consecutive_no_tool_responses += 1
            logger.warning(
                "Sudoku model returned no executable tool call retry=%d/%d",
                consecutive_no_tool_responses,
                MAX_NO_TOOL_RESPONSES,
            )
            messages.extend(
                [
                    turn_messages[-1],
                    assistant_message,
                    {
                        "role": "user",
                        "content": (
                            "Your response was not an executable tool call. Do not explain or repeat text. "
                            "Call exactly one of inspect_candidates, place_digit, or undo now."
                        ),
                    },
                ]
            )
            if consecutive_no_tool_responses >= MAX_NO_TOOL_RESPONSES:
                # Preserve one negative trajectory when recovery fails, but do
                # not train on an earlier malformed response if a retry works.
                turns.append(turn)
                break
            continue
        consecutive_no_tool_responses = 0
        turns.append(turn)
        if len(assistant_message["tool_calls"]) > 1:
            logger.warning(
                "Sudoku model returned %d tool calls in one turn; calls are replayed in order and extras are penalized",
                len(assistant_message["tool_calls"]),
            )
        tool_messages = _execute_tool_calls(assistant_message, game)
        messages.extend([turn_messages[-1], assistant_message, *tool_messages])

    if game.terminal:
        finish_messages = [
            *messages,
            {
                "role": "user",
                "content": "The episode is over. Briefly summarize the observed result without calling a tool.",
            },
        ]
        finish_response = await client.chat.completions.create(
            model="policy",
            messages=finish_messages,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=finish_messages,
                response=finish_response,
            )
        )
    return turns


def _assistant_message(response) -> dict:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ],
    }


def _execute_tool_calls(assistant_message: dict, game: SudokuGame) -> list[dict]:
    """Execute every emitted call in order so each receives a matching result."""

    messages = []
    for call in assistant_message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = _parse_arguments(function.get("arguments"))
        result = game.execute(function.get("name"), arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": function.get("name"),
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return messages


def _parse_arguments(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
