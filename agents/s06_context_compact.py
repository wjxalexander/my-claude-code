#!/usr/bin/env python3
"""
PROBLEM:
The context window is finite.
A single read_file on a 1000-line file costs ~4000 tokens.
After reading 30 files and running 20 bash commands, you hit 100,000+ tokens.
The agent cannot work on large codebases without compression.
SOLUTION:
Three-layer compression pipeline so the agent can work forever:
    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.
Key insight: "The agent can forget strategically and keep working forever."
"""

import os
import json
import re
import logging
from s01_agent_loop import run_bash, client
from dotenv import load_dotenv
from pathlib import Path
from s04_subagent import CHILD_TOOLS as TOOLS, TOOL_HANDLERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."
THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
# Works like human memory: recent tool outputs are kept in full detail,
# older ones are replaced with "[Previous: used bash]" style summaries.
# This saves tokens while preserving the conversation structure.
#
# Before: tool_result #1: "(3000 chars of file content)"  <- old, wastes tokens
#         tool_result #2: "(500 chars of ls output)"       <- old, wastes tokens
#         tool_result #3: "Wrote 128 bytes"                <- recent, keep
# After:  tool_result #1: "[Previous: used read_file]"    <- compressed
#         tool_result #2: "[Previous: used bash]"          <- compressed
#         tool_result #3: "Wrote 128 bytes"                <- kept intact
def micro_compact(messages: list) -> list:
    # Step 1: Scan all messages and collect every tool_result with its position
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append(
                        (msg_idx, part_idx, part)
                    )  # Tuple of (message index, part index, tool_result)
    # Not enough results to bother compressing
    if len(tool_results) <= KEEP_RECENT:
        print(
            f"[micro_compact] {len(tool_results)} tool_results, skipping (threshold: {KEEP_RECENT})"
        )
        return messages
    print(
        f"[micro_compact] {len(tool_results)} tool_results, compressing {len(tool_results) - KEEP_RECENT} old results"
    )
    # Step 2: Build a lookup from tool_use_id -> tool_name (e.g. "toolu_abc" -> "bash")
    # so we can label compressed results with what tool produced them
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Step 3: Keep the last KEEP_RECENT results intact, compress everything older.
    # Only compress results longer than 100 chars (short ones aren't worth it).
    # Mutates the dicts in-place -- the original messages list is modified.
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            print(
                f"[micro_compact]   compressed: {tool_name} ({len(result['content'])} chars -> placeholder)"
            )
            result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
# Nuclear option: replaces the ENTIRE conversation with a 2-message summary.
# Before: messages has 30+ entries (tens of thousands of tokens)
# After:  messages has just 2 entries (a few hundred tokens)
# The full conversation is saved to disk so nothing is truly lost.
def auto_compact(messages: list) -> list:
    # Save full transcript to disk before wiping it
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask a separate LLM call to summarize the conversation
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = client.messages.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Summarize this conversation for continuity. Include: "
                "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                "Be concise but preserve critical details.\n\n" + conversation_text,
            }
        ],
        max_tokens=2000,
    )
    summary = response.content[0].text
    # Return a fresh 2-message conversation to replace all previous messages.
    # The API requires user/assistant alternation starting with user, so we need both:
    #   [0] user:      "Here's a summary of what happened so far..."
    #   [1] assistant:  "Got it, ready to continue."
    # The LLM sees this and thinks: "I know what was done, let's keep going."
    # It's like a time skip -- details are gone, but the big picture remains.
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        },
    ]


TOOLS.extend(
    [
        {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary",
                    }
                },
            },
        },
    ]
)

TOOL_HANDLERS["compact"] = (lambda **kw: "Manual compression requested.",)


# Three layers of compression, from lightest to heaviest:
#   Layer 1 (micro_compact): replace old tool_results with one-line placeholders. Runs every loop.
#   Layer 2 (auto_compact):  save transcript + summarize entire conversation. Triggered by token count.
#   Layer 3 (manual compact): same as Layer 2 but triggered by the AI calling the "compact" tool.
def agent_loop(messages: list):
    while True:
        # Layer 1: lightweight -- trim old tool results before each LLM call
        micro_compact(messages)
        # Layer 2: heavy -- if conversation is too long, nuke it and replace with summary
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            # messages[:] = ... (slice assignment) mutates the list in-place,
            # so the caller's `history` variable also gets updated.
            # Plain `messages = ...` would only reassign the local variable.
            messages[:] = auto_compact(messages)
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    # AI decided the conversation is too long and asked to compress.
                    # We still return a tool_result first (API requires it),
                    # then compress after all tool_results are appended.
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = (
                            handler(**block.input)
                            if handler
                            else f"Unknown tool: {block.name}"
                        )
                    except Exception as e:
                        output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        messages.append({"role": "user", "content": results})
        # Layer 3: same compression as Layer 2, but the AI chose to trigger it.
        # Runs AFTER tool_results are appended (so the API is happy),
        # then replaces the entire conversation with a 2-message summary.
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
