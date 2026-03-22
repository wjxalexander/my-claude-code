#!/usr/bin/env python3
"""
s04_subagent.py - Subagents
Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.
    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.
Key insight: "Process isolation gives context isolation for free."

How It Works:
1. The parent gets a task tool. The child gets all base tools except task (no recursive spawning).
2.The subagent starts with messages=[] and runs its own loop. Only the final text returns to the parent

"""

import os
import subprocess
from pathlib import Path
from s01_agent_loop import run_bash, client
from dotenv import load_dotenv
from s03_todo_write import run_bash, run_read, run_write, run_edit

load_dotenv(override=True)
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# Shared tool handlers used by both parent and subagent.
# The subagent reuses the same handlers but does NOT get the "task" tool,
# preventing recursive subagent spawning (child can't spawn grandchild).
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


# -- Subagent: fresh context, filtered tools, summary-only return --
# This is the key idea: the subagent gets a BLANK conversation (messages=[]).
# It shares the filesystem with the parent (can read/write the same files),
# but has NO memory of the parent's conversation history.
# When done, only the final text summary is returned to the parent.
# The subagent's entire internal conversation is thrown away.
def run_subagent(prompt: str) -> str:
    # Fresh context -- the subagent knows NOTHING about the parent's conversation.
    # It only sees this one prompt.
    sub_messages = [{"role": "user", "content": prompt}]
    count = 0
    # Safety limit: prevent infinite loops (subagent can run at most 30 rounds)
    for _ in range(30):
        count += 1
        print(f"Subagent round {count}")
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if not any(block.type == "tool_use" for block in response.content):
            break
        # Same tool execution pattern as parent, but scoped to the subagent's own loop
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"Unknown tool: {block.name}"
                )
                print(f"    subagent> {block.name}: {str(output)[:120]}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output)[:50000],
                    }
                )
        sub_messages.append({"role": "user", "content": results})
    # Only the final text returns to the parent.
    # The parent sees a short summary, NOT the full chain of tool calls.
    # This keeps the parent's context clean and focused.
    return (
        "".join(b.text for b in response.content if hasattr(b, "text"))
        or "(no summary)"
    )


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short description of the task",
                },
            },
            "required": ["prompt"],
        },
    },
]


# -- Parent agent loop --
# The parent can use all base tools AND the "task" tool.
# When the parent calls "task", it spawns a subagent via run_subagent().
# The subagent does the work in isolation and returns only a summary.
# This way the parent's context stays clean -- it doesn't get polluted
# with dozens of tool calls from the subtask.
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if not any(block.type == "tool_use" for block in response.content):
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "task":
                    # Spawn a subagent -- it gets a fresh messages=[] and works independently.
                    # Only the final summary comes back; the subagent's full conversation is discarded.
                    desc = block.input.get("description", "subtask")
                    print(
                        f"\033[33m> spawning subagent ({desc}): {block.input['prompt'][:80]}\033[0m"
                    )
                    output = run_subagent(block.input["prompt"])
                    print(f"\033[32m> subagent returned: {str(output)[:200]}\033[0m")
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = (
                        handler(**block.input)
                        if handler
                        else f"Unknown tool: {block.name}"
                    )
                    print(f"> {block.name}: {str(output)[:200]}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
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
