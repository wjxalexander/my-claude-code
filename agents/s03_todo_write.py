#!/usr/bin/env python3

# 多步任务中, 模型会丢失进度 -- 重复做过的事、跳步、跑偏。对话越长越严重:
# 工具结果不断填满上下文, 系统提示的影响力逐渐被稀释。一个 10 步重构可能做完 1-3 步就开始即兴发挥,
# 因为 4-10 步已经被挤出注意力了。

"""
problem: On multi-step tasks, the model loses track.
It repeats work, skips steps, or wanders off.
 Long conversations make this worse -- the system prompt fades as tool results
 fill the context. A 10-step refactoring might complete steps 1-3,
 then the model starts improvising because it forgot steps 4-10.

s03_todo_write.py - TodoWrite
The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>
Key insight: "The agent can track its own progress -- and I can see it."
"""

import os
import subprocess
from pathlib import Path
from s01_agent_loop import run_bash, client
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(override=True)
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()

SYSTEM = f"""You are a coding agent at {WORKDIR}.
ALWAYS call the todo tool FIRST to break down the task before doing anything else.
Mark in_progress before starting each step, completed when done. Update todos after each step.
Prefer tools over prose."""


# -- TodoManager: an external "notepad" that the LLM writes to --
# The LLM itself decides when to call the todo tool (based on the system prompt).
# TodoManager just stores and validates whatever the LLM sends.
# This acts as a structured state machine outside the LLM's context,
# so the model can always see its current progress without guessing.
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        # Called when the LLM sends a todo tool_use. Validates + stores items,
        # then returns render() so the LLM sees its updated checklist.
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        # Overwrite the old list entirely -- the LLM always sends the full state
        self.items = validated
        # Return the rendered checklist as the tool_result text.
        # The LLM reads this to know its current progress.
        return self.render()

    def render(self) -> str:
        # Renders items into a human-readable checklist, e.g.:
        #   [x] #1: Create directory
        #   [>] #2: Write utils.py
        #   [ ] #3: Write tests
        #   (1/3 completed)
        # This text goes back to the LLM as tool_result.
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[
                item["status"]
            ]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


# Global singleton -- one notepad shared across the entire agent session.
# Data flow: LLM calls "todo" -> lambda routes to TODO.update() -> render() -> tool_result back to LLM
TODO = TodoManager()


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # The LLM decides to call "todo" on its own. Our code just executes
    # TODO.update() and returns the rendered checklist back to the LLM.
    "todo": lambda **kw: TODO.update(kw["items"]),
}
# Tool definitions sent to the API so the LLM knows what tools are available.
# The LLM reads these schemas and decides which tool to call and with what args.
TOOLS = [
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
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]


# -- Agent loop with nag reminder injection --
# The loop sends messages to the LLM and processes its tool calls.
# Three participants are talking in this loop:
#   1. The real user    -- only speaks once at the beginning (the task)
#   2. The LLM (Claude) -- decides which tools to call, including "todo"
#   3. This code        -- executes tools, returns results as "user" messages,
#                          and sneaks in <reminder> nudges when needed
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # Execute each tool the LLM requested and collect results
        results = []
        used_todo = False
        # Send conversation history + tool definitions to the LLM.
        # The LLM sees TOOLS and decides which ones to call (if any).
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # Append the LLM's response as an assistant message
        messages.append({"role": "assistant", "content": response.content})
        # If the LLM didn't call any tools, the task is done
        if not any(block.type == "tool_use" for block in response.content):
            return

        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = (
                        handler(**block.input)
                        if handler
                        else f"Unknown tool: {block.name}"
                    )
                except Exception as e:
                    output = f"Error: {e}"
                print(f"tool output> {block.name}: {str(output)[:200]}")
                # Every tool_use MUST have a matching tool_result, or the API returns 400
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
                if block.name == "todo":
                    used_todo = True
        # Track how many rounds the LLM has gone without calling todo
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        # Nag reminder: if the LLM hasn't updated todos for 3+ rounds,
        # sneak a <reminder> into the tool results (disguised as a user message).
        # The LLM sees this and thinks the user told it to update todos.
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if last["role"] == "user" and isinstance(last.get("content"), list):
                results.insert(
                    0,
                    {
                        "type": "text",
                        "text": "<reminder>Update your todos.</reminder>",
                    },
                )
        # Send tool results back as a "user" message.
        # The real user didn't say this -- our code fabricates it.
        # The LLM reads these results and decides what to do next.
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # history is shared across turns -- the LLM sees the full conversation
    # each time, which is how it "remembers" previous steps.
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # This is the ONLY real user message. Everything else in the
        # conversation is either the LLM talking or our code faking it.
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
