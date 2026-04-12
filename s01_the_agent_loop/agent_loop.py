#!/usr/bin/env python3
import os
import subprocess
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("LITELLM_MODEL", "dashscope/qwen-turbo")

SYSTEM_PROMPT = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]


@dataclass
class LoopState:
    messages: list = field(default_factory=list)
    turn_count: int = 1


def run_bash(command: str) -> str:
    if any(dangerous in command for dangerous in DANGEROUS_COMMANDS):
        return f"Error: Dangerous command blocked: {command}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def execute_tool_calls(tool_calls: list) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        func = tool_call.function
        if func.name != "bash":
            continue

        args = json.loads(func.arguments)
        command = args.get("command", "")

        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output[:200] if len(output) > 200 else output)

        results.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": output,
            "name": func.name
        })

    return results


def run_one_turn(state: LoopState) -> bool:
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *state.messages
        ],
        tools=TOOLS,
        temperature=0.7,
        max_tokens=4000,
    )

    assistant_message = response.choices[0].message
    state.messages.append({
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": assistant_message.tool_calls
    })

    if assistant_message.tool_calls:
        print(f"\n\033[32m[Model Tool Calls]\033[0m")
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            func = tool_call.function
            print(f"{i}. \033[35m{func.name}\033[0m({func.arguments})")
        print()

    if not assistant_message.tool_calls:
        return False

    results = execute_tool_calls(assistant_message.tool_calls)
    if not results:
        return False

    for result in results:
        state.messages.append({
            "role": result["role"],
            "tool_call_id": result["tool_call_id"],
            "content": result["content"],
            "name": result["name"]
        })

    state.turn_count += 1
    return True


def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass


def main():
    history = []

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)

        final_message = history[-1]
        final_text = final_message.get("content", "")
        if final_text:
            print(final_text)
        print()


if __name__ == "__main__":
    main()
